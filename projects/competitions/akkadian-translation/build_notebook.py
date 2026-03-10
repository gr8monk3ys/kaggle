#!/usr/bin/env python3
"""Build a competition-compliant Akkadian retrieval baseline notebook."""

import os as _os
import sys as _sys
from pathlib import Path


def _find_repo_root(start_dir: str) -> str:
    current = _os.path.abspath(start_dir)
    while True:
        if _os.path.exists(_os.path.join(current, "manage.sh")) and _os.path.isdir(
            _os.path.join(current, "kaggle_portfolio")
        ):
            return current
        parent = _os.path.dirname(current)
        if parent == current:
            return _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        current = parent


_sys.path.insert(0, _find_repo_root(_os.path.dirname(_os.path.abspath(__file__))))

from kaggle_portfolio.shared.build_utils import code, md, write_notebook


cells: list[dict] = []

cells.append(
    md(
        """# Akkadian Translation Retrieval Baseline
**Competition:** [Deep Past Challenge - Translate Akkadian to English](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)
**Runtime:** competition-safe baseline with internet disabled

## What this notebook does
- loads the real competition files from `/kaggle/input`
- builds a lightweight transliteration retrieval index from the training set
- matches each test tablet to the closest training translation
- splits the retrieved English text across the requested line ranges
- writes a valid `submission.csv`

This baseline avoids external model downloads and still uses the actual training pairs, which makes it a better first code-competition submission than copying the sample file."""
    )
)

cells.append(md("## 1. Setup"))

cells.append(
    code(
        """from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

candidates = [
    Path('/kaggle/input/deep-past-initiative-machine-translation'),
    Path('/tmp/akkadian-live/extracted'),
    Path('.'),
]
data_dir = next((path for path in candidates if (path / 'train.csv').exists()), None)
if data_dir is None:
    raise FileNotFoundError('Competition files not found in /kaggle/input or local fallback paths')

print(f'Using data directory: {data_dir}')"""
    )
)

cells.append(md("## 2. Load Competition Files"))

cells.append(
    code(
        """train = pd.read_csv(data_dir / 'train.csv')
test = pd.read_csv(data_dir / 'test.csv')
sample = pd.read_csv(data_dir / 'sample_submission.csv')

print('Train shape:', train.shape)
print('Test shape :', test.shape)
print('Sample shape:', sample.shape)
print('\\nTrain columns:', list(train.columns))
print('Test columns :', list(test.columns))
print('Sample columns:', list(sample.columns))

train.head(2)"""
    )
)

cells.append(md("## 3. Build Retrieval Baseline"))

cells.append(
    code(
        """def normalize_transliteration(text: str) -> str:
    return ' '.join(str(text).lower().split())


def split_translation_by_weights(text: str, weights: list[int]) -> list[str]:
    words = str(text).split()
    if not words:
        return ['' for _ in weights]

    total_weight = sum(weights) or len(weights)
    chunks: list[str] = []
    position = 0

    for idx, weight in enumerate(weights):
        remaining_words = len(words) - position
        remaining_groups = len(weights) - idx

        if idx == len(weights) - 1:
            take = remaining_words
        else:
            take = max(1, round(len(words) * weight / total_weight))
            take = min(take, remaining_words - (remaining_groups - 1))

        chunk_words = words[position : position + take]
        chunks.append(' '.join(chunk_words).strip())
        position += take

    return chunks


sort_cols = [col for col in ['oare_id', 'line_start', 'line_number', 'line_no'] if col in train.columns]
train_ordered = train.sort_values(sort_cols) if sort_cols else train.copy()

train_index = (
    train_ordered.groupby('oare_id', as_index=False)
    .agg(
        transliteration=('transliteration', lambda values: ' '.join(map(str, values))),
        translation=('translation', lambda values: ' '.join(map(str, values))),
    )
)
train_index['line_count'] = train_ordered.groupby('oare_id').size().values
train_index['translit_norm'] = train_index['transliteration'].map(normalize_transliteration)
train_index['token_count'] = train_index['translit_norm'].str.split().str.len().clip(lower=1)

char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), min_df=1)
word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
char_matrix = char_vectorizer.fit_transform(train_index['translit_norm'])
word_matrix = word_vectorizer.fit_transform(train_index['translit_norm'])

predictions: dict[int, str] = {}
matches: list[dict] = []

ordered_test = test.sort_values(['text_id', 'line_start']).copy()
for text_id, group in ordered_test.groupby('text_id', sort=False):
    combined_translit = ' '.join(group['transliteration'].astype(str).tolist())
    combined_norm = normalize_transliteration(combined_translit)
    char_similarity = linear_kernel(char_vectorizer.transform([combined_norm]), char_matrix)[0]
    word_similarity = linear_kernel(word_vectorizer.transform([combined_norm]), word_matrix)[0]
    requested_lines = len(group)
    requested_tokens = max(1, len(combined_norm.split()))

    line_bonus = 1 - (
        (train_index['line_count'] - requested_lines).abs()
        / train_index['line_count'].clip(lower=requested_lines)
    )
    token_bonus = 1 - (
        (train_index['token_count'] - requested_tokens).abs()
        / train_index['token_count'].clip(lower=requested_tokens)
    )
    similarity = 0.55 * char_similarity + 0.35 * word_similarity + 0.10 * line_bonus.fillna(0).to_numpy()

    candidate_scores = pd.DataFrame(
        {
            'idx': range(len(train_index)),
            'base_score': similarity,
        }
    ).nlargest(3, 'base_score')

    reranked: list[tuple[int, float]] = []
    for idx in candidate_scores['idx']:
        idx = int(idx)
        rerank_score = similarity[idx] + 0.05 * float(token_bonus.iloc[idx])
        reranked.append((idx, rerank_score))

    best_idx = max(reranked, key=lambda item: item[1])[0]
    best_row = train_index.iloc[best_idx]

    weights = (
        group['line_end'].fillna(group['line_start']).astype(int)
        - group['line_start'].astype(int)
        + 1
    ).clip(lower=1).tolist()
    chunks = split_translation_by_weights(best_row['translation'], weights)

    for row_idx, chunk in zip(group.index.tolist(), chunks):
        predictions[row_idx] = chunk or best_row['translation']

    matches.append(
        {
            'text_id': text_id,
            'similarity': float(similarity[best_idx]),
            'matched_oare_id': best_row['oare_id'],
            'matched_translation_preview': best_row['translation'][:140],
        }
    )

match_frame = pd.DataFrame(matches).sort_values('similarity', ascending=False)
match_frame.head(10)"""
    )
)

cells.append(md("## 4. Write Submission"))

cells.append(
    code(
        """fallback_submission = sample.copy()
submission = pd.DataFrame(
    {
        'id': test['id'],
        'translation': [
            predictions.get(idx, fallback_submission.iloc[idx]['translation'])
            for idx in range(len(test))
        ],
    }
)
submission.to_csv('submission.csv', index=False)

print('submission.csv written')
print(submission.head().to_string(index=False))"""
    )
)

cells.append(
    md(
        """## Notes
This baseline uses nearest-neighbor retrieval over transliterated tablets. It is still simple, but it uses the actual training pairs and grouped test structure, which makes it a more realistic first competition model."""
    )
)

write_notebook(cells, __file__, "akkadian_submission_baseline.ipynb")
