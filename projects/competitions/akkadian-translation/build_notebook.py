#!/usr/bin/env python3
"""Build a competition-compliant Akkadian sentence-match baseline notebook."""

import os as _os
import sys as _sys


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
        """# Akkadian Translation Sentence-Match Baseline
**Competition:** [Deep Past Challenge - Translate Akkadian to English](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)
**Runtime:** competition-safe baseline with internet disabled

## What this notebook does
- loads the real competition files plus the auxiliary published-text tables shipped with the competition
- finds the closest published Akkadian text to the hidden test tablet
- reconstructs the four competition output rows from line-numbered sentence translations
- falls back to train-set retrieval if no sentence-aligned published match is available
- writes a valid `submission.csv`

This is still a lightweight baseline, but it uses the strongest structure Kaggle gives us: the published-text corpus and sentence alignments."""
    )
)

cells.append(md("## 1. Setup"))

cells.append(
    code(
        """from pathlib import Path
import re

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
test = pd.read_csv(data_dir / 'test.csv').sort_values(['line_start', 'line_end']).reset_index(drop=True)
sample = pd.read_csv(data_dir / 'sample_submission.csv').sort_values('id').reset_index(drop=True)
published = pd.read_csv(data_dir / 'published_texts.csv')
sentences = pd.read_csv(data_dir / 'Sentences_Oare_FirstWord_LinNum.csv')

print('Train shape      :', train.shape)
print('Test shape       :', test.shape)
print('Published texts  :', published.shape)
print('Sentence matches :', sentences.shape)
print('\\nTest rows:')
print(test.to_string(index=False))"""
    )
)

cells.append(md("## 3. Match Against Published Texts"))

cells.append(
    code(
        """def normalize_transliteration(text: str) -> str:
    normalized = str(text or '').lower()
    for old, new in {
        '…': ' ',
        '...': ' ',
        '„': ' ',
        '“': ' ',
        '”': ' ',
        '"': ' ',
        "'": ' ',
        '`': ' ',
        '´': ' ',
        '{': ' ',
        '}': ' ',
        '(': ' ',
        ')': ' ',
        '[': ' ',
        ']': ' ',
        '/': ' ',
        '\\\\': ' ',
        ',': ' ',
        '.': ' ',
        ';': ' ',
        ':': ' ',
        '!': ' ',
        '?': ' ',
    }.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r'<[^>]+>', ' ', normalized)
    return ' '.join(normalized.split())


def best_match(corpus: pd.Series, query: str) -> tuple[int, float]:
    corpus_norm = corpus.fillna('').map(normalize_transliteration)
    query_norm = normalize_transliteration(query)
    char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), min_df=1)
    word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
    char_matrix = char_vec.fit_transform(corpus_norm)
    word_matrix = word_vec.fit_transform(corpus_norm)
    char_score = linear_kernel(char_vec.transform([query_norm]), char_matrix)[0]
    word_score = linear_kernel(word_vec.transform([query_norm]), word_matrix)[0]
    scores = 0.7 * char_score + 0.3 * word_score
    best_idx = int(scores.argmax())
    return best_idx, float(scores[best_idx])


def display_name_candidates(row: pd.Series) -> list[str]:
    names: list[str] = []
    for value in [row.get('label', ''), row.get('aliases', ''), row.get('note', '')]:
        raw = str(value or '').strip()
        if not raw or raw.lower() == 'nan':
            continue
        for part in [item.strip() for item in raw.split('|')]:
            if not part:
                continue
            names.append(part)
            stripped = re.sub(r'^cuneiform\\s+(tablet|envelope)\\s+', '', part, flags=re.IGNORECASE).strip()
            if stripped and stripped != part:
                names.append(stripped)
    deduped = []
    seen = set()
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


query = ' '.join(test['transliteration'].astype(str))
published_idx, published_score = best_match(published['transliteration'], query)
published_row = published.iloc[published_idx]
display_names = sentences['display_name'].astype(str).str.strip()
candidate_rows = pd.DataFrame()
for name in display_name_candidates(published_row):
    matched = sentences.loc[display_names == name]
    if len(matched) > len(candidate_rows):
        candidate_rows = matched

candidate_rows = (
    candidate_rows[['line_number', 'translation']]
    .dropna(subset=['line_number', 'translation'])
    .sort_values('line_number')
    .reset_index(drop=True)
)

print('Best published match score:', round(published_score, 5))
print('Best published label      :', published_row.get('label'))
print('Sentence rows found       :', len(candidate_rows))
candidate_rows.head(10)"""
    )
)

cells.append(md("## 4. Build Hybrid Submission"))

cells.append(
    code(
        """def assign_sentences_to_rows(test_rows: pd.DataFrame, sentence_rows: pd.DataFrame) -> list[str]:
    predictions: list[str] = []
    ordered_test = test_rows.sort_values(['line_start', 'line_end']).reset_index(drop=True)
    ordered_sentences = sentence_rows.sort_values('line_number').reset_index(drop=True)
    for idx, row in ordered_test.iterrows():
        start = int(row['line_start'])
        next_start = int(ordered_test.loc[idx + 1, 'line_start']) if idx + 1 < len(ordered_test) else None
        if next_start is None:
            mask = ordered_sentences['line_number'] >= start
        else:
            mask = (ordered_sentences['line_number'] >= start) & (ordered_sentences['line_number'] < next_start)
        predictions.append(' '.join(ordered_sentences.loc[mask, 'translation'].astype(str)).strip())
    return predictions


def split_translation_by_rows(text: str, test_rows: pd.DataFrame) -> list[str]:
    ordered = test_rows.sort_values(['line_start', 'line_end']).reset_index(drop=True)
    weights = (ordered['line_end'].astype(int) - ordered['line_start'].astype(int) + 1).clip(lower=1).tolist()
    words = str(text or '').split()
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
        chunks.append(' '.join(words[position : position + take]).strip())
        position += take
    return chunks


def train_retrieval_predictions() -> tuple[list[str], float]:
    best_idx, best_score = best_match(train['transliteration'], query)
    best_translation = str(train.iloc[best_idx]['translation'])
    preds = split_translation_by_rows(best_translation, test)
    fallback = sample['translation'].astype(str).tolist()
    preds = [pred.strip() or fallback[idx] for idx, pred in enumerate(preds)]
    return preds, best_score


sentence_predictions = assign_sentences_to_rows(test, candidate_rows) if not candidate_rows.empty else []
train_predictions, train_score = train_retrieval_predictions()
coverage = (
    sum(1 for pred in sentence_predictions if pred.strip()) / len(test)
    if len(sentence_predictions) == len(test)
    else 0.0
)
published_decision_score = min(1.0, published_score + 0.15 * coverage)

if coverage == 1.0 and published_score >= 0.6 and published_decision_score >= train_score:
    chosen_model = 'published_sentence_match'
    predictions = sentence_predictions
    chosen_score = published_decision_score
else:
    chosen_model = 'train_retrieval'
    predictions = train_predictions
    chosen_score = train_score

diagnostics = pd.DataFrame(
    [
        {'model': 'published_sentence_match', 'decision_score': round(published_decision_score, 5)},
        {'model': 'train_retrieval', 'decision_score': round(train_score, 5)},
    ]
)
print('Chosen model:', chosen_model)
print('Chosen score:', round(chosen_score, 5))
diagnostics"""
    )
)

cells.append(md("## 5. Write Submission"))

cells.append(
    code(
        """submission = pd.DataFrame({'id': test['id'], 'translation': predictions})
submission.to_csv('submission.csv', index=False)

print('submission.csv written')
print(submission.to_string(index=False))"""
    )
)

cells.append(
    md(
        """## Notes
This notebook is intentionally conservative: it uses only competition-provided files and prefers exact or near-exact published-text sentence matches before falling back to generic train-set retrieval. That makes it a good code-competition baseline when a hidden test tablet already exists in the auxiliary corpus."""
    )
)

write_notebook(cells, __file__, "akkadian_submission_baseline.ipynb")
