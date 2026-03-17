#!/usr/bin/env python3
"""Build a competition-compliant Akkadian ByT5 seq2seq submission notebook.

Upgrades the original TF-IDF retrieval baseline to ByT5-base fine-tuning
with beam search inference, keeping TF-IDF as a safety fallback.
"""

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

# ── Title ──────────────────────────────────────────────────────────────

cells.append(
    md(
        """# Akkadian Translation: ByT5 Seq2Seq + Retrieval Hybrid
**Competition:** [Deep Past Challenge - Translate Akkadian to English](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)

## Approach
1. **ByT5-base fine-tuning** on competition training data (character-level seq2seq)
2. **Beam search inference** (num_beams=5) for translation quality
3. **TF-IDF retrieval fallback** using published-text corpus for safety
4. **Hybrid selection**: uses ByT5 predictions when non-empty, TF-IDF otherwise

### Why ByT5?
Akkadian transliteration is syllabic (e.g. `a-na A-shur qi2-bi2-ma`) with hyphens,
Sumerograms, and morphological complexity. ByT5 operates at the **byte level**,
so it handles rare subword patterns and character-level structure natively —
no tokenization artifacts from unseen syllables."""
    )
)

# ── Section 1: Setup ──────────────────────────────────────────────────

cells.append(md("## 1. Setup"))

cells.append(
    code(
        """import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"""
    )
)

# ── Section 2: Load Competition Data ──────────────────────────────────

cells.append(md("## 2. Load Competition Data"))

cells.append(
    code(
        """candidates = [
    Path('/kaggle/input/deep-past-initiative-machine-translation'),
    Path('/tmp/akkadian-live/extracted'),
    Path('.'),
]
data_dir = next((p for p in candidates if (p / 'train.csv').exists()), None)
if data_dir is None:
    raise FileNotFoundError('Competition files not found')
print(f'Data directory: {data_dir}')

train = pd.read_csv(data_dir / 'train.csv')
test = pd.read_csv(data_dir / 'test.csv')
sample = pd.read_csv(data_dir / 'sample_submission.csv')

# Optional auxiliary files (for TF-IDF fallback)
def load_optional(path, cols):
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype='object')
    return df

published = load_optional(data_dir / 'published_texts.csv',
                          ['transliteration', 'label', 'aliases', 'note'])
sentences = load_optional(data_dir / 'Sentences_Oare_FirstWord_LinNum.csv',
                          ['display_name', 'line_number', 'translation'])

print(f'Train: {train.shape}  |  Test: {test.shape}')
print(f'Published texts: {published.shape}  |  Sentences: {sentences.shape}')
print(f'\\nTrain columns: {list(train.columns)}')
print(f'Test columns: {list(test.columns)}')
print(f'\\nSample train row:')
print(train.iloc[0].to_string())"""
    )
)

# ── Section 3: Load ByT5 Model ────────────────────────────────────────

cells.append(md("## 3. Load ByT5-base Model"))

cells.append(
    code(
        """# Kaggle Models paths (offline, no internet needed)
MODEL_PATHS = [
    '/kaggle/input/byt5-base/transformers/default/1',
    '/kaggle/input/byt5/transformers/base/1',
    '/kaggle/input/byt5-base',
    '/kaggle/input/google-byt5-base',
    'google/byt5-base',  # online fallback for local testing
]

model = None
tokenizer = None
model_path = None

for path in MODEL_PATHS:
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        model_path = path
        break
    except Exception:
        continue

if model is not None:
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded model from: {model_path}')
    print(f'Parameters: {total_params:,}')
    print(f'Size: ~{total_params * 2 / 1e9:.2f} GB (FP16)')
else:
    print('WARNING: ByT5 model not found. Will use TF-IDF fallback only.')
    print('To fix: Add google/byt5-base as a Model source in notebook settings.')"""
    )
)

# ── Section 4: Dataset & Training ─────────────────────────────────────

cells.append(md("## 4. Fine-tune ByT5 on Training Data"))

cells.append(
    code(
        """class AkkadianDataset(Dataset):
    def __init__(self, df, tokenizer, max_src=512, max_tgt=512,
                 src_col='transliteration', tgt_col='translation'):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.src_col = src_col
        self.tgt_col = tgt_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source = f"translate Akkadian to English: {row[self.src_col]}"
        enc = self.tokenizer(source, max_length=self.max_src,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        result = {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
        }
        if self.tgt_col in row.index:
            tgt_enc = self.tokenizer(str(row[self.tgt_col]),
                                     max_length=self.max_tgt,
                                     padding='max_length', truncation=True,
                                     return_tensors='pt')
            labels = tgt_enc['input_ids'].squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result['labels'] = labels
        return result


def train_model(model, tokenizer, train_df, val_df,
                epochs=10, batch_size=4, grad_accum=4, lr=3e-4):
    model = model.to(DEVICE)
    use_amp = (DEVICE == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    train_ds = AkkadianDataset(train_df, tokenizer)
    val_ds = AkkadianDataset(val_df, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2,
                            num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs // grad_accum
    warmup_steps = total_steps // 10

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_state = None
    step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels).loss
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum
            n_batches += 1

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

        avg_train = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss = model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels).loss
                val_loss += loss.item()
                val_n += 1
        avg_val = val_loss / max(val_n, 1)

        improved = avg_val < best_val_loss
        if improved:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f'Epoch {epoch+1}/{epochs} -- '
              f'train_loss: {avg_train:.4f}  val_loss: {avg_val:.4f}'
              f'{" *" if improved else ""}')

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
        print(f'\\nRestored best checkpoint (val_loss: {best_val_loss:.4f})')

    return model


print('Training helpers defined.')"""
    )
)

cells.append(
    code(
        """if model is not None:
    # Split: 90% train, 10% validation
    val_size = max(int(len(train) * 0.1), 1)
    shuffled = train.sample(frac=1, random_state=42).reset_index(drop=True)
    train_split = shuffled.iloc[:-val_size]
    val_split = shuffled.iloc[-val_size:]
    print(f'Training: {len(train_split)} | Validation: {len(val_split)}')

    model = train_model(model, tokenizer, train_split, val_split,
                        epochs=10, batch_size=4, grad_accum=4, lr=3e-4)
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
else:
    print('Skipping training (no model loaded).')"""
    )
)

# ── Section 5: ByT5 Inference ─────────────────────────────────────────

cells.append(md("## 5. ByT5 Beam Search Inference"))

cells.append(
    code(
        """def translate_batch(model, tokenizer, texts, batch_size=8,
                       max_src=512, max_tgt=512, num_beams=5):
    model.eval()
    all_translations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        prefixed = [f'translate Akkadian to English: {t}' for t in batch_texts]
        inputs = tokenizer(prefixed, max_length=max_src, padding=True,
                           truncation=True, return_tensors='pt').to(DEVICE)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
            outputs = model.generate(
                **inputs,
                max_length=max_tgt,
                num_beams=num_beams,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_translations.extend(translations)

        if (i // batch_size) % 5 == 0:
            print(f'  Translated {min(i + batch_size, len(texts))}/{len(texts)}')

    return all_translations


byt5_predictions = None

if model is not None:
    test_texts = test['transliteration'].astype(str).tolist()
    print(f'Translating {len(test_texts)} test rows with ByT5...')
    byt5_predictions = translate_batch(model, tokenizer, test_texts)

    print(f'\\nSample ByT5 translations:')
    for i in range(min(3, len(byt5_predictions))):
        print(f'  [{i+1}] {test_texts[i][:60]}...')
        print(f'      -> {byt5_predictions[i][:80]}')
else:
    print('No ByT5 model -- skipping neural inference.')"""
    )
)

# ── Section 6: TF-IDF Retrieval Fallback ──────────────────────────────

cells.append(md("## 6. TF-IDF Retrieval Fallback"))

cells.append(
    code(
        r"""def normalize_transliteration(text):
    normalized = str(text or '').lower()
    for old, new in {
        '\u2026': ' ', '...': ' ', '\u201e': ' ', '\u201c': ' ',
        '\u201d': ' ', '\u201f': ' ',
        "'": ' ', '`': ' ', '\u00b4': ' ', '{': ' ', '}': ' ', '(': ' ',
        ')': ' ', '[': ' ', ']': ' ', '/': ' ', '\\': ' ', ',': ' ',
        '.': ' ', ';': ' ', ':': ' ', '!': ' ', '?': ' ',
    }.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r'<[^>]+>', ' ', normalized)
    return ' '.join(normalized.split())


def best_match(corpus, query):
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


def split_translation_by_rows(text, test_rows):
    ordered = test_rows.sort_values(['line_start', 'line_end']).reset_index(drop=True)
    weights = (
        ordered['line_end'].fillna(ordered['line_start']).astype(int)
        - ordered['line_start'].astype(int) + 1
    ).clip(lower=1).tolist()
    words = str(text or '').split()
    if not words:
        return ['' for _ in weights]
    chunks = []
    pos = 0
    for idx, w in enumerate(weights):
        remaining = len(words) - pos
        groups_left = len(weights) - idx
        if idx == len(weights) - 1:
            take = remaining
        else:
            take = max(1, round(len(words) * w / sum(weights)))
            take = min(take, remaining - (groups_left - 1))
        chunks.append(' '.join(words[pos:pos+take]).strip())
        pos += take
    return chunks


# TF-IDF retrieval from training set
query = ' '.join(test['transliteration'].astype(str))
train_idx, train_score = best_match(train['transliteration'], query)
train_translation = str(train.iloc[train_idx]['translation'])
tfidf_predictions = split_translation_by_rows(train_translation, test)

# Fill empty predictions with sample submission
fallback = sample['translation'].astype(str).tolist()
tfidf_predictions = [p.strip() or fallback[i] for i, p in enumerate(tfidf_predictions)]

print(f'TF-IDF best match score: {train_score:.4f}')
print(f'TF-IDF predictions: {len(tfidf_predictions)} rows')"""
    )
)

# ── Section 7: Build Submission ───────────────────────────────────────

cells.append(md("## 7. Build Hybrid Submission"))

cells.append(
    code(
        """if byt5_predictions is not None:
    # Use ByT5 as primary, fall back to TF-IDF for empty predictions
    final_predictions = []
    byt5_used = 0
    tfidf_used = 0

    for i, pred in enumerate(byt5_predictions):
        if pred and pred.strip() and len(pred.strip()) > 2:
            final_predictions.append(pred.strip())
            byt5_used += 1
        else:
            final_predictions.append(tfidf_predictions[i])
            tfidf_used += 1

    chosen_model = f'ByT5 hybrid (ByT5: {byt5_used}, TF-IDF fallback: {tfidf_used})'
else:
    final_predictions = tfidf_predictions
    chosen_model = 'TF-IDF retrieval only (no ByT5 model available)'

print(f'Model: {chosen_model}')
print(f'Predictions: {len(final_predictions)} rows')

submission = pd.DataFrame({'id': test['id'], 'translation': final_predictions})
submission.to_csv('submission.csv', index=False)

print(f'\\nsubmission.csv written ({len(submission)} rows)')
print(submission.head(10).to_string(index=False))
if len(submission) > 10:
    print(f'... ({len(submission) - 10} additional rows omitted)')"""
    )
)

# ── Notes ─────────────────────────────────────────────────────────────

cells.append(
    md(
        """## Notes
- **Model**: ByT5-base (580M params) fine-tuned on competition training data
- **Inference**: beam search (num_beams=5, length_penalty=1.0, no_repeat_ngram_size=3)
- **Training**: 10 epochs, AdamW (lr=3e-4), linear warmup + cosine decay, FP16
- **Fallback**: TF-IDF char+word retrieval from training set (same as original baseline)
- **Metric**: competition uses sqrt(BLEU x chrF++) -- ByT5's byte-level architecture
  naturally optimizes both word-level (BLEU) and character-level (chrF++) overlap"""
    )
)

write_notebook(cells, __file__, "akkadian_submission_baseline.ipynb")
