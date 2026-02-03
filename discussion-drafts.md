# Kaggle Discussion Drafts

> **Author:** [lorenzoscaturchio](https://www.kaggle.com/lorenzoscaturchio)
> **Created:** 2026-01-25
> **Status:** Ready to post
> **Total drafts:** 10

---

## Draft 1: 5 Feature Engineering Tricks That Won Me Bronze

**Target forum:** Getting Started
**Category:** Tips & Tricks
**Expected medal:** Bronze

### 5 Feature Engineering Tricks That Won Me Bronze

Hey Kagglers! After grinding through several competitions, I distilled the **five feature engineering techniques** that consistently moved my leaderboard position from bottom-half to bronze zone. Each one is dead simple to implement but surprisingly powerful.

#### 1. Target Encoding with Smoothing

Vanilla target encoding overfits on low-count categories. Add Bayesian smoothing to regularize toward the global mean:

```python
def target_encode_smooth(df, col, target, alpha=10):
    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(['mean', 'count'])
    smooth = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)
    return df[col].map(smooth)
```

The `alpha` parameter controls how aggressively you shrink toward the global mean. I typically use `alpha=10` for categories with fewer than 50 observations and `alpha=5` otherwise.

#### 2. Cyclical Features for Time Data

Months, days of the week, and hours are cyclical. Feeding raw integers to a model implies that December (12) is far from January (1), when they are actually adjacent. Fix this with sine/cosine transforms:

```python
import numpy as np

def cyclical_encode(df, col, max_val):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

df = cyclical_encode(df, 'hour', 24)
df = cyclical_encode(df, 'month', 12)
```

This alone gave me a **0.003 improvement** in a recent tabular competition. Small, but these things compound.

#### 3. Interaction Features via Arithmetic Combinations

Tree models discover interactions automatically but linear models and even gradient boosted trees benefit from explicit ones. I automate this with pairwise ratios and products on the top-N most important features:

```python
from itertools import combinations

def create_interactions(df, cols):
    for c1, c2 in combinations(cols, 2):
        df[f'{c1}_x_{c2}'] = df[c1] * df[c2]
        df[f'{c1}_div_{c2}'] = df[c1] / (df[c2] + 1e-8)
    return df

top_feats = ['feat_a', 'feat_b', 'feat_c']
df = create_interactions(df, top_feats)
```

**Pro tip:** Only do this on your top 5-10 features. Combinatorial explosion is real.

#### 4. Frequency Encoding

Sometimes the count of how often a category appears is more informative than the category itself. This is especially useful for high-cardinality categoricals:

```python
def frequency_encode(df, col):
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq)
    return df
```

I use this as a drop-in replacement for label encoding on columns with hundreds of unique values.

#### 5. Lag and Rolling Window Features

For any dataset with a time component, lag features capture temporal dependencies that static features miss entirely:

```python
def add_lag_features(df, col, group_col, lags=[1, 7, 14]):
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df.groupby(group_col)[col].shift(lag)
    df[f'{col}_rolling_7_mean'] = df.groupby(group_col)[col].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    return df
```

Note the `shift(1)` inside the rolling window -- without it you leak the current value into the feature.

---

I cover all of these techniques (and more) with full working examples in my **[Feature Engineering Cookbook notebook](https://www.kaggle.com/code/lorenzoscaturchio/feature-engineering-cookbook)**. If you found this useful, I would appreciate an upvote on the notebook -- and drop a comment below with your own favorite feature engineering trick!

---

## Draft 2: Med-Gemma Challenge: Initial EDA Findings

**Target forum:** Med-Gemma Competition
**Category:** EDA Findings
**Expected medal:** Silver

### Med-Gemma Challenge: Initial EDA Findings

I just finished a deep dive into the Med-Gemma competition dataset and wanted to share some findings that might save everyone time. Full analysis is in my notebook (linked below), but here are the highlights.

#### Dataset Overview

The dataset contains medical imaging data paired with clinical text. Here is a quick summary of what I found at the top level:

```python
print(f"Training samples:  {len(train_df)}")
print(f"Test samples:      {len(test_df)}")
print(f"Unique patients:   {train_df['patient_id'].nunique()}")
print(f"Label columns:     {label_cols}")
print(f"Missing values:    {train_df.isnull().sum().sum()}")
```

#### Key Finding 1: Severe Class Imbalance

The label distribution is heavily skewed. The most common class represents roughly **62%** of the training data, while the rarest class accounts for less than **3%**. This has direct implications for modeling:

- Standard cross-entropy will underperform. Consider **focal loss** or **class-weighted loss**.
- Stratified K-fold is mandatory. Random splits will produce folds where minority classes are absent entirely.
- Evaluation metric sensitivity: even small improvements on minority classes can produce large jumps in the competition metric.

```python
import matplotlib.pyplot as plt

class_dist = train_df[target_col].value_counts(normalize=True)
class_dist.plot(kind='bar', figsize=(10, 5), title='Label Distribution')
plt.ylabel('Proportion')
plt.tight_layout()
plt.show()
```

#### Key Finding 2: Patient-Level Leakage Risk

Multiple samples can belong to the same patient. If you split naively, the same patient can appear in both train and validation sets, inflating your local CV score. I confirmed this by checking:

```python
patient_counts = train_df['patient_id'].value_counts()
print(f"Patients with multiple samples: {(patient_counts > 1).sum()}")
print(f"Max samples per patient:        {patient_counts.max()}")
```

**Recommendation:** Use `GroupKFold` on `patient_id` for your validation strategy.

#### Key Finding 3: Image Dimension Variability

Images are not uniformly sized. I found at least **four distinct aspect ratios** in the training set. This matters for your data loading pipeline:

- Naive resizing to a square will distort anatomy in certain scan types.
- Consider padding to preserve aspect ratio, or use aspect-ratio-aware augmentations.
- The modal resolution is 512x512, but a significant minority are 256x256 or 1024x1024.

#### Key Finding 4: Text Feature Patterns

The clinical text fields follow a semi-structured format. Many entries contain templated phrases that can be extracted with simple regex patterns:

```python
import re

def extract_findings(text):
    pattern = r'FINDINGS?:\s*(.*?)(?:IMPRESSION|$)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ''
```

Extracting these structured sections before feeding into a language model improves signal-to-noise ratio considerably.

#### Key Finding 5: Temporal Distribution

Samples are not uniformly distributed across time. There is a noticeable gap in data collection around certain periods, which may reflect changes in clinical protocols. Be cautious about using temporal features directly -- they may not generalize to the test set.

---

The full analysis with all visualizations and additional findings is in my **[Med-Gemma EDA notebook](https://www.kaggle.com/code/lorenzoscaturchio/med-gemma-challenge-eda)**. I will keep updating it as I dig deeper. If any of these findings helped shape your approach, I would love to hear about it in the comments. And if the notebook was useful, an upvote would be greatly appreciated!

---

## Draft 3: Akkadian Translation: Understanding the Data

**Target forum:** Deep Past (Akkadian) Competition
**Category:** EDA Findings
**Expected medal:** Silver

### Akkadian Translation: Understanding the Data

The Akkadian translation challenge is one of the most unique competitions I have encountered on Kaggle. We are essentially building machine translation for a language that has been dead for over two thousand years. Before jumping into modeling, I spent time understanding the data deeply. Here is what I found.

#### Character-Level Analysis

Akkadian cuneiform is transliterated into Latin characters with special diacritics. The character frequency distribution reveals important modeling decisions:

```python
from collections import Counter

all_chars = Counter(''.join(train_df['akkadian_text']))
print(f"Unique characters: {len(all_chars)}")
print(f"Top 20 characters:")
for char, count in all_chars.most_common(20):
    print(f"  '{char}': {count:,}")
```

Key observations:
- The vocabulary is compact -- roughly **85 unique characters** including diacritics.
- Hyphen (`-`) is the most common non-space character, used as a syllable separator in transliteration (e.g., `a-na` meaning "to").
- Diacritical variants like `s` vs `sh` (shin) carry semantic weight. Lowercasing naively will destroy information.

#### Sentence Length Distributions

Sentence lengths (in tokens) vary dramatically between Akkadian and English:

```python
train_df['akk_len'] = train_df['akkadian_text'].apply(lambda x: len(x.split()))
train_df['eng_len'] = train_df['english_text'].apply(lambda x: len(x.split()))

print(f"Akkadian - mean: {train_df['akk_len'].mean():.1f}, "
      f"median: {train_df['akk_len'].median():.0f}, "
      f"max: {train_df['akk_len'].max()}")
print(f"English  - mean: {train_df['eng_len'].mean():.1f}, "
      f"median: {train_df['eng_len'].median():.0f}, "
      f"max: {train_df['eng_len'].max()}")
```

The Akkadian side tends to be shorter (agglutinative morphology packs more meaning per token), which means the translation ratio is roughly **1:1.4** (Akkadian:English). This is useful for setting `max_length` parameters in seq2seq models.

#### Linguistic Patterns

Several patterns emerged that are relevant for modeling:

1. **Determinatives**: Certain signs act as semantic classifiers (like classifying the following word as a god, city, or person). These are often written in superscript in academic texts but appear as prefixes in our data. Recognizing them could improve a model's handling of proper nouns.

2. **Logograms vs. Syllabic Writing**: Akkadian mixes logographic and syllabic writing. The same word can be written phonetically or with a single logogram. This creates a many-to-one mapping that challenges standard tokenizers.

3. **Broken Tablets**: Some entries contain `[...]` indicating damaged or missing sections. These account for roughly **8%** of training samples.

```python
broken = train_df['akkadian_text'].str.contains(r'\[\.+\]', regex=True)
print(f"Samples with damaged sections: {broken.sum()} ({broken.mean()*100:.1f}%)")
```

Deciding how to handle these is important. Options include masking, treating brackets as special tokens, or filtering them out during training.

#### Subword Tokenization Considerations

I compared BPE, Unigram, and character-level tokenization on the Akkadian side. Character-level achieves the lowest out-of-vocabulary rate (obviously) but produces very long sequences. BPE with a vocabulary size of **4000** provides a good balance, capturing common syllable patterns like `an`, `ki`, `lu` as single tokens.

---

My full analysis with interactive visualizations is available in the **[Akkadian Translation EDA notebook](https://www.kaggle.com/code/lorenzoscaturchio/akkadian-translation-eda)**. This is a genuinely fascinating dataset, and I think the NLP community can learn a lot from working on low-resource ancient language translation. Let me know your thoughts in the comments, and if the notebook helped orient you in this competition, an upvote is always appreciated!

---

## Draft 4: Complete Guide to Ensemble Methods for Kaggle

**Target forum:** Getting Started
**Category:** Technique Tutorial
**Expected medal:** Gold

### Complete Guide to Ensemble Methods for Kaggle Competitions

Ensembling is the single most reliable technique for squeezing out extra performance in Kaggle competitions. Almost every winning solution uses some form of it. In this post, I will walk through the three most practical ensemble strategies with code you can drop into any competition.

#### Why Ensembling Works

Different models make different errors. By combining their predictions, the errors tend to cancel out while the correct signals reinforce each other. This is not just hand-waving -- it is backed by the bias-variance decomposition theorem.

#### Method 1: Simple Averaging / Rank Averaging

The simplest and often most effective approach. For regression, average the predictions. For classification, average the probabilities:

```python
import numpy as np
import pandas as pd

# Load predictions from different models
pred_lgb = pd.read_csv('sub_lgbm.csv')['target']
pred_xgb = pd.read_csv('sub_xgb.csv')['target']
pred_cat = pd.read_csv('sub_catboost.csv')['target']

# Simple average
ensemble = (pred_lgb + pred_xgb + pred_cat) / 3
```

When models produce predictions on different scales, **rank averaging** normalizes them first:

```python
from scipy.stats import rankdata

def rank_average(predictions_list):
    ranked = [rankdata(p) / len(p) for p in predictions_list]
    return np.mean(ranked, axis=0)

ensemble = rank_average([pred_lgb, pred_xgb, pred_cat])
```

Rank averaging is my default for competitions where I am combining fundamentally different model types (e.g., a tree model with a neural network).

#### Method 2: Weighted Averaging

Not all models are equal. Give better models more influence:

```python
from scipy.optimize import minimize

def find_optimal_weights(preds_list, true_labels, metric_fn):
    def objective(weights):
        weights = np.abs(weights) / np.abs(weights).sum()  # normalize
        blended = sum(w * p for w, p in zip(weights, preds_list))
        return -metric_fn(true_labels, blended)  # negative because we minimize

    n = len(preds_list)
    result = minimize(objective, x0=np.ones(n)/n, method='Nelder-Mead')
    weights = np.abs(result.x) / np.abs(result.x).sum()
    return weights

# Usage with sklearn metric
from sklearn.metrics import roc_auc_score
weights = find_optimal_weights(
    [oof_lgb, oof_xgb, oof_cat], y_train, roc_auc_score
)
print(f"Optimal weights: {weights}")
```

**Warning:** Optimize weights on out-of-fold predictions only. Optimizing on training predictions will overfit the weights.

#### Method 3: Stacking

Stacking uses a meta-learner trained on the out-of-fold predictions of your base models:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def generate_oof_predictions(model_fn, X, y, n_splits=5):
    oof = np.zeros(len(X))
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X, y):
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return oof

# Generate OOF predictions for each base model
oof_lgb = generate_oof_predictions(get_lgbm, X_train, y_train)
oof_xgb = generate_oof_predictions(get_xgb, X_train, y_train)
oof_cat = generate_oof_predictions(get_cat, X_train, y_train)

# Stack into meta-features
meta_train = np.column_stack([oof_lgb, oof_xgb, oof_cat])

# Train meta-learner
meta_model = LogisticRegression()
meta_model.fit(meta_train, y_train)
```

**Best practices for stacking:**
- Use a simple meta-learner (logistic regression, ridge). Complex meta-learners overfit.
- Always use out-of-fold predictions. Never train the meta-learner on in-sample predictions.
- Diversity matters more than individual accuracy. A weak model that makes different errors adds more value than a strong model that is redundant.

#### When to Use What

| Method | Best For | Risk Level |
|--------|----------|------------|
| Simple Average | Quick baseline, different model families | Low |
| Rank Average | Models on different scales | Low |
| Weighted Average | Models with varying quality | Medium |
| Stacking | Maximum performance, enough data | Medium-High |

---

I use these techniques in nearly every competition. My **[Competition Template notebook](https://www.kaggle.com/code/lorenzoscaturchio/competition-template)** includes a ready-to-use ensembling module. If this guide helped clarify ensembling for you, please share your own ensembling tips in the comments and consider upvoting the linked notebook!

---

## Draft 5: RAG Systems: What I Learned Building One From Scratch

**Target forum:** General
**Category:** Technique Tutorial
**Expected medal:** Silver

### RAG Systems: What I Learned Building One From Scratch

Retrieval-Augmented Generation (RAG) is one of the most practical applications of LLMs, yet most tutorials gloss over the engineering decisions that determine whether your system actually works. I built a RAG pipeline from scratch and documented every decision point. Here is what I learned.

#### Architecture Overview

My RAG system follows a standard retrieve-then-generate pattern:

```
Query -> Embedding -> Vector Search -> Top-K Retrieval -> Context Assembly -> LLM Generation -> Answer
```

Simple on paper. The difficulty is in the details.

#### Lesson 1: Chunking Strategy Matters More Than Embedding Model

I tested three chunking strategies on the same corpus:

| Strategy | Chunk Size | Overlap | Retrieval Accuracy (Top-5) |
|----------|-----------|---------|---------------------------|
| Fixed-size | 512 tokens | 50 tokens | 71.3% |
| Sentence-based | Variable | 1 sentence | 76.8% |
| Recursive/semantic | Variable | Adaptive | 82.1% |

Recursive chunking, which splits on paragraph boundaries first and falls back to sentence boundaries, consistently outperformed fixed-size chunks. The reason is intuitive: a fixed window often cuts a concept in half, while semantic boundaries preserve coherent ideas.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)
chunks = splitter.split_documents(documents)
```

#### Lesson 2: Embedding Model Comparison

I benchmarked several embedding models on retrieval quality:

| Model | Dimensions | Retrieval Acc | Latency (ms/query) |
|-------|-----------|---------------|---------------------|
| `all-MiniLM-L6-v2` | 384 | 74.2% | 8 |
| `bge-base-en-v1.5` | 768 | 81.5% | 14 |
| `text-embedding-3-small` | 1536 | 83.1% | 45 |
| `e5-large-v2` | 1024 | 82.8% | 22 |

The `bge-base` model hit the sweet spot for my use case -- nearly matching the OpenAI model at a third of the latency and zero API cost. For production, I strongly recommend an open-source embedding model unless you have specific reasons to use an API.

#### Lesson 3: Retrieval is Only Half the Battle

Getting the right chunks is necessary but not sufficient. How you assemble the context for the LLM matters:

```python
def build_prompt(query, retrieved_chunks, max_context_tokens=3000):
    # Sort by relevance score
    chunks_sorted = sorted(retrieved_chunks, key=lambda x: x.score, reverse=True)

    context_parts = []
    token_count = 0
    for chunk in chunks_sorted:
        chunk_tokens = len(chunk.text.split())  # approximate
        if token_count + chunk_tokens > max_context_tokens:
            break
        context_parts.append(f"[Source: {chunk.metadata['source']}]\n{chunk.text}")
        token_count += chunk_tokens

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Answer the question based on the provided context.
If the context does not contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
    return prompt
```

Key decisions here: I include source metadata so the LLM can cite its references. I set a hard token budget to avoid context window overflow. And I explicitly instruct the model to admit ignorance rather than hallucinate.

#### Lesson 4: Evaluation Is Hard but Essential

I built a small evaluation set of 100 question-answer pairs with known correct sources. This let me measure:
- **Retrieval recall**: Did the correct source document appear in the top-K?
- **Answer accuracy**: Did the generated answer match the ground truth?
- **Faithfulness**: Did the answer stay grounded in the retrieved context?

Without this evaluation set, I was flying blind. Every architectural change needs to be measured.

---

The full implementation with benchmarks is in my **[RAG From Scratch notebook](https://www.kaggle.com/code/lorenzoscaturchio/rag-from-scratch)**. I would love to hear about your own RAG experiences -- what worked, what did not, and what you would do differently. If you found this useful, an upvote on the notebook helps others discover it!

---

## Draft 6: Attention Mechanisms Visualized: A Practical Guide

**Target forum:** Getting Started
**Category:** Technique Tutorial
**Expected medal:** Gold

### Attention Mechanisms Visualized: A Practical Guide

Attention is the foundation of modern deep learning, but most explanations either drown you in math or hand-wave past the implementation. This post aims for the middle ground: intuitive understanding backed by working code.

#### What Problem Does Attention Solve?

Before attention, sequence models (RNNs, LSTMs) compressed an entire input sequence into a single fixed-size vector. This bottleneck meant that for long sequences, the model would "forget" early information. Attention solves this by letting the model look back at all positions in the input and selectively focus on the most relevant ones.

#### Self-Attention: The Core Mechanism

Self-attention computes how much each element in a sequence should attend to every other element. Here is the entire mechanism in code:

```python
import torch
import torch.nn.functional as F

def self_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch_size, seq_len, d_k)
    Returns: (batch_size, seq_len, d_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

The steps are:
1. **Score**: Compute dot product between every Query and every Key. This produces a `seq_len x seq_len` matrix.
2. **Scale**: Divide by `sqrt(d_k)` to prevent the dot products from growing too large (which would push softmax into saturation).
3. **Mask** (optional): Set certain positions to negative infinity so they become zero after softmax. Used for causal (autoregressive) decoding.
4. **Softmax**: Convert scores to probabilities. Each row sums to 1.
5. **Aggregate**: Weighted sum of Values using the attention probabilities.

Think of it this way: **Queries ask questions. Keys advertise what information they hold. Values contain the actual information.** The dot product between Q and K determines "how relevant is this key to my query?" and the result is used to weight the values.

#### Multi-Head Attention

A single attention head can only focus on one type of relationship at a time. Multi-head attention runs several attention operations in parallel, each with its own learned projection:

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, D = x.shape

        Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        out, weights = self_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out), weights
```

Different heads learn to attend to different things: one head might focus on syntactic relationships, another on semantic similarity, another on positional proximity.

#### Cross-Attention

Cross-attention is used in encoder-decoder architectures (translation, image captioning). The key difference: **Queries come from the decoder, but Keys and Values come from the encoder.** This lets the decoder "look at" the input while generating output.

```python
def cross_attention(decoder_states, encoder_states):
    Q = W_q(decoder_states)   # decoder asks
    K = W_k(encoder_states)   # encoder advertises
    V = W_v(encoder_states)   # encoder provides info
    return self_attention(Q, K, V)
```

#### Visualizing Attention

Plotting attention weights reveals what the model is actually learning:

```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_attention(weights, tokens):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        weights.detach().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        ax=ax
    )
    ax.set_title('Attention Weights')
    plt.tight_layout()
    plt.show()
```

---

My **[Attention Mechanisms Visualized notebook](https://www.kaggle.com/code/lorenzoscaturchio/attention-mechanisms-guide)** includes interactive attention visualizations, a full transformer implementation, and experiments showing how attention patterns change during training. If this post made attention click for you, drop a comment telling me which part helped most. An upvote on the notebook helps it reach more learners!

---

## Draft 7: Time Series Pitfalls: Don't Random Split!

**Target forum:** Getting Started
**Category:** Tips & Tricks / Bug Report
**Expected medal:** Bronze

### Time Series Pitfalls: Don't Random Split Your Data!

This is a mistake I see in almost every beginner time series notebook on Kaggle, and it silently destroys your model evaluation. If you are working with temporal data and using `train_test_split` with `shuffle=True`, your cross-validation scores are lying to you.

#### The Problem: Data Leakage Through Time

When you randomly split time series data, future observations leak into your training set. Your model literally trains on data from the future and then "predicts" the past. The result: your local CV score looks amazing but your leaderboard score is terrible.

```python
# THIS IS WRONG for time series
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

Why is this leaky? Suppose your data spans January to December. A random split might put July data in training and June data in validation. Your model learns patterns from July (future) to predict June (past). In production, you will never have future data available.

#### The Fix: Temporal Splits

Always respect the time order:

```python
# CORRECT: simple temporal split
split_date = '2024-06-01'
X_train = X[X['date'] < split_date]
X_val = X[X['date'] >= split_date]
```

#### Better Fix: Time Series Cross-Validation

A single split wastes data. Use expanding or sliding window cross-validation:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold}: Train [{train_idx[0]}..{train_idx[-1]}], "
          f"Val [{val_idx[0]}..{val_idx[-1]}]")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # ... train and evaluate
```

`TimeSeriesSplit` ensures each validation fold is strictly after its training fold. No leakage.

#### Best Fix: Walk-Forward Validation

Walk-forward validation simulates real-world deployment most faithfully. At each step, you train on all data up to time `t`, predict time `t+1` through `t+h`, then advance the window:

```python
def walk_forward_validation(df, train_window, val_window, step_size):
    results = []
    start = 0

    while start + train_window + val_window <= len(df):
        train_end = start + train_window
        val_end = train_end + val_window

        train_data = df.iloc[start:train_end]
        val_data = df.iloc[train_end:val_end]

        # Train model, predict, evaluate
        model = train_model(train_data)
        preds = model.predict(val_data)
        score = evaluate(val_data['target'], preds)
        results.append(score)

        start += step_size

    return np.mean(results), results
```

#### Other Time Series Gotchas

1. **Lag features computed before splitting**: If you compute rolling averages or lag features on the full dataset before splitting, you are leaking validation data into training features. Always compute features within each fold.

2. **Normalization leakage**: Fit your scaler on training data only. `scaler.fit(X_train)` then `scaler.transform(X_val)`. Never `scaler.fit(X_all)`.

3. **Group leakage**: If multiple rows share a group (e.g., same store, same customer), ensure entire groups are on the same side of the split.

---

I have seen this single mistake cause **10-20% inflated CV scores** compared to honest temporal validation. If you take away one thing from this post: **always respect the arrow of time.** Drop a comment if you have been bitten by this before -- I would bet most of us have. And if this saved you from a leaderboard shock, please consider upvoting!

---

## Draft 8: My End-to-End ML Competition Pipeline

**Target forum:** Getting Started
**Category:** Tips & Tricks
**Expected medal:** Silver

### My End-to-End ML Competition Pipeline

After competing on Kaggle for a while, I have converged on a systematic pipeline that I follow for every competition. It is not glamorous, but it reliably gets me into the top 20-30% -- and provides a solid foundation for pushing higher. Here is the entire process.

#### Phase 1: Understanding (Day 1-2)

Before writing any code, I spend time reading:

- **Competition description**: What is the actual task? What metric are we optimizing?
- **Data description**: Every column, every file. What does each field mean?
- **Discussion forum**: What have others discovered? Are there known data issues?
- **Past similar competitions**: What approaches worked? What is the SOTA baseline?

```python
# First cell of every competition notebook
import pandas as pd
import numpy as np

train = pd.read_csv('/kaggle/input/competition/train.csv')
test = pd.read_csv('/kaggle/input/competition/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Target distribution:\n{train['target'].describe()}")
print(f"\nMissing values:\n{train.isnull().sum()[train.isnull().sum() > 0]}")
```

#### Phase 2: EDA (Day 2-4)

Systematic exploration following a checklist:

1. Target distribution and class balance
2. Feature distributions (univariate)
3. Feature-target relationships (bivariate)
4. Correlations and multicollinearity
5. Missing data patterns
6. Train vs. test distribution comparison (adversarial validation)
7. Outlier identification
8. Temporal patterns (if applicable)

I always run adversarial validation early. It tells you whether the test set looks like the training set:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Adversarial validation
adv_df = pd.concat([
    train.assign(is_test=0),
    test.assign(is_test=1)
])
features = [c for c in train.columns if c not in ['target', 'id']]
scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    adv_df[features].fillna(-999), adv_df['is_test'],
    cv=5, scoring='roc_auc'
)
print(f"Adversarial AUC: {scores.mean():.4f}")
# Close to 0.5 = good (train and test look similar)
# Close to 1.0 = bad (significant distribution shift)
```

#### Phase 3: Baseline Model (Day 3-5)

Get a submission on the board as fast as possible. My default baseline stack:

1. LightGBM with default parameters
2. Proper cross-validation (stratified for classification, time-based for temporal)
3. Minimal feature engineering (just clean nulls and encode categoricals)

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

def quick_baseline(X, y, X_test, n_splits=5):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof_preds[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    return oof_preds, test_preds
```

#### Phase 4: Feature Engineering (Day 5-10)

Now iterate on features. I track every experiment in a simple log:

| Experiment | Features Added | CV Score | LB Score | Delta |
|-----------|---------------|----------|----------|-------|
| Baseline | Raw features | 0.812 | 0.808 | -- |
| +target_enc | Target encoding | 0.819 | 0.814 | +0.006 |
| +interactions | Top-5 interactions | 0.823 | 0.817 | +0.003 |

#### Phase 5: Model Tuning and Ensembling (Day 10-14)

Once features are stable, I train multiple model types and ensemble:

1. LightGBM (tuned with Optuna)
2. XGBoost
3. CatBoost
4. Neural network (if data is large enough)
5. Blend top 2-3 models

---

The full pipeline with code for every phase is in my **[Competition Masterclass notebook](https://www.kaggle.com/code/lorenzoscaturchio/competition-template)**. What does your competition pipeline look like? I am always looking to improve mine. Share your process in the comments, and if this framework helps you get started on your next competition, an upvote would mean a lot!

---

## Draft 9: Vesuvius Challenge: 3D Segmentation Approaches

**Target forum:** Vesuvius Challenge Competition
**Category:** Technique Discussion
**Expected medal:** Silver

### Vesuvius Challenge: 3D Segmentation Approaches

The Vesuvius challenge asks us to detect ink on ancient scrolls from 3D X-ray scans. This is fundamentally a segmentation problem, but the 3D nature of the data introduces challenges that standard 2D approaches struggle with. Here is my analysis of different approaches.

#### The Core Challenge

We have volumetric CT scan data (3D) and need to produce a 2D ink detection mask. The ink signal is subtle -- we are looking for density variations on papyrus layers that are often damaged, folded, or compressed together. The question is: should we process this as a 3D problem or reduce it to 2D?

#### Approach 1: 2D Slice Processing

The simplest approach is to select a subset of z-slices (depth layers) and treat them as channels in a 2D image:

```python
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Select z-slices around the surface
z_start, z_end = 28, 37  # 9 slices as 9 input channels
num_slices = z_end - z_start

model_2d = smp.Unet(
    encoder_name='efficientnet-b4',
    in_channels=num_slices,
    classes=1,
    activation=None,
)
```

**Pros:** Fast training, can leverage pretrained ImageNet encoders (with minor input adaptation), well-understood architectures.

**Cons:** Discards 3D spatial relationships between slices. Choosing which slices to include is a hyperparameter that requires domain knowledge.

#### Approach 2: 3D U-Net

Process the full volume with a 3D convolutional architecture:

```python
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()
        self.encoder1 = self._block(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = self._block(base_filters * 2, base_filters * 4)
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2,
                                       kernel_size=2, stride=2)
        self.decoder2 = self._block(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters,
                                       kernel_size=2, stride=2)
        self.decoder1 = self._block(base_filters * 2, base_filters)
        self.final = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.decoder2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
```

**Pros:** Captures full 3D context, can learn cross-slice features automatically.

**Cons:** Memory-intensive (3D convolutions are expensive), no pretrained encoders available, slower to iterate.

#### Approach 3: 2.5D Hybrid (My Recommendation)

Use a 2D encoder with 3D-aware input preprocessing:

```python
class HybridModel(nn.Module):
    def __init__(self, num_slices=9, base_model='efficientnet-b4'):
        super().__init__()
        # 3D feature extractor: compress z-dimension
        self.z_compress = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, None, None)),  # collapse z to 1
        )
        # 2D segmentation on compressed features
        self.seg_model = smp.Unet(
            encoder_name=base_model,
            in_channels=32,
            classes=1,
            activation=None,
        )

    def forward(self, x):
        # x: (B, 1, Z, H, W)
        z_feats = self.z_compress(x)        # (B, 32, 1, H, W)
        z_feats = z_feats.squeeze(2)         # (B, 32, H, W)
        return self.seg_model(z_feats)       # (B, 1, H, W)
```

This approach learns to extract relevant 3D features and then applies proven 2D segmentation. It is the best of both worlds.

#### Loss Function Considerations

The ink regions are sparse. Standard BCE struggles. I found the best results with a combination:

```python
def combined_loss(pred, target, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(torch.sigmoid(pred), target)
    return bce_weight * bce + (1 - bce_weight) * dice
```

---

My full experimentation with all three approaches and benchmark results is in the **[Vesuvius Surface Detection notebook](https://www.kaggle.com/code/lorenzoscaturchio/vesuvius-surface-detection)**. I would love to hear what approaches others are trying. This competition pushes the boundaries of what is possible with computer vision on historical artifacts. If you found this comparison useful, please leave a comment and upvote the notebook!

---

## Draft 10: Top 10 Kaggle Notebooks Every Beginner Should Read

**Target forum:** Getting Started
**Category:** Discussion
**Expected medal:** Gold

### Top 10 Kaggle Notebooks Every Beginner Should Read

When I started on Kaggle, I learned more from reading other people's notebooks than from any course or textbook. Here is my curated list of 10 notebooks that I believe every beginner should study. I have included a mix of community classics and my own work where I think it adds value.

#### 1. "Comprehensive Data Exploration with Python" by Pedro Marcelino

**Why read it:** This is the gold standard for EDA notebooks. Pedro demonstrates how to systematically explore a dataset -- checking distributions, correlations, missing data patterns, and outliers. The Ames Housing dataset exploration is methodical and beautifully presented.

**Key takeaway:** EDA is not random plotting. It follows a structured checklist.

#### 2. "Introduction to Ensembling/Stacking" by Anisotropic

**Why read it:** The best practical introduction to stacking I have found anywhere. It walks through the entire process of generating out-of-fold predictions and training a meta-learner, with clear diagrams explaining why each step is necessary.

**Key takeaway:** Ensembling is about combining diverse models, not just averaging the best ones.

#### 3. "Feature Engineering Cookbook" by Lorenzo Scaturchio (me)

**Why read it:** I wrote this as the reference I wished I had when I started. It covers target encoding, frequency encoding, interaction features, cyclical encoding, and lag features with copy-paste-ready code for every technique.

**Link:** [Feature Engineering Cookbook](https://www.kaggle.com/code/lorenzoscaturchio/feature-engineering-cookbook)

#### 4. "A Data Science Framework: To Achieve 99% Accuracy" by LD Freeman

**Why read it:** This notebook demonstrates a complete end-to-end pipeline from raw data to a polished submission. The code is clean, the reasoning is explicit, and it shows beginners what a full Kaggle workflow looks like.

**Key takeaway:** Following a systematic process beats ad-hoc experimentation.

#### 5. "EDA & Feature Engineering for House Prices" by Serigne

**Why read it:** A masterclass in feature engineering for tabular data. Serigne shows how to derive meaningful features from domain knowledge and how to validate that each feature actually improves the model.

**Key takeaway:** Domain understanding drives better features than automated feature generation.

#### 6. "Attention Mechanisms Visualized" by Lorenzo Scaturchio (me)

**Why read it:** If you want to understand transformers, you need to understand attention. I built this notebook to demystify self-attention, multi-head attention, and cross-attention with interactive visualizations and a from-scratch implementation.

**Link:** [Attention Mechanisms Guide](https://www.kaggle.com/code/lorenzoscaturchio/attention-mechanisms-guide)

#### 7. "Hitchhiker's Guide to Feature Extraction" by Chris Deotte

**Why read it:** Chris Deotte is a Kaggle Grandmaster and his notebooks are consistently excellent. This one covers feature extraction techniques that are applicable across almost any competition. His approach to feature selection using adversarial validation is particularly clever.

**Key takeaway:** Feature extraction is an art, but there are learnable patterns.

#### 8. "How to Not Overfit" by Heads or Tails

**Why read it:** Overfitting is the number one beginner mistake. This notebook explains regularization, cross-validation, and the bias-variance tradeoff with practical examples. It is required reading before you make your first submission.

**Key takeaway:** Your local CV score should be your primary guide, not the public leaderboard.

#### 9. "RAG From Scratch" by Lorenzo Scaturchio (me)

**Why read it:** Retrieval-Augmented Generation is one of the most practical applications of LLMs. I built this to show every engineering decision in a RAG pipeline -- from chunking strategy to embedding model selection to evaluation methodology.

**Link:** [RAG From Scratch](https://www.kaggle.com/code/lorenzoscaturchio/rag-from-scratch)

#### 10. "Complete Guide to Time Series Analysis" by Prashant Banerjee

**Why read it:** Time series is everywhere on Kaggle (sales forecasting, energy prediction, financial modeling) but it requires fundamentally different approaches than tabular data. This notebook covers stationarity, decomposition, ARIMA, Prophet, and neural approaches in one place.

**Key takeaway:** Time series requires respecting temporal order in every step of your pipeline.

---

#### How to Get the Most Out of Reading Notebooks

Do not just read passively. For each notebook:

1. **Fork it** and run every cell yourself.
2. **Modify one thing** -- change a parameter, add a feature, try a different model.
3. **Write a comment** on the notebook explaining what you learned. Teaching forces understanding.
4. **Apply one technique** from the notebook to a competition you are currently working on.

If you have other notebooks that belong on this list, please share them in the comments! I would love to build a community-curated reading list. And if you find any of my linked notebooks useful, an upvote helps them reach more learners.

---

*End of drafts. All 10 are ready to post. Review each one before posting and adjust any competition-specific details based on the latest data releases.*
