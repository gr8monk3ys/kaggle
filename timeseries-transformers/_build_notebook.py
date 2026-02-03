#!/usr/bin/env python3
"""Build the timeseries_transformers.ipynb notebook programmatically."""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def code(source):
    return {"cell_type": "code", "metadata": {"trusted": True}, "source": source.split("\n"), "outputs": [], "execution_count": None}

cells = []

# ── Cell 1: Title Banner ─────────────────────────────────────────────────────
cells.append(md("""\
# <center>Time Series Forecasting with Transformers</center>

<center>

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)

</center>

---

**Author:** Lorenzo Scaturchio | **Last Updated:** January 2026

A deep-dive, from-scratch guide to applying Transformer architectures for time series forecasting -- covering theory, implementation, modern methods (PatchTST, Informer, Autoformer), and production-ready tips.

> **Relevant Competitions:**
> - [Hull Tactical Short-Term Market Prediction](https://www.kaggle.com/competitions/hull-tactical-asset-allocation)
> - [Store Sales -- Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
> - [G-Research Crypto Forecasting](https://www.kaggle.com/competitions/g-research-crypto-forecasting)
"""))

# ── Cell 2: TL;DR ────────────────────────────────────────────────────────────
cells.append(md("""\
## TL;DR

| What | Detail |
|------|--------|
| **Goal** | Forecast future values of a time series using Transformer models |
| **Key Insight** | Self-attention captures long-range temporal dependencies that RNNs struggle with |
| **Methods Covered** | Vanilla Transformer, PatchTST, Informer, Autoformer, HuggingFace `TimeSeriesTransformerModel` |
| **Stack** | PyTorch, HuggingFace Transformers, scikit-learn, pandas, matplotlib |
| **Takeaway** | Transformers are now state-of-the-art for many forecasting tasks when properly configured |
"""))

# ── Cell 3: Table of Contents ────────────────────────────────────────────────
cells.append(md("""\
## Table of Contents

1. [Setup & Imports](#1)
2. [Why Transformers for Time Series?](#2)
3. [Time Series Fundamentals](#3)
4. [Positional Encoding for Time](#4)
5. [Building a Transformer from Scratch](#5)
6. [Training Pipeline](#6)
7. [Modern Approaches](#7)
8. [Using HuggingFace TimeSeriesTransformer](#8)
9. [Evaluation](#9)
10. [Production Tips](#10)
11. [Further Reading](#11)
"""))

# ── Cell 4: Section 1 Header ─────────────────────────────────────────────────
cells.append(md("""\
---
<a id="1"></a>
## 1. Setup & Imports

We install and import everything we need. GPU is recommended for the training sections.
"""))

# ── Cell 5: Imports ───────────────────────────────────────────────────────────
cells.append(code("""\
# ============================================================
# 1. Setup & Imports
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
"""))

# ── Cell 6: Callout ──────────────────────────────────────────────────────────
cells.append(md("""\
> **Key Takeaway -- Device Setup**
>
> Always set seeds for reproducibility and check for GPU availability early.
> Time series Transformers benefit significantly from GPU acceleration during training.
"""))

# ── Cell 7: Section 2 Header ─────────────────────────────────────────────────
cells.append(md("""\
---
<a id="2"></a>
## 2. Why Transformers for Time Series?

### Limitations of Traditional Methods

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **ARIMA** | Solid statistical foundation, interpretable | Assumes linearity & stationarity; poor with long sequences |
| **LSTM/GRU** | Handles non-linearity; sequential by design | Vanishing gradients on long sequences; slow training (no parallelism) |
| **Prophet** | Easy to use; handles holidays/seasonality | Limited expressiveness; univariate only |
| **Transformer** | Parallel training; captures long-range deps via attention; multi-variate native | Needs more data; careful positional encoding; quadratic attention cost |

### How Self-Attention Captures Long-Range Dependencies

In an LSTM, information from time step $t_0$ must propagate through every hidden state to reach $t_N$, leading to degraded signals over long horizons. Transformers compute **direct pairwise attention** between all time steps:

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V$$

This means step $t_0$ can directly attend to $t_N$ regardless of sequence length -- a game-changer for seasonal patterns with long periods.
"""))

# ── Cell 8: Section 2 Code -- quick comparison illustration ──────────────────
cells.append(code("""\
# ============================================================
# 2. Visual: Information Flow -- LSTM vs Transformer
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# LSTM information path
ax = axes[0]
n_steps = 8
for i in range(n_steps - 1):
    ax.annotate("", xy=(i + 1, 0), xytext=(i, 0),
                arrowprops=dict(arrowstyle="->", lw=2, color="steelblue"))
ax.scatter(range(n_steps), [0]*n_steps, s=200, zorder=5, color="steelblue")
for i in range(n_steps):
    ax.text(i, 0.15, f"$t_{i}$", ha="center", fontsize=11)
ax.set_xlim(-0.5, n_steps - 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_title("LSTM: Sequential Information Flow", fontsize=13, fontweight="bold")
ax.axis("off")

# Transformer information path
ax = axes[1]
ax.scatter(range(n_steps), [0]*n_steps, s=200, zorder=5, color="darkorange")
for i in range(n_steps):
    ax.text(i, 0.18, f"$t_{i}$", ha="center", fontsize=11)
# Show direct connections from t0 to all others
for j in range(1, n_steps):
    ax.annotate("", xy=(j, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="darkorange", alpha=0.6,
                                connectionstyle=f"arc3,rad={0.15 * j / n_steps}"))
ax.set_xlim(-0.5, n_steps - 0.5)
ax.set_ylim(-0.6, 0.6)
ax.set_title("Transformer: Direct Attention", fontsize=13, fontweight="bold")
ax.axis("off")

plt.tight_layout()
plt.show()
"""))

# ── Cell 9: Callout ──────────────────────────────────────────────────────────
cells.append(md("""\
> **Key Takeaway -- Why Transformers?**
>
> Transformers overcome the sequential bottleneck of RNNs by computing all pairwise
> relationships in parallel. For time series with long seasonal cycles (e.g., yearly
> patterns in daily data = 365 steps), this is a major advantage.
"""))

# ── Cell 10: Section 3 Header ────────────────────────────────────────────────
cells.append(md("""\
---
<a id="3"></a>
## 3. Time Series Fundamentals

We generate a **synthetic** time series with:
- **Trend**: slow linear growth
- **Seasonality**: a weekly cycle + a yearly cycle
- **Noise**: Gaussian random noise

This controlled setup lets us verify our model captures each component.
"""))

# ── Cell 11: Generate Synthetic Data ─────────────────────────────────────────
cells.append(code("""\
# ============================================================
# 3. Generate Synthetic Time Series Data
# ============================================================
n_points = 2000
t = np.arange(n_points)

# Components
trend = 0.005 * t
weekly_season = 3.0 * np.sin(2 * np.pi * t / 7)
yearly_season = 5.0 * np.sin(2 * np.pi * t / 365.25)
noise = np.random.normal(0, 0.8, n_points)

# Composite signal
y = trend + weekly_season + yearly_season + noise

# Put into a DataFrame with date index
dates = pd.date_range(start="2020-01-01", periods=n_points, freq="D")
df = pd.DataFrame({"date": dates, "value": y})
df.set_index("date", inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
df.head()
"""))

# ── Cell 12: Visualize Synthetic Data ────────────────────────────────────────
cells.append(code("""\
# ============================================================
# Visualization: Components + Composite
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

axes[0].plot(dates, trend, color="tab:blue")
axes[0].set_title("Trend Component", fontweight="bold")
axes[0].set_ylabel("Value")

axes[1].plot(dates, weekly_season, color="tab:orange")
axes[1].set_title("Weekly Seasonal Component", fontweight="bold")
axes[1].set_ylabel("Value")

axes[2].plot(dates, yearly_season, color="tab:green")
axes[2].set_title("Yearly Seasonal Component", fontweight="bold")
axes[2].set_ylabel("Value")

axes[3].plot(dates, y, color="tab:red", alpha=0.8)
axes[3].set_title("Composite Signal (Trend + Seasonality + Noise)", fontweight="bold")
axes[3].set_ylabel("Value")
axes[3].set_xlabel("Date")

plt.tight_layout()
plt.show()
"""))

# ── Cell 13: Train/Test Split ────────────────────────────────────────────────
cells.append(md("""\
### Train/Test Split Strategy for Time Series

**CRITICAL:** Never use random splits for time series!  Random splits cause **data leakage** -- the model sees future data during training.

| Strategy | When to Use |
|----------|-------------|
| **Fixed cutoff** | Single series; simple and effective |
| **Expanding window** | When you want to simulate real deployment |
| **Sliding window** | When older data is less relevant |
| **Time Series Cross-Validation** | For robust metric estimation |

We use a fixed cutoff at 80/20.
"""))

# ── Cell 14: Split Code ──────────────────────────────────────────────────────
cells.append(code("""\
# ============================================================
# Train/Test Split -- Fixed Cutoff (NO random split!)
# ============================================================
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"Train: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} days)")
print(f"Test:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} days)")

# Normalize using ONLY training statistics (avoid leakage)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[["value"]]).flatten()
test_scaled = scaler.transform(test_df[["value"]]).flatten()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(train_df.index, train_scaled, label="Train", color="steelblue")
ax.plot(test_df.index, test_scaled, label="Test", color="darkorange")
ax.axvline(train_df.index[-1], color="red", linestyle="--", label="Split Point")
ax.legend()
ax.set_title("Scaled Train / Test Split", fontweight="bold")
plt.tight_layout()
plt.show()
"""))

# ── Cell 15: Callout ─────────────────────────────────────────────────────────
cells.append(md("""\
> **Key Takeaway -- No Random Splits!**
>
> Time series data has temporal ordering. Always split chronologically and fit
> scalers on **training data only**, then transform test data with those same
> parameters. This prevents data leakage.
"""))

# ── Cell 16: Section 4 Header ────────────────────────────────────────────────
cells.append(md("""\
---
<a id="4"></a>
## 4. Positional Encoding for Time

Transformers have **no inherent notion of order**. We must inject positional information explicitly. For time series, this is doubly important because *when* something happens matters as much as *what* happens.

### Two Approaches:
1. **Sinusoidal Positional Encoding** -- the classic approach from "Attention Is All You Need"
2. **Temporal Embeddings** -- learnable embeddings for hour-of-day, day-of-week, month-of-year, etc.
"""))

# ── Cell 17: Positional Encoding Implementation ──────────────────────────────
cells.append(code("""\
# ============================================================
# 4. Positional Encoding -- Sinusoidal
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    \"\"\"
    Classic sinusoidal positional encoding from Vaswani et al. (2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This gives each position a unique signature that the model can use
    to understand ordering and relative distances.
    \"\"\"

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                      # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float() # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                         # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model) -- batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"x: (batch, seq_len, d_model)\"\"\"
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# Temporal Feature Embeddings
# ============================================================
class TemporalEmbedding(nn.Module):
    \"\"\"
    Learnable embeddings for calendar features.
    Encodes hour-of-day, day-of-week, and month-of-year.
    \"\"\"

    def __init__(self, d_model: int):
        super().__init__()
        self.hour_embed  = nn.Embedding(24, d_model)
        self.dow_embed   = nn.Embedding(7,  d_model)
        self.month_embed = nn.Embedding(12, d_model)

    def forward(self, hour: torch.Tensor, dow: torch.Tensor, month: torch.Tensor) -> torch.Tensor:
        \"\"\"Each input: (batch, seq_len) of integer indices.\"\"\"
        return self.hour_embed(hour) + self.dow_embed(dow) + self.month_embed(month)


print("Positional encoding classes defined.")
"""))

# ── Cell 18: Visualize Positional Encoding ───────────────────────────────────
cells.append(code("""\
# ============================================================
# Visualize Sinusoidal Positional Encoding
# ============================================================
d_model = 64
pe_layer = SinusoidalPositionalEncoding(d_model=d_model, max_len=200, dropout=0.0)

# Extract the encoding matrix
pe_matrix = pe_layer.pe.squeeze(0).numpy()  # (200, 64)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
im = axes[0].imshow(pe_matrix[:100, :], aspect="auto", cmap="RdBu_r", interpolation="nearest")
axes[0].set_xlabel("Encoding Dimension")
axes[0].set_ylabel("Position (time step)")
axes[0].set_title("Positional Encoding Heatmap", fontweight="bold")
plt.colorbar(im, ax=axes[0])

# Individual dimensions
for dim_idx in [0, 1, 4, 5, 10, 11]:
    axes[1].plot(pe_matrix[:100, dim_idx], label=f"dim {dim_idx}", alpha=0.8)
axes[1].set_xlabel("Position (time step)")
axes[1].set_ylabel("Encoding Value")
axes[1].set_title("Individual Encoding Dimensions", fontweight="bold")
axes[1].legend(ncol=2, fontsize=9)

plt.tight_layout()
plt.show()
"""))

# ── Cell 19: Callout ─────────────────────────────────────────────────────────
cells.append(md("""\
> **Key Takeaway -- Positional Encoding**
>
> The sinusoidal encoding creates a unique "fingerprint" for each position, and nearby
> positions have similar encodings. The different frequencies across dimensions let the
> model reason about both short-range and long-range relative positions. For time series,
> adding **calendar embeddings** (day-of-week, month, etc.) further helps the model
> recognize cyclical patterns.
"""))

# ── Cell 20: Section 5 Header ────────────────────────────────────────────────
cells.append(md("""\
---
<a id="5"></a>
## 5. Building a Transformer from Scratch

We implement a complete Time Series Transformer in PyTorch, building up from:
1. **Scaled Dot-Product Attention**
2. **Multi-Head Attention**
3. **Transformer Encoder Layer** (attention + feed-forward + layer norm)
4. **Full TimeSeriesTransformer** model

All code is heavily commented for educational clarity.
"""))

# ── Cell 21: Transformer Implementation ──────────────────────────────────────
cells.append(code("""\
# ============================================================
# 5. Time Series Transformer -- Built from Scratch
# ============================================================

# ── 5a. Scaled Dot-Product Attention ─────────────────────────
def scaled_dot_product_attention(
    query: torch.Tensor,   # (batch, heads, seq_len, d_k)
    key: torch.Tensor,     # (batch, heads, seq_len, d_k)
    value: torch.Tensor,   # (batch, heads, seq_len, d_v)
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    \"\"\"
    Compute attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Returns:
        output: (batch, heads, seq_len, d_v)
        attn_weights: (batch, heads, seq_len, seq_len)
    \"\"\"
    d_k = query.size(-1)
    # QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (e.g., causal mask for autoregressive forecasting)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


# ── 5b. Multi-Head Attention ─────────────────────────────────
class MultiHeadAttention(nn.Module):
    \"\"\"
    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.

    Instead of one attention function with d_model dimensions, we project
    Q, K, V into h heads of d_k dimensions each.
    \"\"\"

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # 1. Linear projections: (batch, seq, d_model) -> (batch, seq, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Reshape to (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Scaled dot-product attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 4. Concatenate heads: (batch, n_heads, seq, d_k) -> (batch, seq, d_model)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # 5. Final linear projection
        return self.W_o(attn_output)


# ── 5c. Feed-Forward Network ────────────────────────────────
class PositionwiseFeedForward(nn.Module):
    \"\"\"
    Two-layer feed-forward network applied independently to each position.
    FFN(x) = ReLU(xW1 + b1)W2 + b2
    Typically d_ff = 4 * d_model.
    \"\"\"

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ── 5d. Transformer Encoder Layer ────────────────────────────
class TransformerEncoderLayer(nn.Module):
    \"\"\"
    A single encoder layer:
        x -> LayerNorm -> MultiHeadAttention -> Residual ->
             LayerNorm -> FeedForward -> Residual

    We use Pre-LayerNorm (modern best practice) rather than Post-LayerNorm.
    \"\"\"

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm multi-head attention with residual
        normed = self.norm1(x)
        x = x + self.dropout(self.attn(normed, normed, normed, mask))

        # Pre-norm feed-forward with residual
        normed = self.norm2(x)
        x = x + self.dropout(self.ff(normed))

        return x


# ── 5e. Full Time Series Transformer ────────────────────────
class TimeSeriesTransformer(nn.Module):
    \"\"\"
    Complete Transformer for univariate time series forecasting.

    Architecture:
        Input (batch, seq_len, 1)
          -> Linear projection to d_model
          -> Add positional encoding
          -> N x TransformerEncoderLayer
          -> LayerNorm
          -> Linear projection to forecast horizon

    Args:
        input_dim:    Number of input features (1 for univariate)
        d_model:      Model dimension
        n_heads:      Number of attention heads
        n_layers:     Number of encoder layers
        d_ff:         Feed-forward hidden dimension
        seq_len:      Input sequence length (look-back window)
        forecast_len: Number of future steps to predict
        dropout:      Dropout rate
    \"\"\"

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        seq_len: int = 96,
        forecast_len: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.forecast_len = forecast_len

        # Project input features to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=seq_len + forecast_len, dropout=dropout)

        # Transformer encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm (Pre-LN convention)
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection: map from d_model back to forecast
        self.output_projection = nn.Linear(d_model * seq_len, forecast_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            x: (batch, seq_len, input_dim) -- input time series window

        Returns:
            forecast: (batch, forecast_len) -- predicted future values
        \"\"\"
        batch_size = x.size(0)

        # Project input to d_model dimensions
        x = self.input_projection(x)          # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)              # (batch, seq_len, d_model)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)                      # (batch, seq_len, d_model)

        # Final normalization
        x = self.final_norm(x)                # (batch, seq_len, d_model)

        # Flatten and project to forecast horizon
        x = x.reshape(batch_size, -1)         # (batch, seq_len * d_model)
        forecast = self.output_projection(x)  # (batch, forecast_len)

        return forecast


# ── Verify model ─────────────────────────────────────────────
SEQ_LEN = 96
FORECAST_LEN = 24

model = TimeSeriesTransformer(
    input_dim=1, d_model=64, n_heads=4, n_layers=3,
    d_ff=256, seq_len=SEQ_LEN, forecast_len=FORECAST_LEN, dropout=0.1
).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test forward pass
dummy_input = torch.randn(8, SEQ_LEN, 1).to(DEVICE)
dummy_output = model(dummy_input)
print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {dummy_output.shape}")
"""))

# ── Cell 22: Callout ─────────────────────────────────────────────────────────
cells.append(md("""\
> **Key Takeaway -- Architecture Decisions**
>
> - **Pre-LayerNorm** is more stable for training than Post-LayerNorm.
> - **GELU** activation in the feed-forward network outperforms ReLU for most tasks.
> - The output projection flattens the entire encoder output and maps to the forecast horizon.
>   An alternative is to use only the last token's representation, or a decoder cross-attention mechanism.
"""))

# ── Cell 23: Section 6 Header ────────────────────────────────────────────────
cells.append(md("""\
---
<a id="6"></a>
## 6. Training Pipeline

Components:
1. **Sliding Window Dataset** -- creates (input_window, target_window) pairs
2. **DataLoader** -- batches and shuffles (shuffling is OK within windows, not across time for split)
3. **Training Loop** with early stopping
4. **Loss Curves** visualization
"""))

# ── Cell 24: Dataset & DataLoader ────────────────────────────────────────────
cells.append(code("""\
# ============================================================
# 6a. Sliding Window Dataset
# ============================================================
class TimeSeriesDataset(Dataset):
    \"\"\"
    Creates sliding window samples from a 1-D time series.

    Each sample is a tuple of:
        - input_window:  (seq_len, 1) tensor
        - target_window: (forecast_len,) tensor
    \"\"\"

    def __init__(self, data: np.ndarray, seq_len: int, forecast_len: int):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.forecast_len = forecast_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.forecast_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len].unsqueeze(-1)  # (seq_len, 1)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.forecast_len]  # (forecast_len,)
        return x, y


# Create datasets
train_dataset = TimeSeriesDataset(train_scaled, SEQ_LEN, FORECAST_LEN)
test_dataset  = TimeSeriesDataset(test_scaled,  SEQ_LEN, FORECAST_LEN)

print(f"Training samples: {len(train_dataset):,}")
print(f"Test samples:     {len(test_dataset):,}")

# DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# Quick sanity check
x_batch, y_batch = next(iter(train_loader))
print(f"Input batch shape:  {x_batch.shape}")   # (32, 96, 1)
print(f"Target batch shape: {y_batch.shape}")    # (32, 24)
"""))

# ── Cell 25: Training Loop ───────────────────────────────────────────────────
cells.append(code("""\
# ============================================================
# 6b. Training Loop with Early Stopping
# ============================================================
def train_model(
    model, train_loader, test_loader, n_epochs=50, lr=1e-3, patience=7
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        # ── Training ─────────────────────────────────────────
        model.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ── Validation ───────────────────────────────────────
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                epoch_val_loss += loss.item() * x_batch.size(0)

        epoch_val_loss /= len(test_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step()

        # ── Early Stopping ───────────────────────────────────
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {epoch_train_loss:.6f} | "
                f"Val Loss: {epoch_val_loss:.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if epochs_no_improve >= patience:
            print(f"\\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    print(f"\\nBest validation loss: {best_val_loss:.6f}")
    return train_losses, val_losses

# Train!
train_losses, val_losses = train_model(model, train_loader, test_loader, n_epochs=50, lr=1e-3, patience=10)
"""))

# ── Cell 26: Loss Curves ─────────────────────────────────────────────────────
cells.append(code("""\
# ============================================================
# 6c. Training / Validation Loss Curves
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, label="Train Loss", color="steelblue", linewidth=2)
ax.plot(val_losses, label="Validation Loss", color="darkorange", linewidth=2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("MSE Loss", fontsize=12)
ax.set_title("Training & Validation Loss Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# ── Cell 27: Callout ─────────────────────────────────────────────────────────
cells.append(md("""\
> **Key Takeaway -- Training Best Practices**
>
> - **AdamW** with weight decay provides better generalization than vanilla Adam.
> - **Cosine annealing** scheduler smoothly decays the learning rate.
> - **Gradient clipping** (max_norm=1.0) prevents exploding gradients.
> - **Early stopping** saves the best model and prevents overfitting.
"""))

# ── Cell 28: Section 7 Header ────────────────────────────────────────────────
cells.append(md("""\
---
<a id="7"></a>
## 7. Modern Approaches

The vanilla Transformer has O(n^2) attention cost. Recent architectures address this and other issues for time series:

### PatchTST (2023)

```
Input Time Series: [x1, x2, ..., x_96]
                         |
                   Patch (size P, stride S)
                         |
           [patch_1, patch_2, ..., patch_N]
                         |
              Transformer Encoder
                         |
              Linear Head -> Forecast
```

**Key Idea:** Divide the time series into non-overlapping patches (like Vision Transformers divide images into patches). This:
- Reduces sequence length from L to L/P, dramatically cutting attention cost
- Captures local semantic information within each patch
- Achieves SOTA on many benchmarks

### Informer (AAAI 2021 Best Paper)

```
Input -> ProbSparse Attention -> Distilling Layer -> ... -> Forecast
```

**Key Idea:** Replace full attention with **ProbSparse attention** that selects only the top-k most "active" queries (measured by KL-divergence from uniform distribution). Reduces attention from O(n^2) to O(n log n).

### Autoformer (NeurIPS 2021)

```
Input -> Series Decomposition -> Auto-Correlation Mechanism -> Forecast
            |                         |
         Trend                   Seasonal
```

**Key Idea:** Replace attention with an **auto-correlation mechanism** that discovers period-based dependencies by operating in the frequency domain. Built-in series decomposition separates trend from seasonal components.
"""))

# ── Cell 29: Callout ─────────────────────────────────────────────────────────
cells.append(md("""\
> **Key Takeaway -- Modern Transformer Variants**
>
> | Model | Key Innovation | Complexity | Best For |
> |-------|---------------|-----------|----------|
> | **PatchTST** | Patching + channel independence | O((L/P)^2) | Long-horizon, multivariate |
> | **Informer** | ProbSparse attention + distilling | O(L log L) | Very long sequences |
> | **Autoformer** | Auto-correlation in frequency domain | O(L log L) | Strong seasonal patterns |
>
> For competitions like Store Sales Forecasting, PatchTST is often the strongest starting point.
"""))

# ── Cell 30: Section 8 Header ────────────────────────────────────────────────
cells.append(md("""\
---
<a id="8"></a>
## 8. Using HuggingFace TimeSeriesTransformer

HuggingFace provides a ready-to-use `TimeSeriesTransformerModel` with a probabilistic forecast head. This is the fastest path to strong results.
"""))

# ── Cell 31: HuggingFace Code ────────────────────────────────────────────────
cells.append(code("""\
# ============================================================
# 8. HuggingFace TimeSeriesTransformer
# ============================================================
try:
    from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("HuggingFace transformers not available or version too old. Skipping this section.")

if HF_AVAILABLE:
    # ── Configure the model ──────────────────────────────────
    config = TimeSeriesTransformerConfig(
        prediction_length=FORECAST_LEN,           # How far ahead to forecast
        context_length=SEQ_LEN,                    # How much history to use
        input_size=1,                              # Univariate
        # Architecture
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        # Distribution head for probabilistic forecasts
        distribution_output="student_t",
        # Scaling
        scaling="mean",
        # Lags to use as additional features
        lags_sequence=[1, 2, 3, 7, 14, 28],
        num_time_features=1,                       # Number of time features
        num_dynamic_real_features=0,
        num_static_categorical_features=0,
        num_static_real_features=0,
        cardinality=[1],
        embedding_dimension=[2],
    )

    hf_model = TimeSeriesTransformerForPrediction(config)
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"HuggingFace TST Parameters: {hf_params:,}")
    print(f"Config: prediction_length={config.prediction_length}, context_length={config.context_length}")
    print(f"Distribution: {config.distribution_output}")
    print("\\nNote: The HF model outputs a probability distribution (Student-t by default),")
    print("enabling probabilistic forecasting with confidence intervals.")
"""))

# ── Cell 32: HF Note ─────────────────────────────────────────────────────────
cells.append(md("""\
> **Key Takeaway -- HuggingFace TST**
>
> The HuggingFace implementation adds a **probabilistic forecast head** (Student-t or
> Negative Binomial distribution) on top of the Transformer, outputting distribution
> parameters instead of point forecasts. This enables **prediction intervals** out of the box.
>
> For production use, the [GluonTS](https://ts.gluon.ai/) library (which HF integrates with) provides
> data loading utilities, evaluation metrics, and backtesting pipelines.
"""))

# ── Cell 33: Section 9 Header ────────────────────────────────────────────────
cells.append(md("""\
---
<a id="9"></a>
## 9. Evaluation

We evaluate our scratch Transformer using:
- **MAE** (Mean Absolute Error) -- robust to outliers
- **RMSE** (Root Mean Squared Error) -- penalizes large errors
- **MAPE** (Mean Absolute Percentage Error) -- scale-independent

We also compare against simple baselines.
"""))

# ── Cell 34: Evaluation Code ─────────────────────────────────────────────────
cells.append(code("""\
# ============================================================
# 9a. Generate Forecasts from Scratch Transformer
# ============================================================
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        preds = model(x_batch).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y_batch.numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

print(f"Predictions shape: {all_preds.shape}")
print(f"Targets shape:     {all_targets.shape}")
"""))

# ── Cell 35: Metrics ──────────────────────────────────────────────────────────
cells.append(code("""\
# ============================================================
# 9b. Compute Metrics
# ============================================================
def compute_metrics(y_true, y_pred, label=""):
    \"\"\"Compute MAE, RMSE, and MAPE.\"\"\"
    # Flatten for overall metrics
    yt = y_true.flatten()
    yp = y_pred.flatten()

    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    # MAPE: avoid division by zero
    mask = np.abs(yt) > 1e-8
    mape = np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100

    print(f"{label}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ── Transformer metrics ─────────────────────────────────────
transformer_metrics = compute_metrics(all_targets, all_preds, label="Transformer (Ours)")

# ── Baseline: Naive (repeat last value) ─────────────────────
naive_preds = np.tile(
    test_scaled[SEQ_LEN - 1 : SEQ_LEN - 1 + len(all_targets)].reshape(-1, 1),
    (1, FORECAST_LEN),
)
# Clip to correct shape
naive_preds = naive_preds[:len(all_targets)]
naive_metrics = compute_metrics(all_targets, naive_preds, label="\\nNaive Baseline (last value)")

# ── Baseline: Seasonal Naive (repeat value from 7 days ago) ─
seasonal_preds = []
for i in range(len(all_targets)):
    start = i
    # Use value from 7 steps back in the input window
    val = test_scaled[start + SEQ_LEN - 7 : start + SEQ_LEN - 7 + FORECAST_LEN]
    if len(val) == FORECAST_LEN:
        seasonal_preds.append(val)
    else:
        seasonal_preds.append(np.zeros(FORECAST_LEN))
seasonal_preds = np.array(seasonal_preds)[:len(all_targets)]
seasonal_metrics = compute_metrics(all_targets, seasonal_preds, label="\\nSeasonal Naive (7-day lag)")

# ── Summary Table ────────────────────────────────────────────
print("\\n" + "=" * 55)
print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
print("=" * 55)
for name, m in [("Transformer", transformer_metrics),
                ("Naive Baseline", naive_metrics),
                ("Seasonal Naive", seasonal_metrics)]:
    print(f"{name:<25} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['mape']:>7.2f}%")
print("=" * 55)
"""))

# ── Cell 36: Forecast Visualization ──────────────────────────────────────────
cells.append(code("""\
# ============================================================
# 9c. Forecast vs Actual Visualization
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Pick three sample forecast windows
sample_indices = [0, len(all_targets) // 3, 2 * len(all_targets) // 3]

for ax, idx in zip(axes, sample_indices):
    steps = np.arange(FORECAST_LEN)
    ax.plot(steps, all_targets[idx], "o-", label="Actual", color="steelblue", markersize=4)
    ax.plot(steps, all_preds[idx], "s--", label="Transformer", color="darkorange", markersize=4)
    ax.fill_between(steps,
                    all_preds[idx] - 0.5,
                    all_preds[idx] + 0.5,
                    alpha=0.2, color="darkorange", label="Approx. interval")
    ax.set_title(f"Forecast Window (sample index={idx})", fontweight="bold")
    ax.set_xlabel("Forecast Horizon (steps)")
    ax.set_ylabel("Scaled Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""))

# ── Cell 37: Callout ─────────────────────────────────────────────────────────
cells.append(md("""\
> **Key Takeaway -- Evaluation**
>
> Always compare against **naive baselines** (last value, seasonal lag). If your fancy model
> cannot beat a naive baseline, something is wrong with your pipeline. For competition scoring,
> check which metric is used (e.g., RMSLE for Store Sales, weighted RMSE for Hull Tactical).
"""))

# ── Cell 38: Section 10 Header ───────────────────────────────────────────────
cells.append(md("""\
---
<a id="10"></a>
## 10. Production Tips

### Handling Missing Values

```python
# Forward fill + backward fill (simplest)
df["value"] = df["value"].ffill().bfill()

# Interpolation (better for smooth series)
df["value"] = df["value"].interpolate(method="time")

# Learned imputation: add a "missing" indicator feature
df["is_missing"] = df["value"].isna().astype(int)
df["value"] = df["value"].fillna(0)
```

### Multi-Variate Forecasting

For multiple correlated series (e.g., sales across 100 stores):
- **Channel Independence** (PatchTST approach): Treat each variable independently. Surprisingly effective and avoids cross-variable overfitting.
- **Channel Mixing**: Use cross-attention between variables. Better when variables are strongly correlated (e.g., temperature and energy demand).

### Probabilistic Forecasting

Point forecasts are rarely sufficient in production. Options:
1. **Parametric**: Output distribution parameters (mean + variance). Used by HuggingFace TST.
2. **Quantile Regression**: Output multiple quantiles (0.1, 0.5, 0.9). Used in Amazon's DeepAR.
3. **Conformal Prediction**: Distribution-free coverage guarantees.

### Conformal Prediction Intervals

```python
# 1. Compute residuals on calibration set
residuals = np.abs(y_cal_true - y_cal_pred)

# 2. Find quantile for desired coverage (e.g., 90%)
alpha = 0.10
q = np.quantile(residuals, 1 - alpha)

# 3. Prediction interval: [y_pred - q, y_pred + q]
lower = y_test_pred - q
upper = y_test_pred + q
# Guaranteed to cover true value ~90% of the time (finite-sample valid!)
```

### Deployment Checklist

- [ ] Data pipeline handles late-arriving and missing data
- [ ] Model retraining schedule (weekly? monthly?)
- [ ] Monitoring: track forecast error over time for concept drift
- [ ] Fallback: if model fails, use seasonal naive as backup
- [ ] Latency: for real-time forecasting, use ONNX or TorchScript export
"""))

# ── Cell 39: Quick demo of conformal prediction ─────────────────────────────
cells.append(code("""\
# ============================================================
# 10. Quick Demo: Conformal Prediction Intervals
# ============================================================
# Use first half of test predictions as calibration, second half as held-out check
cal_size = len(all_preds) // 2
cal_residuals = np.abs(all_targets[:cal_size] - all_preds[:cal_size])

# Per-horizon quantiles at 90% coverage
alpha = 0.10
quantiles_per_horizon = np.quantile(cal_residuals, 1 - alpha, axis=0)
print(f"Conformal quantiles (90% coverage) per forecast step:\\n{np.round(quantiles_per_horizon, 3)}")

# Check coverage on held-out portion
held_out_targets = all_targets[cal_size:]
held_out_preds = all_preds[cal_size:]
lower = held_out_preds - quantiles_per_horizon
upper = held_out_preds + quantiles_per_horizon

coverage = np.mean((held_out_targets >= lower) & (held_out_targets <= upper))
print(f"\\nEmpirical coverage: {coverage:.2%} (target: {1-alpha:.0%})")

# Visualize one example
fig, ax = plt.subplots(figsize=(12, 5))
idx = 10
steps = np.arange(FORECAST_LEN)
ax.plot(steps, held_out_targets[idx], "o-", label="Actual", color="steelblue", markersize=5)
ax.plot(steps, held_out_preds[idx], "s-", label="Forecast", color="darkorange", markersize=5)
ax.fill_between(steps, lower[idx], upper[idx], alpha=0.3, color="darkorange", label="90% Conformal Interval")
ax.set_xlabel("Forecast Horizon (steps)")
ax.set_ylabel("Scaled Value")
ax.set_title("Conformal Prediction Interval (90% Coverage)", fontweight="bold", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# ── Cell 40: Further Reading ─────────────────────────────────────────────────
cells.append(md("""\
---
<a id="11"></a>
## 11. Further Reading

### Papers
| Paper | Year | Key Contribution |
|-------|------|-----------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Original Transformer architecture |
| [Informer](https://arxiv.org/abs/2012.07436) | 2021 | ProbSparse attention for long sequences |
| [Autoformer](https://arxiv.org/abs/2106.13008) | 2021 | Auto-correlation mechanism |
| [PatchTST](https://arxiv.org/abs/2211.14730) | 2023 | Patching for time series |
| [Are Transformers Effective for TSF?](https://arxiv.org/abs/2205.13504) | 2022 | Critical analysis (DLinear baseline) |
| [iTransformer](https://arxiv.org/abs/2310.06625) | 2024 | Inverted Transformer for multivariate |
| [TimesFM](https://arxiv.org/abs/2310.10688) | 2024 | Foundation model for time series |

### Libraries & Resources
- [HuggingFace Time Series Guide](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)
- [GluonTS](https://ts.gluon.ai/) -- Time series toolkit by Amazon
- [NeuralForecast](https://nixtla.github.io/neuralforecast/) -- Neural forecasting made easy
- [tsai](https://timeseriesai.github.io/tsai/) -- fastai for time series

### Kaggle Competitions for Practice
- [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- [Hull Tactical Short-Term Market Prediction](https://www.kaggle.com/competitions/hull-tactical-asset-allocation)
- [Web Traffic Time Series Forecasting](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting)
- [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
"""))

# ── Cell 41: CTA ─────────────────────────────────────────────────────────────
cells.append(md("""\
---

## Thank You for Reading!

If you found this notebook helpful, please consider **upvoting** it -- it helps others discover the content and motivates me to write more deep-dives.

**Summary of what we covered:**
1. Why Transformers are well-suited for time series forecasting
2. Time series data handling (no random splits!)
3. Positional and temporal encoding strategies
4. A full Transformer built from scratch in PyTorch (~200 lines)
5. Training pipeline with early stopping, gradient clipping, and cosine scheduling
6. Modern architectures: PatchTST, Informer, Autoformer
7. HuggingFace's ready-to-use TimeSeriesTransformer
8. Rigorous evaluation with baselines and proper metrics
9. Production-ready tips: missing data, probabilistic forecasting, conformal prediction

**Connect with me:**
- [GitHub](https://github.com/gr8monk3ys)
- [Kaggle](https://www.kaggle.com/lorenzoscaturchio)

Happy forecasting!
"""))

# ── Assemble notebook ────────────────────────────────────────────────────────
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12",
            "mimetype": "text/x-python",
            "file_extension": ".py",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": []
}

# Fix cells: ensure source is a list of lines (each ending with \n except last)
for cell in cells:
    if isinstance(cell["source"], str):
        raw = cell["source"]
    else:
        raw = "\n".join(cell["source"])
    lines = raw.split("\n")
    # Add \n to all lines except the last
    formatted = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        formatted.append(lines[-1])
    cell["source"] = formatted
    notebook["cells"].append(cell)

out_path = "/Users/gr8monk3ys/code/ml-portfolio/kaggle/timeseries-transformers/timeseries_transformers.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {out_path}")
print(f"Total cells: {len(notebook['cells'])}")
md_count = sum(1 for c in notebook['cells'] if c['cell_type'] == 'markdown')
code_count = sum(1 for c in notebook['cells'] if c['cell_type'] == 'code')
print(f"  Markdown: {md_count}")
print(f"  Code:     {code_count}")
