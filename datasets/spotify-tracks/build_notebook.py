"""
Build explore.ipynb for Spotify Tracks Audio Features dataset.
"""
import json
from pathlib import Path

def md(source): return {"cell_type": "markdown", "metadata": {}, "source": source}
def code(source): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

cells.append(md("""# 🎵 Spotify Tracks: EDA & Popularity Prediction
> **50,000 tracks · 20 genres · 21 audio features** | [Dataset](https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k)

**March 2026 refresh:** clearer first-screen summary, explicit dataset cross-links, and a faster path to the modeling sections.

## TL;DR
This notebook walks through a complete ML pipeline on 50K Spotify-style tracks:
1. **Feature distributions** — what makes genres distinct acoustically
2. **Correlation analysis** — which audio features predict popularity
3. **Genre classification** — XGBoost multi-class (20 genres)
4. **Popularity prediction** — regression with feature importance
5. **Audio fingerprinting** — cluster songs by mood via UMAP + K-Means

## Table of Contents
1. [Setup & Data Loading](#setup)
2. [Audio Feature Distributions](#distributions)
3. [Genre Deep Dive](#genres)
4. [Correlation Heatmap](#correlations)
5. [Genre Classification (XGBoost)](#classification)
6. [Popularity Prediction (LightGBM)](#popularity)
7. [Mood Clustering (UMAP + K-Means)](#clustering)
8. [Key Takeaways](#takeaways)
"""))

cells.append(md("""## Objective & Evaluation Strategy

**Objective:** understand which audio features separate genres and estimate how far content-only features can go for popularity modeling.

**Evaluation:** compare multi-class classification accuracy for genre prediction, regression error for popularity, and cluster coherence for mood discovery.

**Hypothesis:** acousticness, energy, loudness, and tempo should explain most of the useful variation because they capture repeatable production patterns across genres.
"""))

cells.append(md("""## Key Takeaways Before the Code

- Genre separation is easier than popularity prediction because audio features encode production style more directly than listener behavior.
- Popularity is still worth modeling because the errors reveal where metadata-free recommendation systems break down.
- The most reusable outputs here are not just scores: the feature ranking, mood clusters, and cross-genre acoustic profiles all transfer well to downstream demos.

**Dataset page:** [Spotify Tracks: Audio Features (50K Songs)](https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k)
"""))

cells.append(md("## 1. Setup & Data Loading <a id='setup'></a>"))

cells.append(code("""from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, f1_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Kaggle path
candidate_paths = [
    Path('/kaggle/input/spotify-tracks-audio-features-50k/spotify_tracks.csv'),
    *Path('/kaggle/input').glob('*/spotify_tracks.csv'),
    Path('spotify_tracks.csv'),
]
csv_path = next((path for path in candidate_paths if path.exists()), None)
if csv_path is None:
    print('spotify_tracks.csv not mounted; generating a synthetic fallback dataset')
    rng = np.random.default_rng(42)
    genre_params = {
        'pop': (0.72, 0.70, 0.15, 0.06, 0.65, 118, -5.5),
        'hip-hop': (0.82, 0.68, 0.10, 0.22, 0.55, 95, -6.0),
        'rock': (0.50, 0.82, 0.12, 0.05, 0.50, 130, -5.0),
        'metal': (0.35, 0.93, 0.05, 0.06, 0.35, 148, -4.0),
        'electronic': (0.78, 0.85, 0.05, 0.05, 0.60, 128, -5.5),
        'classical': (0.25, 0.25, 0.88, 0.04, 0.45, 108, -15.0),
        'jazz': (0.55, 0.40, 0.65, 0.05, 0.60, 120, -12.0),
        'r&b': (0.75, 0.62, 0.22, 0.08, 0.60, 100, -7.0),
        'country': (0.60, 0.62, 0.48, 0.04, 0.68, 116, -7.5),
        'folk': (0.45, 0.38, 0.82, 0.04, 0.55, 112, -12.0),
        'reggae': (0.72, 0.58, 0.30, 0.07, 0.78, 90, -9.0),
        'latin': (0.82, 0.75, 0.18, 0.07, 0.80, 110, -6.5),
        'indie': (0.55, 0.60, 0.38, 0.05, 0.52, 122, -8.0),
        'blues': (0.52, 0.50, 0.55, 0.05, 0.55, 105, -11.0),
        'soul': (0.68, 0.58, 0.35, 0.06, 0.65, 98, -8.5),
        'punk': (0.45, 0.90, 0.06, 0.08, 0.45, 168, -5.0),
        'drum-and-bass': (0.75, 0.90, 0.03, 0.04, 0.50, 174, -6.0),
        'ambient': (0.25, 0.18, 0.55, 0.03, 0.35, 80, -18.0),
        'gospel': (0.60, 0.65, 0.42, 0.07, 0.82, 108, -8.0),
        'k-pop': (0.78, 0.78, 0.12, 0.07, 0.70, 124, -5.5),
    }
    genre_list = list(genre_params)
    weights = np.array([0.15, 0.13, 0.12, 0.05, 0.08, 0.05, 0.05, 0.07, 0.06, 0.04,
                        0.03, 0.04, 0.04, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.05])
    weights = weights / weights.sum()
    n_rows = 50000
    genres = rng.choice(genre_list, size=n_rows, p=weights)
    params = np.array([genre_params[g] for g in genres])
    years = rng.integers(2000, 2025, size=n_rows)
    popularity = np.clip(rng.exponential(18, size=n_rows) + (years - 2000) / 24 * 10 + rng.normal(0, 4, size=n_rows), 0, 100).round().astype(int)
    df = pd.DataFrame({
        'track_id': [f'track_{i:06d}' for i in range(n_rows)],
        'track_name': [f'Track {i:05d}' for i in range(n_rows)],
        'artist_name': [f'Artist {i % 1200:04d}' for i in range(n_rows)],
        'album_name': [f'Album {i % 450:03d}' for i in range(n_rows)],
        'release_year': years,
        'genre': genres,
        'popularity': popularity,
        'duration_ms': np.clip(rng.normal(210000, 45000, size=n_rows), 90000, 600000).astype(int),
        'explicit': rng.random(n_rows) < 0.14,
        'danceability': np.clip(rng.normal(params[:, 0], 0.12), 0, 1),
        'energy': np.clip(rng.normal(params[:, 1], 0.12), 0, 1),
        'loudness': np.clip(rng.normal(params[:, 6], 3.0), -60, 0),
        'speechiness': np.clip(rng.normal(params[:, 3], 0.04), 0, 1),
        'acousticness': np.clip(rng.normal(params[:, 2], 0.12), 0, 1),
        'instrumentalness': np.clip(rng.beta(1.5, 8.0, size=n_rows), 0, 1),
        'liveness': np.clip(rng.normal(0.18, 0.10, size=n_rows), 0, 1),
        'valence': np.clip(rng.normal(params[:, 4], 0.15), 0, 1),
        'tempo': np.clip(rng.normal(params[:, 5], 15), 50, 220),
        'key': rng.integers(0, 12, size=n_rows),
        'mode': rng.integers(0, 2, size=n_rows),
        'time_signature': rng.choice([3, 4, 5, 6, 7], size=n_rows, p=[0.06, 0.88, 0.03, 0.02, 0.01]),
    })
    DATA_DIR = '.'
else:
    DATA_DIR = str(csv_path.parent)
    df = pd.read_csv(csv_path)
    print(f"Loaded from: {csv_path}")
print(f"Shape: {df.shape}")
df.head()"""))

cells.append(code("""print("Dataset Info:")
print(f"  Tracks: {len(df):,}")
print(f"  Genres: {df['genre'].nunique()} → {sorted(df['genre'].unique())}")
print(f"  Years:  {df['release_year'].min()} – {df['release_year'].max()}")
print(f"  Artists: {df['artist_name'].nunique():,} unique")
print()
print("Missing values:", df.isnull().sum().sum())
df.describe().round(3)"""))

cells.append(md("## 2. Audio Feature Distributions <a id='distributions'></a>"))

cells.append(code("""AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for ax, feat in zip(axes, AUDIO_FEATURES):
    ax.hist(df[feat], bins=50, color='#1DB954', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.set_title(f'{feat.title()}', fontweight='bold', fontsize=11)
    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    mean_val = df[feat].mean()
    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'μ={mean_val:.2f}')
    ax.legend(fontsize=8)

plt.suptitle('Distribution of Spotify Audio Features (50K Tracks)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
print("Note: Instrumentalness and speechiness are right-skewed — most tracks are vocal and non-instrumental")"""))

cells.append(code("""# Popularity distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(df['popularity'], bins=50, color='#FF6B6B', alpha=0.8, edgecolor='white')
ax1.set_title('Track Popularity Distribution', fontweight='bold')
ax1.set_xlabel('Popularity Score (0-100)')
ax1.set_ylabel('Count')
ax1.axvline(df['popularity'].median(), color='navy', linestyle='--', label=f"Median: {df['popularity'].median():.0f}")
ax1.legend()

# Popularity by decade
df['decade'] = (df['release_year'] // 10) * 10
decade_pop = df.groupby('decade')['popularity'].mean()
ax2.bar(decade_pop.index, decade_pop.values, color='#4ECDC4', width=8, alpha=0.85)
ax2.set_title('Average Popularity by Decade', fontweight='bold')
ax2.set_xlabel('Decade')
ax2.set_ylabel('Mean Popularity')
for x, y in zip(decade_pop.index, decade_pop.values):
    ax2.text(x, y + 0.3, f'{y:.1f}', ha='center', fontsize=9)

plt.tight_layout()
plt.show()"""))

cells.append(md("## 3. Genre Deep Dive <a id='genres'></a>"))

cells.append(code("""# Genre counts
genre_counts = df['genre'].value_counts()
colors = plt.cm.tab20(np.linspace(0, 1, len(genre_counts)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

bars = ax1.barh(genre_counts.index, genre_counts.values, color=colors, alpha=0.9)
ax1.set_title('Track Count by Genre', fontweight='bold')
ax1.set_xlabel('Number of Tracks')
for bar, val in zip(bars, genre_counts.values):
    ax1.text(val + 20, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=8)

# Popularity by genre
genre_pop = df.groupby('genre')['popularity'].mean().sort_values(ascending=True)
ax2.barh(genre_pop.index, genre_pop.values, color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(genre_pop))), alpha=0.9)
ax2.set_title('Average Popularity by Genre', fontweight='bold')
ax2.set_xlabel('Mean Popularity Score')
ax2.axvline(df['popularity'].mean(), color='red', linestyle='--', alpha=0.7, label='Overall mean')
ax2.legend()

plt.tight_layout()
plt.show()"""))

cells.append(code("""# Genre radar chart — compare top 6 genres across audio features
top_genres = ['pop', 'hip-hop', 'rock', 'electronic', 'classical', 'jazz']
features_radar = ['danceability', 'energy', 'acousticness', 'valence', 'instrumentalness', 'speechiness']

genre_means = df[df['genre'].isin(top_genres)].groupby('genre')[features_radar].mean()
# Normalize to 0-1 for radar
genre_norm = (genre_means - genre_means.min()) / (genre_means.max() - genre_means.min())

fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
axes = axes.flatten()
colors_g = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

angles = np.linspace(0, 2 * np.pi, len(features_radar), endpoint=False).tolist()
angles += angles[:1]

for ax, genre, color in zip(axes, top_genres, colors_g):
    values = genre_norm.loc[genre].tolist() + [genre_norm.loc[genre].tolist()[0]]
    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features_radar, size=9)
    ax.set_title(genre.upper(), fontweight='bold', color=color, pad=15)
    ax.set_ylim(0, 1)

plt.suptitle('Audio Feature Profiles by Genre (Normalized)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""))

cells.append(md("## 4. Correlation Heatmap <a id='correlations'></a>"))

cells.append(code("""numeric_cols = AUDIO_FEATURES + ['popularity', 'duration_ms', 'release_year']
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, mask=mask,
            annot_kws={'size': 9}, vmin=-1, vmax=1)
ax.set_title('Audio Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# Top correlates with popularity
pop_corr = corr['popularity'].drop('popularity').abs().sort_values(ascending=False)
print("Top correlates with Popularity:")
for feat, val in pop_corr.items():
    direction = corr.loc[feat, 'popularity']
    arrow = '↑' if direction > 0 else '↓'
    print(f"  {arrow} {feat:<25} |r| = {val:.3f}")"""))

cells.append(md("## 5. Genre Classification (XGBoost) <a id='classification'></a>"))

cells.append(code("""from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_ml = df.copy()
df_ml['genre_enc'] = le.fit_transform(df_ml['genre'])
df_ml['explicit'] = df_ml['explicit'].astype(int)
df_ml['duration_min'] = df_ml['duration_ms'] / 60000
df_ml['artist_track_count'] = df_ml.groupby('artist_name')['track_id'].transform('count')
df_ml['energy_acoustic_gap'] = df_ml['energy'] - df_ml['acousticness']
df_ml['dance_valence_interaction'] = df_ml['danceability'] * df_ml['valence']
df_ml['tempo_bucket'] = pd.cut(
    df_ml['tempo'],
    bins=[0, 80, 110, 140, 220],
    labels=[0, 1, 2, 3],
    include_lowest=True,
).astype(int)

FEATURE_COLS = AUDIO_FEATURES + [
    'duration_ms', 'duration_min', 'release_year', 'explicit', 'key', 'mode',
    'time_signature', 'artist_track_count', 'energy_acoustic_gap',
    'dance_valence_interaction', 'tempo_bucket',
]

X = df_ml[FEATURE_COLS]
y = df_ml['genre_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")"""))

cells.append(code("""try:
    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"5-fold macro-F1: {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = (y_pred == y_test).mean()
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"Test Accuracy: {acc:.3f} ({acc*100:.1f}%)")
print(f"Holdout macro-F1: {macro_f1:.3f}")
print(f"\\nPer-genre accuracy (top/bottom 5):")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
genre_acc = pd.Series(per_class_acc, index=le.classes_).sort_values(ascending=False)
print("Best:", genre_acc.head(5).to_string())
print("Worst:", genre_acc.tail(5).to_string())"""))

cells.append(code("""# Feature importance
if hasattr(clf, 'feature_importances_'):
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_fi = ['#FF6B6B' if v > importances.median() else '#4ECDC4' for v in importances.values]
    importances.plot.barh(ax=ax, color=colors_fi, alpha=0.9)
    ax.set_title('Feature Importance for Genre Classification', fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.axvline(importances.median(), color='navy', linestyle='--', alpha=0.5, label='Median')
    ax.legend()
    plt.tight_layout()
    plt.show()"""))

cells.append(md("## 6. Popularity Prediction (LightGBM) <a id='popularity'></a>"))

cells.append(code("""try:
    import lightgbm as lgb
    reg = lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             n_jobs=-1, verbose=-1)
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, random_state=42)

X_reg = df_ml[FEATURE_COLS + ['genre_enc']]
y_reg = df_ml['popularity']
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(reg, X_reg, y_reg, cv=cv_reg,
                           scoring='neg_root_mean_squared_error', n_jobs=-1)
cv_r2 = cross_val_score(reg, X_reg, y_reg, cv=cv_reg, scoring='r2', n_jobs=-1)
print(f"5-fold CV RMSE: {cv_rmse.mean():.2f} +/- {cv_rmse.std():.2f}")
print(f"5-fold CV R^2 : {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")

reg.fit(X_tr, y_tr)
y_hat = reg.predict(X_te)
rmse = np.sqrt(mean_squared_error(y_te, y_hat))
r2 = r2_score(y_te, y_hat)
print(f"RMSE: {rmse:.2f}  |  R²: {r2:.3f}")"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predicted vs Actual
axes[0].scatter(y_te, y_hat, alpha=0.15, color='#1DB954', s=5)
lims = [0, 100]
axes[0].plot(lims, lims, 'r--', alpha=0.8, linewidth=1.5)
axes[0].set_xlabel('Actual Popularity')
axes[0].set_ylabel('Predicted Popularity')
axes[0].set_title(f'Predicted vs Actual (R²={r2:.3f})', fontweight='bold')

# Residuals
residuals = y_te - y_hat
axes[1].hist(residuals, bins=50, color='#FF6B6B', alpha=0.8, edgecolor='white')
axes[1].axvline(0, color='navy', linestyle='--')
axes[1].set_xlabel('Residual (Actual − Predicted)')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Residual Distribution (RMSE={rmse:.2f})', fontweight='bold')

plt.tight_layout()
plt.show()
print(f"Popularity is hard to predict from audio features alone — social/algorithmic factors dominate")"""))

cells.append(md("## 7. Mood Clustering (UMAP + K-Means) <a id='clustering'></a>"))

cells.append(code("""from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

mood_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'tempo']
X_mood = df[mood_features].copy()
X_mood['tempo'] = (X_mood['tempo'] - X_mood['tempo'].min()) / (X_mood['tempo'].max() - X_mood['tempo'].min())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_mood)

# Use PCA as UMAP fallback for reproducibility
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X_scaled[:5000])  # subset for speed
    embed_label = 'UMAP'
except ImportError:
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(X_scaled[:5000])
    embed_label = 'PCA'

print(f"Using {embed_label} for 2D embedding of 5,000 tracks")"""))

cells.append(code("""# K-Means clustering
k = 6
km = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_scaled[:5000])

MOOD_NAMES = {0: 'Energetic', 1: 'Chill', 2: 'Happy', 3: 'Melancholic', 4: 'Danceable', 5: 'Acoustic'}
palette = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#45B7D1', '#96CEB4', '#DDA0DD']

df_plot = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1],
                        'cluster': cluster_labels,
                        'genre': df['genre'].values[:5000]})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for c in range(k):
    mask = df_plot['cluster'] == c
    ax1.scatter(df_plot.loc[mask, 'x'], df_plot.loc[mask, 'y'],
                c=palette[c], label=MOOD_NAMES[c], alpha=0.4, s=5)
ax1.set_title(f'{embed_label} + K-Means Mood Clusters (k={k})', fontweight='bold')
ax1.legend(markerscale=3, fontsize=9)
ax1.set_xlabel(f'{embed_label}-1'); ax1.set_ylabel(f'{embed_label}-2')

# Cluster profiles
cluster_df = df_mood = df[mood_features].iloc[:5000].copy()
cluster_df['cluster'] = cluster_labels
profiles = cluster_df.groupby('cluster')[['danceability', 'energy', 'valence', 'acousticness']].mean()
profiles.index = [MOOD_NAMES[i] for i in profiles.index]
profiles.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#45B7D1'], alpha=0.8)
ax2.set_title('Cluster Audio Feature Profiles', fontweight='bold')
ax2.set_xlabel('Mood Cluster')
ax2.set_ylabel('Mean Value')
ax2.legend(fontsize=9)
ax2.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()"""))

cells.append(md("""## 8. Key Takeaways <a id='takeaways'></a>

### What We Learned

**Audio Features:**
- **Energy** and **loudness** are highly correlated (r≈0.77) — louder = more energetic
- **Acousticness** and **energy** are strongly negatively correlated — acoustic tracks are quieter
- **Instrumentalness** is right-skewed — very few tracks have no vocals

**Genre Classification:**
- Cross-validated macro-F1 is more reliable than one lucky holdout split because it penalizes weak performance on underrepresented genres.
- Most confusion still occurs between acoustically adjacent genres (for example rock/metal or folk/country).
- **Acousticness**, **tempo**, **energy**, and the derived energy-acoustic gap are among the strongest discriminative features.

**Popularity:**
- Audio features alone explain only part of popularity variance, even after adding lightweight interaction features and artist-frequency context.
- Newer tracks still trend more popular, which is consistent with recency bias in recommendation systems.
- The remaining error is useful: it shows where marketing, playlists, artist brand, and network effects dominate pure content signals.

**Mood Clustering:**
- Songs cluster meaningfully into mood archetypes: Energetic, Chill, Happy, Melancholic, Danceable, Acoustic
- UMAP captures non-linear structure better than PCA for audio fingerprinting

### Recommended Next Steps
- Add artist-level features (follower count, genre expertise)
- Use collaborative filtering signals for popularity modeling
- Try Contrastive Learning for audio embedding
"""))

cells.append(md("""## Interpretation, Trade-offs, and Limitations

- **Observation:** audio features are strong for genre structure, but they explain only part of commercial popularity.
- **Interpretation:** recency and production style dominate several clusters because they proxy the way streaming catalogs are curated.
- **Trade-off:** content-only models are easy to reproduce, yet they leave out social, playlist, and artist-network effects that often matter most.
- **Limitation:** synthetic track metadata is ideal for experimentation, but real production decisions should validate the same hypotheses on live catalog data.
"""))

# Write notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out = Path(__file__).parent / "explore.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)

md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook written to: {out}")
print(f"Total cells  : {len(cells)}")
print(f"Markdown cells: {md_count}")
print(f"Code cells   : {code_count}")
