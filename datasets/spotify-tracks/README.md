# Spotify Tracks: Audio Features (50K Songs)

> 50,000 synthetic Spotify-style tracks for genre modeling, popularity regression, clustering, and recommender demos.

**Kaggle dataset:** [lorenzoscaturchio/spotify-tracks-audio-features-50k](https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k)  
**Companion notebook:** [Spotify Tracks EDA & Popularity Prediction](https://www.kaggle.com/code/lorenzoscaturchio/spotify-tracks-eda-popularity-prediction)  
**License:** GPL-3.0

## Why this dataset is useful

This dataset is built to behave like a compact Spotify audio-features export without depending on proprietary user data. It gives you enough scale to train real baseline models, enough genre variety to make clustering interesting, and enough feature richness to build portfolio-quality visualizations quickly.

The data spans 2000 through 2024 and covers 20 genres. Each row contains a track identifier, artist and album metadata, a popularity target, and the core audio descriptors most people expect from Spotify-style feature tables.

## File Summary

- `spotify_tracks.csv`
- Rows: `50,000`
- Columns: `21`
- File size: `13.35 MB`
- Coverage: `2000-01-01` to `2024-12-31`
- Geography: `Global (synthetic)`

## Column Groups

### Identity and metadata

- `track_id`, `track_name`, `artist_name`, `album_name`
- `release_year`, `genre`, `explicit`

### Targets and ranking-style fields

- `popularity`

### Audio features

- `danceability`, `energy`, `loudness`, `speechiness`
- `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- `key`, `mode`, `time_signature`, `duration_ms`

## Practical Use Cases

- Genre classification from structured audio features
- Popularity regression without lyric or playlist metadata
- Mood clustering with PCA, UMAP, or K-Means
- Feature-importance demos for tree models and boosting methods
- Recommender-system prototypes based on acoustic similarity

## Linked Kaggle Assets

- Dataset page: <https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k>
- Explore notebook: <https://www.kaggle.com/code/lorenzoscaturchio/spotify-tracks-eda-popularity-prediction>

## Provenance

- Synthetic data generated from repository scripts in this project
- Built with seeded statistical rules and genre-aware feature distributions
- Intended for education, prototyping, benchmarking, and demos

## Citation

Scaturchio, Lorenzo (2026). *Spotify Tracks: Audio Features (50K Songs).* Kaggle Dataset. <https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k>
