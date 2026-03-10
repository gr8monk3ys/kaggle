# Spotify Tracks: Audio Features (50K Songs)

> 50,000 synthetic Spotify-style tracks for genre classification, popularity regression, clustering, and recommendation demos.

**Kaggle dataset:** [lorenzoscaturchio/spotify-tracks-audio-features-50k](https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k)  
**Companion notebook:** [Spotify Tracks EDA & Popularity Prediction](https://www.kaggle.com/code/lorenzoscaturchio/spotify-tracks-eda-popularity-prediction)  
**License:** GPL-3.0

## Overview

This dataset is built to feel like a compact export from Spotify's audio-feature ecosystem without relying on proprietary listener data. It is large enough for real baseline models, small enough for fast iteration, and structured for portfolio-quality visualizations.

Each row represents one synthetic track with:
- track identity and release metadata
- a primary genre label
- a popularity target on a 0-100 scale
- the core continuous audio descriptors most practitioners expect from Spotify-style feature tables

## Quick Facts

| Property | Value |
|---|---|
| File | `spotify_tracks.csv` |
| Rows | `50,000` |
| Columns | `21` |
| Coverage | `2000-01-01` to `2024-12-31` |
| Genres | `20` |
| Geography | `Global (synthetic)` |

## Recommended Starting Tasks

1. Genre classification from structured acoustic features.
2. Popularity regression without lyrics, playlists, or artist-follower metadata.
3. Mood clustering with PCA, UMAP, or K-Means.
4. Similarity search and recommendation prototypes using feature distance.
5. Feature-importance demos for tree ensembles and boosting models.

## Column Guide

### Identity and release metadata

- `track_id`: unique synthetic identifier
- `track_name`: track title
- `artist_name`: primary artist name
- `album_name`: album title
- `release_year`: release year from 2000 to 2024
- `genre`: primary genre label
- `explicit`: boolean explicit-content flag

### Target / outcome field

- `popularity`: synthetic 0-100 popularity score for ranking and regression tasks

### Audio features

- `danceability`: how rhythmically suitable the track is for dancing
- `energy`: perceptual intensity and activity level
- `loudness`: overall loudness in dB
- `speechiness`: spoken-word intensity
- `acousticness`: confidence that the track is acoustic
- `instrumentalness`: confidence that the track has little or no vocal content
- `liveness`: likelihood of live audience presence
- `valence`: musical positivity
- `tempo`: estimated BPM
- `key`: encoded musical key from 0 to 11
- `mode`: 1 for major, 0 for minor
- `time_signature`: estimated beats per bar
- `duration_ms`: track duration in milliseconds

## Linked Assets

- Dataset page: <https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k>
- Explore notebook: <https://www.kaggle.com/code/lorenzoscaturchio/spotify-tracks-eda-popularity-prediction>

## Practical Notes

- Genre patterns are intentionally distinct enough to support classification baselines.
- Popularity is harder than genre because the table omits playlist, social, and artist-network effects.
- This is synthetic data intended for education, prototyping, and benchmarking rather than production music analytics.

## Provenance

- Generated from repository scripts in this project
- Built with seeded statistical rules and genre-aware feature distributions
- Designed for reproducible ML demos, EDA, and teaching workflows

## Citation

Scaturchio, Lorenzo (2026). *Spotify Tracks: Audio Features (50K Songs).* Kaggle Dataset. <https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k>
