# Kaggle Discussion Posts - Ready to Post

**Date:** March 10, 2026  
**Status:** Current and ready for manual posting  
**Focus:** discussion visibility that supports live notebooks, datasets, and competition activity

## Posting Order

1. `titanic` competition comment
2. `spaceship-titanic` competition comment
3. `nlp-getting-started` competition comment
4. `store-sales-time-series-forecasting` discussion post
5. `general` dataset spotlight for Spotify Tracks
6. `deep-past-initiative-machine-translation` post after competition rules are accepted

## Post 1

**Forum:** <https://www.kaggle.com/competitions/titanic/discussion>  
**Format:** comment on an existing feature-engineering or baseline thread

```text
I revisited Titanic with a stronger local CatBoost setup and pushed my public score from 0.77511 to 0.77751 today.

The biggest lift came from simple structured features rather than a more exotic model: title extraction, family size, fare-per-person, cabin deck, and ticket-prefix cleanup.

Notebook here if useful:
https://www.kaggle.com/code/lorenzoscaturchio/titanic-ml-guide-zero-to-top-5-accuracy
```

## Post 2

**Forum:** <https://www.kaggle.com/competitions/spaceship-titanic/discussion>  
**Format:** comment on a modeling, feature-engineering, or CatBoost thread

```text
I just pushed a stronger Spaceship Titanic submission and moved my public score from 0.80079 to 0.80874.

The features that mattered most were deck and side parsing from Cabin, grouped-passenger features, total spend, no-spend flags, and CryoSleep interaction terms.

Notebook:
https://www.kaggle.com/code/lorenzoscaturchio/spaceship-titanic-complete-ml-guide
```

## Post 3

**Forum:** <https://www.kaggle.com/competitions/nlp-getting-started/discussion>  
**Format:** comment on a baseline, TF-IDF, or transformer thread

```text
I finally pushed a full submission from my disaster tweets notebook today and landed at 0.79681 on the public board.

The current version combines strong sparse baselines with a transformer-oriented workflow, and the next thing I want to improve is a cleaner out-of-fold blend instead of a simple probability mix.

Notebook:
https://www.kaggle.com/code/lorenzoscaturchio/nlp-disaster-tweets-bert-guide
```

## Post 4

**Forum:** <https://www.kaggle.com/competitions/store-sales-time-series-forecasting/discussion>  
**Format:** short topic or reply tied to validation strategy

**Suggested title:** Validation setup that mirrored the leaderboard best for me

```text
The biggest thing that improved my Store Sales workflow was treating validation design as the first modeling decision rather than an afterthought.

Once I switched to a strictly forward-looking split and inspected errors by store, family, and holiday regime, it became much easier to tell whether a new lag block was a real improvement or just leaking convenience.

I refreshed my notebook around that idea here:
https://www.kaggle.com/code/lorenzoscaturchio/store-sales-forecasting-lightgbm
```

## Post 5

**Forum:** <https://www.kaggle.com/discussions/general>  
**Format:** short discussion post highlighting one useful dataset pattern

**Suggested title:** A small music dataset that is actually useful for ML demos

```text
I have been refreshing a synthetic Spotify-style dataset and one reason it has been useful is that it is large enough to train real baselines but still compact enough to explore end to end in one sitting.

Genre classification is much easier than popularity prediction, which makes it a nice teaching example for the difference between style signals and audience signals.

Dataset:
https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k

Notebook:
https://www.kaggle.com/code/lorenzoscaturchio/spotify-tracks-eda-popularity-prediction
```

## Post 6

**Forum:** <https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion>  
**Format:** topic after you manually accept the competition rules

**Suggested title:** Preprocessing seems higher leverage than model size so far

```text
I put together an Akkadian baseline notebook and one thing that stood out immediately is how much preprocessing quality matters before model choice.

Token normalization, transliteration consistency, and sequence length handling all look like higher-leverage decisions than jumping straight into a bigger model.

I'm using a ByT5-style baseline as the starting point:
https://www.kaggle.com/code/lorenzoscaturchio/akkadian-translation-eda-byt5-seq2seq-baseline
```

## Manual Checklist

- [ ] Post the three competition comments first
- [ ] Publish the Store Sales thread
- [ ] Publish the Spotify dataset spotlight
- [ ] Enter Deep Past Akkadian and accept rules before posting there
- [ ] Check replies after 12 to 24 hours and answer anything technical
