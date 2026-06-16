# Playground Series S6E6 — Stellar Object Classification

**Competition:** [playground-series-s6e6](https://www.kaggle.com/competitions/playground-series-s6e6)
**Task:** multi-class classification of astronomical objects — `GALAXY` / `QSO` / `STAR` — from photometric and spectral features.
**Submission format:** `id,class`

## Data

| Split | Rows | Notes |
|-------|------|-------|
| train | 577,347 | labeled, no missing values |
| test  | 247,435 | |

- **Numeric features:** `alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift` (`redshift` is the strongest single signal — stars sit near 0).
- **Categorical features:** `spectral_type` (M, O/B, G/K, A/F), `galaxy_population` (Red_Sequence, Blue_Cloud).
- **Class balance:** GALAXY ≈ 65% · QSO ≈ 20% · STAR ≈ 14%.

## Baseline (`baseline.py`)

A `HistGradientBoostingClassifier` with native categorical support and honest
stratified 5-fold cross-validation. Chosen because it is a strong, fast,
dependency-light tabular learner (no xgboost/lightgbm needed) that handles the
mixed numeric/categorical features and class imbalance well.

- Categoricals are ordinal-encoded and flagged via `categorical_features`.
- `SEED = 42` throughout (folds + model) for reproducibility.
- Reports CV accuracy **and** macro-F1, then refits on all training data for the
  submission.

### Cross-validation results

| Metric | 5-fold CV |
|--------|-----------|
| Accuracy | **0.9673** ± 0.0006 |
| Macro-F1 | **0.9559** ± 0.0009 |

Stable across all folds (per-fold accuracy 0.9664–0.9681) — no overfitting
signal. The refit model's predicted test class mix (GALAXY 65.5% / QSO 20.2% /
STAR 14.3%) matches the training priors, a good calibration sanity check.

## Reproduce

```bash
# data: kaggle competitions download -c playground-series-s6e6
python baseline.py --data-dir /path/to/data --out submission.csv --folds 5
kaggle competitions submit -c playground-series-s6e6 -f submission.csv -m "HistGBM baseline"
```

## Next steps

- Feature engineering on color indices (`u-g`, `g-r`, `r-i`, `i-z`) — standard SDSS separators.
- Add xgboost/lightgbm and blend with HistGBM.
- Tune `redshift` handling (log/bucketing) and per-class thresholds if the metric rewards macro-F1.
