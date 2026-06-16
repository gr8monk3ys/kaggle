# ROGII - Wellbore Geology Prediction (STARTER)

Live Kaggle competition: <https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction>

**Status: STARTER, not a trained baseline.** The competition data is
download-gated (HTTP 403 from the Kaggle API until you accept the rules in the
web UI), so this package ships a fully runnable, schema-adaptive pipeline plus a
synthetic smoke test that proves the logic — but no real CV/LB score yet,
because no real training has been run. Nothing here is faked.

## The task

Predict the **geology along a horizontal wellbore** to automate *geosteering*
(steering the drill bit to stay inside the target reservoir). Concretely, the
target is **TVT (True Vertical Thickness)** — the vertical offset of the drilled
lateral relative to the geological marker — predicted at every station along the
wellbore trajectory.

This is **not a single flat tabular table**. Each well is a pair of time/depth
series that must be joined, and rows within a well are strongly autocorrelated,
so the modeling and the cross-validation both have to respect well boundaries.
That is why this is set up as a starter rather than a naive `train.csv` baseline.

## Data schema (from `kaggle competitions files -c rogii-wellbore-geology-prediction`)

Per well (each identified by an 8-hex id, e.g. `000d7d20`):

| File | Split(s) | Contents |
|------|----------|----------|
| `<id>__horizontal_well.csv` | train, test | The drilled **lateral** log: petrophysical measurements (gamma-ray, resistivity, density, etc.) and survey channels indexed by **measured depth (MD)**. This is what you predict TVT along. |
| `<id>__typewell.csv` | train, test | The **offset / pilot (vertical) reference** log: the same kinds of measurements indexed by **true vertical depth (TVD)**. The geosteering signal comes from comparing the lateral's logs to the typewell's expected response at the same TVD. |
| `<id>.png` | train only | A rendered geosteering cross-section image for the well (visual reference / optional CNN feature source). |
| `sample_submission.csv` | — | The required submission shape (one TVT prediction per scored lateral station). |
| `AI_wellbore_geology_prediction_task_en.pptx` | — | Official task description deck. |

> Exact log column names are only visible after download. The pipeline is
> **schema-adaptive**: it discovers numeric log columns at runtime and matches
> depth/target columns against candidate-name lists in `baseline.py`
> (`MD_CANDIDATES`, `TVD_CANDIDATES`, `TARGET_CANDIDATES`, `LOG_CANDIDATES`).
> Refine those constants once you have seen the real headers.

## Metric

An **RMSE-style error on TVT** (lower is better); public baselines on the
leaderboard report single-digit scores (e.g. ~9–13), consistent with TVT in
feet/metres. `baseline.py` uses RMSE for its internal CV so local numbers track
the leaderboard direction. Confirm the exact metric on the competition's
Evaluation tab and update this section.

## Proposed approach (implemented in `baseline.py`)

1. **Join** each lateral with its typewell via `merge_asof` on TVD (nearest), so
   every lateral station carries the typewell's expected logs at its vertical
   depth — the core geosteering comparison.
2. **Feature engineering** within each well: normalized MD position, MD step,
   per-log gradients, rolling mean / rolling std (windows of 5). These capture
   the *shape* of the log response, which is what marks formation tops.
3. **Model**: `HistGradientBoostingRegressor` (sklearn, dependency-light). An
   `xgboost` swap is trivial and `xgboost==3.2.0` is available in this env.
4. **Honest CV**: `GroupKFold` with the **well id as the group**, so no rows from
   a training well appear in its own validation fold. (Plain KFold would be
   optimistic because of within-well autocorrelation.)
5. **Refit** on all wells and write the submission in the sample column order.

### Next steps to turn this into a competitive model
- Sequence models (1D-CNN / GRU / Temporal Conv) over the lateral, since TVT
  evolves smoothly along MD — tree models ignore that ordering.
- Use the per-well `.png` geosteering images as an auxiliary CNN branch.
- Dynamic-time-warping alignment of lateral vs. typewell log signatures.
- Per-well bias correction / post-hoc smoothing of predicted TVT along MD.

## Reproduce

```bash
# 1. Verify the pipeline with no download (synthetic data matching the schema):
python baseline.py --smoke-test
# -> prints GroupKFold-by-well CV and writes a valid synthetic submission.

# 2. Real run: accept rules first at the competition URL above, then:
kaggle competitions download -c rogii-wellbore-geology-prediction -p ./data
cd data && unzip -q rogii-wellbore-geology-prediction.zip && cd ..
python baseline.py --data-dir ./data --out /tmp/rogii_submission.csv
```

The smoke test currently passes (synthetic CV RMSE ~0.32 against a synthetic
noise floor of 0.3, 23 engineered features), confirming the load → join →
feature → GroupKFold CV → submission path is correct. Real-data CV will be
filled in here once the rules are accepted and the script is run on the actual
wells.

## Dependencies

`numpy`, `pandas`, `scikit-learn` (required); `xgboost` (optional). All present
in this repo's environment. Data files are gitignored (`.gitignore`).
