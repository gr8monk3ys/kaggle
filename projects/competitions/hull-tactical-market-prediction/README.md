# Hull Tactical - Market Prediction (STARTER)

Live Kaggle competition: <https://www.kaggle.com/competitions/hull-tactical-market-prediction>

**Status: STARTER, not a trained baseline.** The data (`train.csv`, `test.csv`)
is download-gated (HTTP 403 from the Kaggle API until you accept the rules in the
web UI), and scoring runs through an inference-server gateway rather than a
static `submission.csv`. This package ships a runnable offline trainer + honest
time-series CV + the live gateway wiring, with a synthetic smoke test that
proves the logic. No real CV/LB number is claimed because no real training has
been run. Nothing here is faked.

## The task

Predict a **daily allocation to the S&P 500**, bounded in **[0, 2]**
(0 = all cash, 1 = fully invested, up to 2 = 2x leverage). The signal to learn
is the **forward excess return** of the S&P 500 over the risk-free rate (the
train target is along the lines of `market_forward_excess_returns`). The
competition's framing is an explicit challenge to the Efficient Market
Hypothesis: can you tactically size exposure to beat buy-and-hold on a
risk-adjusted basis?

This is **not a flat offline tabular submission**. It is a streaming time series:
the gateway feeds you one trading day at a time and you must respond with that
day's allocation, so look-ahead is structurally impossible. That, plus the
Sharpe-style metric, is why this is a starter rather than a naive
`HistGradientBoosting -> submission.csv` baseline.

## Data schema (from `kaggle competitions files -c hull-tactical-market-prediction`)

| File | Contents |
|------|----------|
| `train.csv` (~8.5 MB) | Historical daily rows: a wide set of anonymized market/macro features plus the forward-return target (`market_forward_excess_returns`). |
| `test.csv` (~11 KB) | A small example test set used for local gateway debugging; the real scored test set is hidden and streamed at rerun time. |
| `kaggle_evaluation/` | The competition's gateway package (inference server + gateway + protobuf relay). Do **not** edit; it ships with the data on Kaggle. |

> Exact feature column names are only visible after download. The trainer is
> schema-robust: it resolves the target by name (`TARGET_CANDIDATES` in
> `baseline.py`), treats every other numeric column as a feature, and excludes
> id/outcome columns (`date_id`, `is_scored`, `weight`, ...).

## Metric

A **modified, volatility-penalized Sharpe ratio** of the strategy's returns
(higher is better). Public write-ups describe a penalty when the portfolio's
realized volatility exceeds the benchmark's by more than ~20%, and a penalty for
failing to beat the market. So **position sizing and risk control matter as much
as directional accuracy** — a model that is merely accurate but over-levered can
score worse than a tame one. Confirm the exact formula on the Evaluation tab.

`baseline.py` reports both regression RMSE and an annualized **strategy Sharpe
proxy** during CV so local numbers track the leaderboard objective.

## Files

| File | Role |
|------|------|
| `baseline.py` | Offline trainer: loader, expanding-window time-series CV (no shuffle), `HistGradientBoostingRegressor`, and `allocation_from_pred()` (the predicted-return → [0,2] allocation link). `--smoke-test` runs it all on synthetic market data. |
| `inference_server.py` | Live-submission skeleton: reuses `baseline.py`'s feature/target logic and allocation link, defines the gateway `predict(test_row) -> float` callback, and starts the `kaggle_evaluation` server. Runs unchanged inside the Kaggle notebook. |

## Proposed approach (implemented)

1. **Leak-free features**: every numeric column except the target/ids; missing
   and infinite values handled. (Next: explicit lags / rolling vol / regime
   flags — careful to use only information available up to day *t*.)
2. **Honest CV**: expanding-window, **no shuffling** — train on `[0:k]`, validate
   on the next block. Random KFold would leak the future into the past and is
   meaningless for trading. Reports RMSE *and* the strategy Sharpe proxy.
3. **Model**: `HistGradientBoostingRegressor` (sklearn, dependency-light);
   `xgboost==3.2.0` is available for a drop-in swap.
4. **Allocation link**: `allocation = clip(1 + mu / (2*sigma), 0, 2)` — monotone
   in predicted edge, centered at fully-invested, scaled by historical return
   spread. Replace with mean-variance / fractional-Kelly sizing once the exact
   metric is confirmed, since the metric rewards risk control.
5. **Live serving**: `inference_server.predict()` aligns the incoming day's row
   to the trained feature order and returns one bounded allocation.

### Next steps toward a competitive model
- Volatility targeting: size so realized vol ≈ benchmark vol (directly defends
  against the >20%-vol penalty).
- Time-aware features: lagged returns, rolling vol, drawdown state, momentum.
- Ensemble a directional return model with a separate volatility model and let
  the sizing rule combine them.
- Walk-forward hyperparameter selection on the expanding-window splits.

## Reproduce

```bash
# 1. Verify the offline pipeline with no download (synthetic market data):
python baseline.py --smoke-test
# -> expanding-window CV (RMSE + strategy Sharpe proxy) and a bounded allocation.

# 2. Real offline training: accept rules first at the competition URL, then:
kaggle competitions download -c hull-tactical-market-prediction -p ./data
cd data && unzip -q hull-tactical-market-prediction.zip && cd ..
python baseline.py --data-dir ./data

# 3. Live submission: copy baseline.py + inference_server.py into the Kaggle
#    submission notebook (where kaggle_evaluation + data are mounted) and run
#    `python inference_server.py` so the gateway scores your predict() callback.
```

The smoke test currently passes (synthetic CV RMSE ~0.0104, strategy Sharpe
proxy ~1.4, last-day allocation correctly bounded in [0,2]), confirming the
load → time-series CV → fit → allocation-link → gateway path is correct.
Real-data CV will be filled in here once the rules are accepted and the script
is run on the actual `train.csv`.

## Dependencies

`numpy`, `pandas`, `scikit-learn` (required); `xgboost` (optional);
`kaggle_evaluation` (provided by the competition, only for live scoring). Data
files and the gateway package are gitignored (`.gitignore`).
