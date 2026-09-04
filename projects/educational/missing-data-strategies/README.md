# Missing-data strategies: a real benchmark

Seven ways of handling missing values, scored on three binary-classification
tasks with two model families (HistGradientBoosting and scaled logistic
regression), 5-fold stratified CV, seed 42.

Reproduce with:

```bash
python projects/educational/missing-data-strategies/benchmark.py
```

Full numbers live in `results.json`. This file records what they mean, and
what turned out to be wrong with the setup.

## Method

Every imputer is fit **inside** the training fold and only then applied to the
held-out fold. Fitting an imputer on the full dataset leaks test-fold
statistics (a median computed over rows you are about to score) and inflates
every strategy roughly equally, which is how imputation benchmarks end up
reporting differences that do not survive a real split.

`mental-health-tech` carries native missingness (1.6% of cells). The other two
have 20% of cells blanked completely at random, seeded.

## What the numbers say

**1. The sophisticated imputers are not worth their runtime.** On
breast-cancer, RF-imputation (0.9922) beats median (0.9887) by 0.0035 AUC
against a fold-to-fold standard deviation of ~0.011 — the gap is a third of
the noise, for 7x the wall-clock. MICE and KNN land in the same place. On
mental-health-tech, plain median is the *best* tree result (0.7220); every
fancier method scores slightly lower.

**2. Sentinel fill (-999) is the one choice that can destroy a model.** It is
harmless for trees, which learn to route the sentinel down its own branch
(0.9870 vs 0.9887 median). For a linear model it is catastrophic:

| dataset | median | constant -999 |
|---|---|---|
| breast-cancer, linear | 0.9921 | **0.6910** |
| credit-card-fraud, linear | 0.9999 | **0.5434** |

A logistic regression reads -999 as an extreme real value, so a fifth of the
rows become enormous outliers on every axis. This is the only result in the
benchmark where the choice of strategy is worth more than a rounding error —
and it is model-dependent, not universal.

**3. "Drop the rows" is often not available at all.** Both injected datasets
report `n/a` for row-dropping, which is not a bug: at 20% per-cell MCAR across
30 columns, the chance a row is complete is `0.8^30 = 0.12%` — 0.7 rows out of
569. Dropping does not degrade the model, it deletes the dataset. Row-dropping
is also ill-defined at prediction time: you may drop training rows, but every
test row still has to receive a score, so it needs an imputer anyway.

## A limitation worth recording

`credit-card-fraud` and `student-performance` — both synthetic datasets in this
repo — are **trivially separable**. Every strategy scores 0.9997-1.0000 AUC on
the fraud subsample, and a plain HistGradientBoosting baseline on
`student-performance` scores a clean 1.0000 with no missingness at all.

A benchmark cannot rank imputers on a task where everything already scores
1.000, so only the mental-health and breast-cancer results carry information
about ranking. The fraud numbers are kept because the sentinel-fill collapse
(0.9999 to 0.5434) still shows up clearly there.

This is worth fixing at the source: a published dataset whose target a baseline
solves perfectly is a weak dataset, and it is the kind of thing a reader checks.
