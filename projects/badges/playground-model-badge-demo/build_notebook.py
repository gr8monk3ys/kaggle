#!/usr/bin/env python3
"""Build the Playground Model Badge Demo tutorial notebook.

This builder generates a self-contained tutorial that walks through the
Kaggle Model Hub workflow inside a competition notebook: attaching a
pretrained artifact, pinning its version for reproducibility, loading it
with joblib, running inference, and visualising the predictions. The
underlying model is a tiny scikit-learn iris logistic-regression
classifier published as a Model Hub instance.
"""

from __future__ import annotations

import os as _os
import sys as _sys


def _find_repo_root(start_dir: str) -> str:
    current = _os.path.abspath(start_dir)
    while True:
        if _os.path.exists(_os.path.join(current, "manage.sh")) and _os.path.isdir(
            _os.path.join(current, "kaggle_portfolio")
        ):
            return current
        parent = _os.path.dirname(current)
        if parent == current:
            return _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        current = parent


_sys.path.insert(0, _find_repo_root(_os.path.dirname(_os.path.abspath(__file__))))

from kaggle_portfolio.shared.build_utils import code, md, write_notebook


cells: list[dict] = []

# ---------------------------------------------------------------------------
# Title (H1) -- first non-empty cell, states problem / data / approach.
# ---------------------------------------------------------------------------
cells.append(
    md(
        """# Kaggle Model Hub in Competition Notebooks: Attach, Pin, Load & Serve

**Problem.** Competition notebooks frequently need a *pretrained* model
that was trained elsewhere -- a backbone, a baseline, or a heavyweight
checkpoint that would be wasteful to retrain on every run. The Kaggle
**Model Hub** lets you attach such an artifact to a notebook the same way
you attach a dataset, so it is available offline under `/kaggle/input`.

**Data / model.** This tutorial attaches a deliberately tiny artifact: a
scikit-learn **logistic-regression** classifier trained on the classic
**iris** dataset (four flower measurements, three species). It is small
enough to reason about end-to-end while exercising the full Model Hub
workflow.

**Approach.** We discover the artifact across the `/kaggle/input` search
roots, read its **model card** and `label_names.json`, load the estimator
with **joblib**, run inference on demo samples, map integer predictions
back to human-readable labels, and then *visualise and interpret* the
results. Throughout, we emphasise the reproducibility benefit of
**pinning the model version**.
"""
    )
)

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
cells.append(
    md(
        """## 1. Objective & Introduction

The **goal** of this notebook is to give you a copy-pasteable recipe for
the Kaggle Model Hub workflow inside a competition kernel. By the end you
will be able to:

1. Attach a published model to a notebook and locate its files at runtime.
2. Load a serialized scikit-learn estimator with `joblib` and serve
   predictions.
3. Translate raw integer class ids into the label names shipped with the
   artifact.
4. Inspect predicted-class distributions and per-sample probabilities
   with real charts.
5. Understand *why* pinning the model **version** is the single most
   important reproducibility lever in this workflow.

**Why attach pretrained models at all?** Three reasons dominate:

- **Speed & cost.** You skip retraining on every commit, which keeps the
  9-hour notebook budget free for inference and post-processing.
- **Reuse.** One curated artifact can be shared across many competition
  notebooks and teammates instead of being copy-pasted.
- **Reproducibility.** A model version is an *immutable* snapshot. Pin it
  and every future run starts from byte-identical weights -- the key idea
  we return to in the reproducibility section.
"""
    )
)

cells.append(
    code(
        """# Standard library + scientific stack. The Model Hub artifact is a
# scikit-learn estimator serialized with joblib, so the runtime
# dependencies are intentionally light.
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

print("Python  :", sys.version.split()[0])
print("NumPy   :", np.__version__)
print("pandas  :", pd.__version__)
print("joblib  :", joblib.__version__)"""
    )
)

cells.append(
    md(
        """### Reproducibility setup

We fix a single `SEED` constant up front and reuse it anywhere we sample
or shuffle. There is no model *training* here (the artifact is already
trained), but we still seed NumPy so that any illustrative sampling below
is deterministic across runs.
"""
    )
)

cells.append(
    code(
        """# A single source of truth for randomness. Reused for every sampling
# call so results are byte-for-byte reproducible across sessions.
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
print(f"Random seed fixed to {SEED}")"""
    )
)

# ---------------------------------------------------------------------------
# Data / Model overview
# ---------------------------------------------------------------------------
cells.append(
    md(
        """## 2. Data & Model Overview

When you attach a model via the notebook's *Add Input -> Models* panel
(or the `model_sources` field in `kernel-metadata.json`), Kaggle mounts
its files read-only under `/kaggle/input/<owner>/<model>/<framework>/<instance>/<version>/`.

The attachment for this notebook is pinned to:

```
lorenzoscaturchio/iris-logistic-regression-badge-demo/ScikitLearn/scikit-baseline/2
```

The trailing `/2` is the **version number** -- the pin that guarantees
reproducibility. The artifact ships three files:

| File | Purpose |
| --- | --- |
| `iris_logreg.joblib` | The serialized scikit-learn estimator. |
| `label_names.json` | Maps integer class ids to species names. |
| `README.md` | The model card (summary, characteristics, usage). |
"""
    )
)

cells.append(
    md(
        """### Discovering the artifact across search roots

Hard-coding an absolute path is brittle: the exact mount point depends on
owner, framework, and version. Instead we search a small list of
candidate roots and glob for the artifact file name. This keeps the
notebook portable between Kaggle (`/kaggle/input`) and a local checkout
of this repository.
"""
    )
)

cells.append(
    code(
        """# Candidate roots, in priority order. On Kaggle the artifact lives under
# /kaggle/input; the local repo path lets the notebook also run offline.
search_roots = [
    Path("/kaggle/input"),
    Path("/workspaces/kaggle/projects/badges/kaggle-model-badge-demo/artifacts"),
]

model_path = None
for root in search_roots:
    if root.exists():
        matches = sorted(root.rglob("iris_logreg.joblib"))
        if matches:
            model_path = matches[0]
            break

if model_path is None:
    raise FileNotFoundError(
        "iris_logreg.joblib not found. Attach the Model Hub artifact "
        "or run from a checkout that includes the local artifacts/ folder."
    )

print("Resolved artifact directory:", model_path.parent)
print("Model file                :", model_path.name)"""
    )
)

cells.append(
    md(
        """### Reading the model card

The model card (`README.md`) is the human-readable contract for the
artifact: what it does, how it was trained, and what shape of input it
expects. Reading it programmatically is a good habit -- it documents the
provenance of the predictions your competition submission depends on.
"""
    )
)

cells.append(
    code(
        """# Surface the model card if it shipped alongside the estimator.
card_path = model_path.with_name("README.md")
if card_path.exists():
    print(card_path.read_text(encoding="utf-8").strip())
else:
    print("No README.md model card found next to the artifact.")"""
    )
)

cells.append(
    md(
        """### Loading the label map

`label_names.json` maps the estimator's integer class ids (`0`, `1`, `2`)
to the species names (`setosa`, `versicolor`, `virginica`). Shipping this
file *with* the model is what lets a downstream notebook produce
human-readable output without re-deriving the class ordering -- a common
source of silent label-swap bugs.
"""
    )
)

cells.append(
    code(
        """# The label map is the bridge from integer predictions to species names.
label_path = model_path.with_name("label_names.json")
label_names = json.loads(label_path.read_text(encoding="utf-8"))

label_table = pd.DataFrame(
    sorted(((int(k), v) for k, v in label_names.items()), key=lambda kv: kv[0]),
    columns=["class_id", "label"],
)
print(label_table.to_string(index=False))"""
    )
)

# ---------------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------------
cells.append(
    md(
        """## 3. Method: Loading & Serving the Model

With the files located, the actual *serving* code is short. The method
has three parts:

1. **Deserialize** the estimator with `joblib.load`.
2. **Introspect** the loaded object so we know it matches the model card
   (estimator type, number of features, class ordering).
3. **Map** integer predictions back to label names using the label map we
   just read.

A note on version pinning, since it is the heart of this tutorial: the
estimator below was fit with a specific scikit-learn release. Pinning the
Model Hub **version** freezes the exact serialized bytes, so the
`classes_` ordering and the learned coefficients can never shift under
you between runs.
"""
    )
)

cells.append(
    code(
        """# Deserialize the trained estimator. joblib is scikit-learn's recommended
# format because it stores large NumPy arrays efficiently.
model = joblib.load(model_path)

print("Estimator       :", type(model).__name__)
print("Expects features:", getattr(model, "n_features_in_", "unknown"))
print("Class id order  :", getattr(model, "classes_", "unknown"))
print("Has predict_proba:", hasattr(model, "predict_proba"))"""
    )
)

cells.append(
    md(
        """### A tiny helper to serve labelled predictions

We wrap the integer-to-label mapping in a small function so the rest of
the notebook reads cleanly. Guarding the lookup with `str(int(idx))`
matches the JSON key type and avoids `KeyError`s when ids arrive as NumPy
integers.
"""
    )
)

cells.append(
    code(
        """def to_labels(int_predictions):
    \"\"\"Map an iterable of integer class ids to human-readable labels.\"\"\"
    return [label_names[str(int(idx))] for idx in int_predictions]


# Smoke-test the helper on the known class ids.
print(to_labels(sorted(int(k) for k in label_names)))"""
    )
)

cells.append(
    md(
        """### Demo inference samples

The four columns are the canonical iris measurements: **sepal length**,
**sepal width**, **petal length**, **petal width** (all in centimetres).
The three rows are hand-picked to sit near the centre of each species'
feature region, so we expect one prediction per class.
"""
    )
)

cells.append(
    code(
        """FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X_demo = np.array(
    [
        [5.1, 3.5, 1.4, 0.2],  # expected: setosa
        [6.0, 2.9, 4.5, 1.5],  # expected: versicolor
        [6.9, 3.1, 5.4, 2.1],  # expected: virginica
    ]
)

demo_df = pd.DataFrame(X_demo, columns=FEATURE_NAMES)
demo_df"""
    )
)

cells.append(
    code(
        """# Serve predictions for the demo samples.
predictions = model.predict(X_demo)
predicted_labels = to_labels(predictions)

for row, label in zip(X_demo.tolist(), predicted_labels):
    print({"features": row, "prediction": label})"""
    )
)

# ---------------------------------------------------------------------------
# Evaluation + visualisation
# ---------------------------------------------------------------------------
cells.append(
    md(
        """## 4. Results & Evaluation

A bare list of predictions is hard to reason about. Below we attach the
predicted labels to the feature frame and then build three genuine
charts:

- a **bar chart** of predicted-class counts,
- a **scatter** of two features coloured by predicted class, and
- a per-sample **probability heatmap** (since this estimator exposes
  `predict_proba`).

Together these turn three opaque numbers into an interpretable picture of
*what the model is doing and how confident it is*.
"""
    )
)

cells.append(
    code(
        """# Assemble a tidy results frame for analysis and plotting.
results = demo_df.copy()
results["pred_id"] = predictions
results["pred_label"] = predicted_labels
results"""
    )
)

cells.append(
    md(
        """### Predicted-class distribution

The count of predictions per class. For a tiny demo set we expect each
species exactly once; on a real batch this chart immediately flags class
imbalance or a degenerate model that collapses onto a single label.
"""
    )
)

cells.append(
    code(
        """# Bar chart of predicted-class counts.
class_counts = (
    results["pred_label"]
    .value_counts()
    .reindex(label_table["label"], fill_value=0)
)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index,
            palette="viridis", legend=False, ax=ax)
ax.set_title("Predicted class distribution (demo batch)")
ax.set_xlabel("Predicted species")
ax.set_ylabel("Number of samples")
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 0.02, str(int(v)), ha="center", va="bottom")
plt.tight_layout()
plt.show()"""
    )
)

cells.append(
    md(
        """### Feature space coloured by prediction

Iris is famously separable in petal space. Plotting **petal length vs.
petal width** coloured by the predicted label shows the three demo points
landing in well-separated regions -- a quick visual sanity check that the
attached model behaves the way the model card claims.
"""
    )
)

cells.append(
    code(
        """# Scatter of two informative features, coloured by predicted class.
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=results,
    x="petal_length",
    y="petal_width",
    hue="pred_label",
    palette="viridis",
    s=180,
    edgecolor="black",
    ax=ax,
)
for _, r in results.iterrows():
    ax.annotate(r["pred_label"], (r["petal_length"], r["petal_width"]),
                textcoords="offset points", xytext=(8, 4), fontsize=9)
ax.set_title("Demo samples in petal feature space")
ax.set_xlabel("Petal length (cm)")
ax.set_ylabel("Petal width (cm)")
ax.legend(title="Predicted")
plt.tight_layout()
plt.show()"""
    )
)

cells.append(
    md(
        """### Prediction confidence (probability heatmap)

Logistic regression is a probabilistic classifier, so `predict_proba`
gives a full distribution over the three species for each sample. A
heatmap of these probabilities exposes *how confident* each prediction is
-- a near-1.0 cell means the model is sure, while two comparable cells in
a row would signal a borderline sample.
"""
    )
)

cells.append(
    code(
        """# Probability heatmap, guarded so the notebook still runs for estimators
# that do not expose predict_proba.
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_demo)
    proba_df = pd.DataFrame(
        proba,
        columns=[label_names[str(c)] for c in model.classes_],
        index=[f"sample {i}" for i in range(len(X_demo))],
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(proba_df, annot=True, fmt=".3f", cmap="viridis",
                vmin=0, vmax=1, cbar_kws={"label": "probability"}, ax=ax)
    ax.set_title("Per-sample class probabilities")
    ax.set_xlabel("Species")
    ax.set_ylabel("Demo sample")
    plt.tight_layout()
    plt.show()

    proba_df["max_prob"] = proba_df.max(axis=1)
    display(proba_df)
else:
    print("This estimator does not expose predict_proba; skipping heatmap.")"""
    )
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
cells.append(
    md(
        """## 5. Reproducibility: why pinning the version matters

Two ingredients make this notebook deterministic:

1. **The fixed `SEED`** above governs any sampling we do in-notebook.
2. **The pinned model version** (`.../scikit-baseline/2`) governs the
   *weights themselves*.

The second is easy to overlook. If you attach a model by *name* and a new
version is published, the next run silently picks up different
coefficients -- the classic **model-drift** failure. Pinning the integer
version freezes the artifact, so the loaded `classes_` ordering and the
learned weights are byte-identical forever.

The cell below records a lightweight fingerprint of the loaded model. Run
it twice (or on Kaggle vs. locally): the values should match exactly,
which is the practical proof that the pin is doing its job.
"""
    )
)

cells.append(
    code(
        """# A reproducibility fingerprint: deterministic given a pinned version.
fingerprint = {
    "estimator": type(model).__name__,
    "n_features_in": int(getattr(model, "n_features_in_", -1)),
    "classes": [int(c) for c in getattr(model, "classes_", [])],
    "coef_checksum": round(float(np.abs(model.coef_).sum()), 6)
    if hasattr(model, "coef_") else None,
    "seed": SEED,
}
print(json.dumps(fingerprint, indent=2))"""
    )
)

cells.append(
    code(
        """# Demonstrate seeded sampling is repeatable: draw the same indices twice.
draw_a = rng.choice(len(X_demo), size=2, replace=False)
rng_again = np.random.default_rng(SEED)
draw_b = rng_again.choice(len(X_demo), size=2, replace=False)
print("First draw :", draw_a.tolist())
print("Reseeded   :", draw_b.tolist())
print("Deterministic:", np.array_equal(draw_a, draw_b))"""
    )
)

# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------
cells.append(
    md(
        """## 6. Insights & Interpretation

A few concrete observations from the run above:

- **Finding.** Each demo sample was assigned to a different species and
  the probability heatmap shows the winning class dominating, *because*
  the hand-picked points sit deep inside each species' feature region.
  This is the expected, well-behaved outcome for separable iris data.

- **Interpretation.** Petal length and width carry most of the signal;
  *therefore* the scatter plot separates the classes cleanly even with
  only three points. Sepal features alone would be far more ambiguous.

- **Limitation.** Three samples is a *demo*, not an evaluation. We cannot
  estimate accuracy, calibration, or per-class recall from it -- the
  charts validate the *plumbing*, not the model's real-world quality.

- **Caveat / trade-off.** A pinned version protects against silent
  drift, but it also means you must *consciously* re-pin to benefit from
  an improved retrain. The trade-off is stability today versus freshness
  tomorrow.

- **Hypothesis.** If we fed borderline samples (measurements between
  versicolor and virginica), we would expect the heatmap to show two
  comparable probabilities in a row -- a useful follow-up experiment to
  probe the decision boundary.
"""
    )
)

# ---------------------------------------------------------------------------
# Conclusion & Next Steps (closing)
# ---------------------------------------------------------------------------
cells.append(
    md(
        """## 7. Conclusion & Next Steps

**Summary.** We attached a Kaggle Model Hub artifact to a competition
notebook, discovered it across the `/kaggle/input` search roots, read its
model card and label map, loaded the scikit-learn estimator with joblib,
served labelled predictions, and visualised both the predicted-class
distribution and per-sample confidence. The central **takeaway** is that
pinning the model *version* is what makes the whole workflow
reproducible.

**Recommended next steps / future work:**

1. **Scale up the artifact.** Swap the iris baseline for a real
   competition checkpoint (e.g. a gradient-boosted model or a deep
   backbone) using the identical attach-and-load recipe to *improve*
   submission quality.
2. **Batch inference.** Replace the three demo rows with the full
   competition test set and write `submission.csv`; chunk the input if it
   does not fit in memory.
3. **Probe the boundary.** Run the borderline-sample hypothesis above to
   characterise the model's decision boundary and calibration.
4. **Version hygiene.** Publish a new model version whenever you retrain,
   then *deliberately* bump the pin in `kernel-metadata.json` so every
   change is auditable.
5. **Enrich the model card.** Add evaluation metrics and intended-use
   notes so downstream notebooks can judge fitness before attaching.

**Final thoughts.** The Model Hub turns a trained model into a
first-class, versioned, shareable competition input. Master this small
loop and the same muscle memory carries over to far heavier artifacts.
"""
    )
)

write_notebook(cells, __file__, "playground_model_badge_demo.ipynb")
