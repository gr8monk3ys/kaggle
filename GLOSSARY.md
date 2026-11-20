# Kaggle Portfolio

A monorepo of Kaggle artifacts plus the package that validates, publishes and
tracks them. The goal it serves is Grandmaster across all four Kaggle
categories, so most concepts here are either *an artifact we publish* or *a
measure of how that artifact is doing*.

## Language

### Artifacts

**Notebook**:
A published unit of analysis or teaching, living as one `.ipynb` in a
`projects/*` folder. This is the portfolio-facing name for the artifact.
_Avoid_: kernel (except when naming Kaggle's own API surface — see **Kernel**)

**Kernel**:
Kaggle's own name for a Notebook, used only where Kaggle's vocabulary is being
spoken: `kernel-metadata.json`, and the `kernels` CLI operations.
_Avoid_: using this for the artifact itself in our own prose

**Dataset**:
A published collection of data files plus its metadata, living in a `datasets/*`
folder.
_Avoid_: data, corpus

**Competition**:
A Kaggle contest we may enter, track, or submit to. Only Featured and Research
Competitions award Medals.
_Avoid_: contest, comp

**Ref**:
Kaggle's fully-qualified identifier for an artifact, `owner/name`. This is what
Kaggle's API returns and what we match on.
_Avoid_: id, full name

**Slug**:
The trailing segment of a **Ref** — the name alone, without the owner. Used for
folder names and for Competition identifiers.
_Avoid_: name, key

### Measures

Two different numbers are colloquially called "score". They are not
interchangeable and must never be compared to each other.

**Usability**:
Kaggle's rating of a Dataset, 0.0-1.0. Kaggle's to define; we can only influence
it.
_Avoid_: usability score, quality

**Leaderboard Score**:
A Competition's own metric for a submission. Its scale and direction differ per
Competition.
_Avoid_: score, result

**Medal**:
A Kaggle award on a single artifact. The count of these per category is what
Grandmaster is measured in. Earned only from other people's votes or from a
Competition placing, never from anything automated here (ADR-0007).
_Avoid_: award, badge

**Tracker**:
The hand-maintained record of where the portfolio stands against the Grandmaster
goal. It is the baseline that live Kaggle counts are synced into, and the source
of truth when the two disagree.
_Avoid_: report, dashboard

**Snapshot**:
A point-in-time copy of the Tracker's numbers, written by `sync` into
`medal_ops/history/`. The digest compares the latest two.
_Avoid_: scorecard, report

### Competitions

**Benchmark**:
A local, reproducible model run for one Competition, producing a comparable
result without submitting anything.
_Avoid_: experiment, run, model

**Competition Lab**:
The set of Benchmarks and the harness that fetches their data and optionally
submits their output. Shortened to `lab` inside its own package
(`competition_lab/`, `lab_root`, `LabResult`), where the context is unambiguous;
spell it out everywhere else.
_Avoid_: playground

**Bronze Cutoff**:
How many of the top places on a medal-awarding Competition earn at least bronze:
40% of the field under 250 teams, 100 places up to 1,000, 10% beyond.
_Avoid_: medal zone, threshold
