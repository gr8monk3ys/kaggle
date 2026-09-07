# Kaggle Portfolio

A monorepo of Kaggle artifacts plus the automation that publishes, scores and
promotes them. The goal it exists to serve is Grandmaster across all four Kaggle
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
A Kaggle contest we may enter, track, or submit to.
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

Three different numbers are all colloquially called "score". They are not
interchangeable and must never be compared to each other.

**Quality Score**:
Our own rubric's rating of a Notebook, 0-100. Ours to define and to change.
_Avoid_: score, rating

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
Grandmaster is measured in.
_Avoid_: award, badge

**Tracker**:
The hand-maintained record of where the portfolio stands against the Grandmaster
goal. It is the baseline that live Kaggle counts are synced into, and the source
of truth when the two disagree.
_Avoid_: report, dashboard

**Scorecard**:
A generated snapshot of the portfolio's current standing, derived from the
**Tracker** plus live counts. Regenerated, never edited.
_Avoid_: report, summary

### Discussion pipeline

**Draft**:
A written discussion post that has not been published to Kaggle. Drafts are
authored by hand and reviewed before they become eligible to post.
_Avoid_: post, article

**Draft Queue**:
The ordered set of Drafts together with their status and schedule. It is the
single answer to "what posts next".
_Avoid_: backlog, schedule

**Draft Status**:
A Draft's position in its lifecycle — `idea`, `unverified`, `ready`,
`scheduled`, `pending`, `posted`, `expired`, `skipped`, `won-medal`.
_Avoid_: state, stage

**Postable**:
The subset of Draft Statuses eligible to be published: `ready`, `scheduled`,
`pending`. Every other status is terminal for scheduling purposes.
_Avoid_: active, live

### Automation

**Benchmark**:
A local, reproducible model run for one Competition, producing a comparable
result without submitting anything.
_Avoid_: experiment, run, model

**Competition Lab**:
The set of Benchmarks and the harness that fetches their data and optionally
submits their output.
_Avoid_: lab, playground

**Flywheel**:
The loop that picks one automation action per tick and records that it happened,
so the same action is never taken twice.
_Avoid_: scheduler, runner

**Campaign**:
A multi-channel promotion effort for a published artifact, queued and dispatched
rather than posted directly.
_Avoid_: promotion, blast
