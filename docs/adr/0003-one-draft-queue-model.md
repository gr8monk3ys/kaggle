# One Draft Queue model, and the scheduler's selector is the canonical one

The Draft Queue existed twice — once in the scheduler, once in the automation
scripts — joined only by a JSON file and three environment variables across a
process. The copies had drifted, most consequentially in *which Draft posts
next*. We collapsed them into one imported model and kept the scheduler's
`select_next_post` as the single selector.

The two selectors disagreed structurally, not cosmetically. `select_next_post`
orders by due-ness, schedule, priority and id, and treats an unscheduled `ready`
Draft as due. `next_pending` returned nothing at all whenever any future-scheduled
item existed and nothing was due — so a `ready` Draft could be starved
indefinitely by an unrelated scheduled one.

## Consequences

- The Flywheel's recorded target becomes truthful. It already called
  `select_next_post` to predict what would post, while the poster used
  `next_pending` — so it could mark one Draft done while a different one was
  published, and its dedupe gate would then starve the first permanently.
- The poster keeps running as a separate process. That seam is justified by a
  real constraint — Playwright is a heavyweight, differently-provisioned
  dependency that must not become reachable from a `kaggle_portfolio` import.
  What crosses the process gap is now a shared model, not a duplicated one.
- Adopting the scheduler's status normalisation is inert on current data: of 62
  queued Drafts, none has a blank or annotated status, and both copies already
  agreed on which statuses are Postable.
- This changes what the poster would publish, which is why it ships disabled.
  See ADR-0004.
