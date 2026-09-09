# Posting stays off by default, and enabling it is a separate act

Unifying the Draft Queue selector (ADR-0003) changes which Drafts are eligible to
publish: under the old poster, twelve future-scheduled items caused it to decline
entirely; under the canonical selector, thirty unscheduled `ready` Drafts count as
due and the cadence would resume on the next run. We landed the unified selector
with publishing behind a flag that defaults to **off**.

This decision is trivially reversible — it is one flag — and it is recorded
precisely because of that. A future reader, or a future agent, will see a
disabled code path in otherwise finished work and be tempted to enable it as
cleanup. It is not cleanup.

Two reasons it stays off until someone deliberately turns it on:

1. **The Drafts are not cleared for publication.** Drafts in this queue have
   previously asserted measured results the repo could not back, which is why the
   `unverified` status exists at all. They need an integrity check against the
   Tracker before anything is published under our name.
2. **Publishing is the one irreversible external effect in this system.** Every
   other part of this automation writes files we can regenerate or opens PRs we
   can close. A published discussion post cannot be recalled, and a refactor is
   not the change that should start it.

Enabling it is a one-line change plus a decision. Make the decision first.

---

## Update, 2026-09-08: the decision was made, and it was not the flag that mattered

The owner authorised posting, on the condition that drafts get a humanizing pass
first. Both reasons above were then worked through, and the outcome is not what
the ADR anticipated.

**Reason 1 is now genuinely discharged, and it caught something.** Running the
integrity check found that `asserts_unbacked_results` had two gaps, and two
drafts had been sitting in the postable set reporting experiments that were never
run:

- `draft_005` — "I tested three chunking strategies on the same corpus", with a
  table of 71.3% / 76.8% / 82.1% retrieval accuracy.
- `draft_032` — "For each, I tested preprocessing combinations and measured
  accuracy delta", reporting +0.3%, -0.8%, -0.5% to -1.2%.

None of those numbers exists anywhere in this repo. The detector missed them
because `METRIC_NUMBER` matched only `0.xxx` and both drafts stated results in
percent, and because `RESULT_TABLE` required the metric to open the cell while
draft_005's header read "| Retrieval Accuracy (Top-5) |". Both patterns are
widened, both drafts are now `unverified`, and a test asserts no postable draft
trips the check. A separate test ties dataset announcements to the shipped CSVs,
after draft_038 was found naming two programming languages its dataset does not
contain.

**Reason 2 turns out to be understated.** Publishing is not one flag away. The
poster runs inside a pi-automation container that has never been built — no
image, no container, no LaunchAgent — so `DISCUSSION_POSTING_ENABLED` is read by
nothing. And `discussion_post.py` authenticates by filling the Kaggle login form
with `KAGGLE_EMAIL` / `KAGGLE_PASSWORD`; the API token cannot substitute, because
`kaggle forums` is read-only (`list` and `topics` only). Enabling posting
therefore means standing up a Playwright container holding an account password,
which is a deployment decision rather than a code one.

So the flag stays off, for a different reason than before: not because the drafts
are unsafe — that has now been checked, twice, and fixed — but because the thing
it gates does not exist yet, and the route to making it exist stores a password.

`./manage.sh next-post` already prints the next due draft with its forum URL and
the command to mark it posted. That path needs no deployment and no stored
credential, and it is the recommended way to post until someone decides the
container is worth standing up.
