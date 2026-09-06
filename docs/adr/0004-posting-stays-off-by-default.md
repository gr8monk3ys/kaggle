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
