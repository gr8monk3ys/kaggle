# Regenerating the Campaign queue merges; it never replaces

`campaign-pack` builds the Campaign queue from the current Datasets and used to
write the result wholesale, dropping `claim_count`, `claimed_at`, `completed_at`
and `updated_at`. Re-running it therefore reset finished actions to `planned`.
Two of the four channels are `x` and `linkedin`, so a reset action is an
already-published promotion becoming eligible to post again. Regeneration now
merges by action id: the fresh plan supplies the planning fields, the existing
queue keeps everything the run recorded.

The Campaign queue is its own model in `campaigns/campaign_queue.py`, alongside
but sharing nothing with `discussions/draft_queue.py`. The two look alike from a
distance — a JSON queue of scheduled work — and are different underneath: a dict
envelope rather than a bare list, disjoint status vocabularies
(`planned/in_progress/done/blocked` against `idea/ready/scheduled/posted/…`), and
a claim/lease concept Drafts have no equivalent of. A shared base would be
abstraction for a resemblance that stops at the file extension.

## Consequences

- **Claiming has one rule**: only a `planned` action can be claimed.
  `campaign_execute` already guarded on status; `campaign_dispatcher` claimed
  unconditionally, so it could re-claim work in flight or finished and increment
  `claim_count` — which exists to count retries and cannot also count how many
  times someone ran the command. A deliberate re-run is `requeue()`, which says
  what it is.
- **Finished work that the new plan no longer proposes is kept.** A Dataset
  dropping out of the campaign criteria must not erase the record that it was
  promoted.
- **Unfinished work dropped from the plan is discarded**, which is what makes
  this a regeneration rather than an append.
- A future reader will be tempted to simplify this back to a single `write_json`.
  That is the bug; this file is why it is not.
