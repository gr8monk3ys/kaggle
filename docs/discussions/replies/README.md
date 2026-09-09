# Drafted replies to open Kaggle threads

Replies written for specific live threads, held here so they survive a reboot
(they were first written to /tmp, which does not). Each is humanizer-checked and
every figure it cites is traceable to work in this repo.

Posting is manual: paste into the thread's comment box, which takes markdown.
Mark a reply posted by moving it under `## Posted` below with the date.

## Open

### low-signal-modelling.md
<https://www.kaggle.com/discussions/questions-and-answers/740036>
"Looking for ideas for modeling this low-signal problem" — 2 votes, 3 comments.

The asker is stuck at 52.3% accuracy on next-day return direction. Both existing
replies suggest more feature engineering. The reply instead points out that 52.3%
is 1.45 standard errors from chance at n=1000 and 4.60 at n=10,000, so whether it
is a result depends on a number they have not given.

Cites: the citation-prediction reversal in `datasets/ai-research-trends` (random
split beats the mean baseline by 4.2%, time-ordered split loses to it by 11.7%),
and the confounder in `datasets/job-postings` that correlates at +0.68 marginally
and collapses to roughly zero within experience level.

### rag-evaluation.md
<https://www.kaggle.com/discussions/questions-and-answers/739943>
"What is the right way to evaluate a RAG system beyond retrieval accuracy?" — no replies.

The asker already listed every standard metric, so the reply skips the list and
gives two diagnostics they are missing: a no-retrieval baseline, and
leave-one-component-out ablation.

Deliberately cites no RAG benchmark numbers. `draft_005`'s chunking table was
fabricated and is marked `unverified`; the figures quoted here come from the
tabular work and are labelled as coming from a different problem.

## Posted

_(none yet)_
