The metric list is thorough. What I would add is less about which numbers to collect and more about what makes them diagnostic, because you asked for something that locates the failure rather than scoring it.

**Run a no-retrieval baseline first.** Answer every query with the LLM alone, no context. If that scores close to your full pipeline, retrieval is not contributing and none of your Recall@K numbers matter yet. This is the cheapest experiment in the whole setup and the one most evaluations skip. I ran into the general version of this recently on a citation-prediction task: every model configuration I tried lost to predicting the mean, and the useful finding was that the target was mostly unpredictable, not that my features were bad. Knowing where the floor is changes what a score means.

**Then ablate one component at a time.** Swap the retriever for random passages. Truncate context to half. Remove reranking. Watch which one moves the answer score. On a tabular problem last week, leave-one-feature-out took the AUC from 0.722 to 0.550 when I removed a single input, and cost 0.002 for another. Same model, same folds, and it told me the thing was reading one question and ignoring the other twenty. The RAG equivalent tells you which stage you are actually paying for.

That also answers your separate-or-end-to-end question. Measure end-to-end, because that is the objective, then use ablations to attribute. Retrieval metrics on their own are a proxy that you have already shown can disagree with the outcome, in your own RAG A / RAG B example.

**On sample size.** Fewer queries than you think will be stable, and it is worth measuring rather than assuming. On a 119-row classification set I was working with, one row was worth 4.2 accuracy points, and the same input swung from 54.2% to 87.5% across five folds of one split. Before you compare two systems, bootstrap your existing eval set and look at the width of the interval. If the gap between RAG A and RAG B fits inside it, you do not have a result yet.

**On LLM-as-a-judge.** The failure I would watch for is not bias in general but bias that correlates with what you are testing. Check judge agreement against humans within strata, not overall. A judge with 85% agreement can be near-random on exactly the subset where your two systems differ, and the marginal number hides it. This is the same shape as a confounded feature: I had one that correlated with salary at +0.68 marginally and collapsed to roughly zero inside every experience level.

**Multiple valid answers** are the case where automated correctness quietly stops working. The practical route is to grade against a rubric of required claims rather than a reference string, and accept that you are then evaluating the rubric too.

One thing that looks good and travels badly: faithfulness measured only on answered queries. A system that abstains more looks more faithful. Track abstention rate alongside it or the metric rewards silence.
