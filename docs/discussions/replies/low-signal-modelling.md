Before trying more models, I would spend an hour establishing whether 52.3% is a result at all.

Take the size of your validation set and compute the standard error of a proportion at 0.5, which is 0.5/sqrt(n). On 1,000 samples that is about 1.6 percentage points, so 52.3% sits around one and a half standard errors from chance and a run-of-the-mill reshuffle can produce it. On 10,000 it is about 0.5pp and 52.3% starts to mean something. Same number, completely different conclusion, and you cannot tell which one you have without n. Bootstrap the validation set and look at the width of the interval rather than the point estimate.

This matters more than the next model because next-day direction on a systematic allocation is close to a coin flip by construction. If it were reliably predictable from 20 days of public returns and volumes, it would have been arbitraged. A realistic ceiling here is low single digits above chance, so an approach that gets you from 52.3% to 52.9% may be the win, and you will not be able to see it over the noise unless you have already fixed the measurement.

The other thing I would check is how you are splitting. Random K-fold on a time series lets the model see the future, and tree models are very good at exploiting that. Refit with a strictly time-ordered split, train on everything before a date and test after, and compare. If the score drops toward 50%, the earlier number was the split talking.

I hit exactly that recently on an unrelated citation-prediction problem. On a random split my best configuration beat the mean baseline by about 4%. On a time-ordered split the baseline won by 11.7%, and the honest conclusion was that the target was mostly unpredictable rather than that I needed better features. Worth knowing early, because it changes what you spend the next month on.

One trap in the same family: check whether any identifier column encodes time. Mine did, an ID that embedded year and month, so hashing it quietly put the date back into a split I had built specifically to remove it.

On the feature-engineering suggestions above, products and ratios are worth trying, but generate them and then test whether each one survives a time-ordered split. On a tabular problem last week I had a feature correlating with the target at +0.68 that collapsed to roughly zero once I conditioned on a grouping variable. It was a confounder, and it would have looked like signal in any marginal check.

Last thing: accuracy on direction weights a 0.01% move the same as a 3% move. If the underlying objective is returns, evaluate on returns. A model at 51% accuracy that is right on the large moves can beat one at 53% that is right on the noise.
