# No automated engagement with Kaggle

We removed every path that acts on Kaggle as the account owner through the
website rather than the API: Playwright upvoting, following, commenting and
discussion posting, the Draft Queue and its scheduler, the Campaign queue, the
growth flywheel, the browser adapter, and the Raspberry Pi stack that ran them.
This supersedes ADR-0003 through ADR-0006.

Medals are awarded for votes from other people. Automated voting, following and
posting is the behaviour Kaggle penalises with medal removal or account action,
and the account is the asset this repository exists to grow. The machinery also
produced no measurable result: after months of it, discussion stood at 17 net
votes and notebooks at about one vote each.

Publishing stays automated because it goes through the official API
(ADR-0001). Discussion participation is done by hand, in threads where there is
something useful to say.
