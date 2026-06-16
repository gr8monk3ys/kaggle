# Kaggle Discussion Engagement Strategy

**Goal:** Earn 50 bronze discussion medals to reach **Discussion Expert** tier
**Profile:** [kaggle.com/lorenzoscaturchio](https://www.kaggle.com/lorenzoscaturchio)
**Current Assets:** ~59 notebooks, 11 datasets (live counts; see `docs/reports/grandmaster-tracker.md`)
**Target Timeline:** 8-12 weeks (aggressive) / 16 weeks (steady)

---

## Table of Contents

1. [Discussion Medal Mechanics](#1-discussion-medal-mechanics)
2. [High-Value Discussion Post Templates](#2-high-value-discussion-post-templates)
3. [Comment Templates](#3-comment-templates)
4. [Weekly Engagement Routine](#4-weekly-engagement-routine)
5. [Content Calendar](#5-content-calendar)
6. [Cross-Promotion Strategy](#6-cross-promotion-strategy)
7. [Progress Tracking Checklist](#7-progress-tracking-checklist)

---

## 1. Discussion Medal Mechanics

### How Discussion Medals Work

Discussion medals are awarded on **individual discussion posts and comments** based on **net votes** (upvotes minus downvotes).

| Medal | Net Votes Required |
|--------|-------------------|
| **Bronze** | >= 1 net vote |
| **Silver** | >= 5 net votes |
| **Gold** | >= 10 net votes |

### Critical Rules

1. **Only non-novice votes count.** Upvotes from users at Novice tier are excluded from the medal calculation entirely. Only votes from Contributor-level users and above are counted.

2. **Self-upvotes do not count.** Your own upvote on your post or comment is not included.

3. **Old post votes may be excluded.** Kaggle excludes votes cast on very old posts from the medal calculation, preventing retroactive gaming.

4. **Downvotes subtract.** Net votes = upvotes - downvotes. A post with 3 upvotes and 2 downvotes has 1 net vote (bronze).

5. **Both posts AND comments earn medals.** You do not need to create a top-level discussion thread. A single helpful comment that receives 1 non-novice upvote earns a bronze medal.

6. **Silver and gold medals count toward bronze requirements.** A gold medal also counts as fulfilling silver and bronze requirements for tier progression. So 50 bronze medals means 50 total discussion medals of any tier.

7. **No asking for upvotes.** Kaggle community norms strongly discourage vote solicitation. Posts that say "please upvote" are frequently downvoted. Let quality speak for itself.

8. **Medals are permanent.** Once a medal is earned, it remains even if votes are later removed (as long as the post/comment is not deleted).

### What This Means Strategically

- The barrier per medal is very low: **just 1 upvote from a non-novice user**
- The challenge is doing it **50 times consistently**
- Both posts AND comments count, so you have two channels to earn medals
- Quality matters more than quantity -- a helpful 2-sentence comment can earn a medal just as easily as a long post
- Active competition forums have the highest concentration of non-novice users who can give qualifying upvotes

### Tier Progression Requirements (Discussion Category)

| Tier | Requirement |
|------|------------|
| **Novice** | Default starting tier |
| **Contributor** | Complete profile requirements |
| **Expert** | 50 bronze discussion medals |
| **Master** | 200 discussion medals, at least 50 silver |
| **Grandmaster** | 500 discussion medals, at least 50 gold |

---

## 2. High-Value Discussion Post Templates

Each template below is designed to maximize the probability of receiving at least 1 non-novice upvote. Templates are ordered by typical effort-to-upvote ratio (easiest wins first).

### Template 1: Competition EDA Findings Post

**Where to post:** Competition Discussion tab
**When to post:** Within the first 1-2 weeks of a competition launch
**Why it works:** Early EDA insights are the most valued content on Kaggle. Competitors need this information and will upvote useful findings.

```
Title: [Competition Name] - Key EDA Findings & Data Insights

Hi everyone,

I've been exploring the [competition name] dataset and wanted to share some key findings that might help the community:

## Dataset Overview
- Training set: [X] samples, [Y] features
- Test set: [X] samples
- Target distribution: [describe balance/imbalance]

## Key Findings

### 1. [Finding Title]
[Description with specific numbers/percentages]
[Include a key visualization or table if possible]

### 2. [Finding Title]
[Description with specific numbers/percentages]

### 3. [Finding Title]
[Description with specific numbers/percentages]

## Potential Data Issues
- [Issue 1: e.g., missing values in column X (Y%)]
- [Issue 2: e.g., potential data leakage in feature Z]
- [Issue 3: e.g., class imbalance ratio of X:Y]

## Implications for Modeling
Based on these findings, I think [approach] might work well because [reasoning].

Full EDA notebook here: [link to your notebook]

Would love to hear if others have found similar patterns or additional insights!
```

**Real example topic for Med-Gemma Impact Challenge:**
"Med-Gemma Challenge: EDA of Medical Image Distribution & Class Imbalance Analysis"

---

### Template 2: Bug Report / Data Issue Identification

**Where to post:** Competition Discussion tab or Dataset Discussion tab
**When to post:** As soon as you discover the issue
**Why it works:** This is one of the highest-upvote post types because it saves everyone time. Competition hosts also appreciate it.

```
Title: [Data Issue] [Specific description of the problem]

## Issue Description
I found what appears to be [a data issue / inconsistency / error] in the [training/test/supplementary] data.

## How to Reproduce
1. Load [specific file]
2. Filter/check [specific condition]
3. Observe [the unexpected result]

## Code to Verify
```python
import pandas as pd
df = pd.read_csv('[filename]')
# Code that demonstrates the issue
print(df[df['column'].isna()].shape)  # Shows X rows with missing values
```

## Expected vs. Actual
- **Expected:** [What you'd expect]
- **Actual:** [What you found]

## Impact on Modeling
This could affect models because [explanation].

## Suggested Workaround
Until this is addressed, you can [workaround].

Has anyone else noticed this? @[competition host username]
```

**Real example for Vesuvius Challenge:**
"Vesuvius Surface Detection: Inconsistent Label Masks in Training Tiles 42-47"

---

### Template 3: Technique Tutorial Post

**Where to post:** Competition Discussion or Getting Started or Learn forums
**When to post:** When you successfully apply a technique relevant to an active competition
**Why it works:** Educational content is evergreen and attracts upvotes from both competitors and learners.

```
Title: [Technique Name] Applied to [Competition/Task] - A Practical Guide

## Overview
[Technique name] is [brief 1-2 sentence description]. I applied it to [competition/task] and saw [specific improvement/result].

## Why This Technique?
The [competition/task] has [characteristic] which makes [technique] particularly effective because [reason].

## Implementation

### Step 1: [Setup/Prerequisites]
```python
# Required imports and setup
```

### Step 2: [Core Implementation]
```python
# The key technique code with comments
```

### Step 3: [Integration with Competition Pipeline]
```python
# How to plug this into your competition workflow
```

## Results
- Baseline score: [X]
- After applying [technique]: [Y]
- Improvement: [Z]

## Tips & Gotchas
1. [Practical tip that isn't obvious from documentation]
2. [Common mistake to avoid]
3. [Performance consideration]

## References
- [Paper/blog post that introduced the technique]
- [Relevant Kaggle notebook]

Full notebook: [link]
```

**Real example for AIMO3:**
"Applying Chain-of-Thought Prompting with Self-Verification for Math Olympiad Problems"

---

### Template 4: "Lessons Learned" Post-Competition Writeup

**Where to post:** Competition Discussion tab (after competition ends)
**When to post:** Within 1-3 days after a competition deadline
**Why it works:** The community values transparency. Post-competition writeups are some of the most upvoted content on Kaggle because they share real, tested insights.

```
Title: [Competition Name] - Lessons Learned [Your Final Position: e.g., Top X%]

## My Approach
### What Worked
1. **[Technique/approach]:** [Brief description and why it helped]
2. **[Technique/approach]:** [Brief description and improvement gained]
3. **[Technique/approach]:** [Brief description]

### What Didn't Work
1. **[Failed approach]:** [Why you tried it and why it failed]
2. **[Failed approach]:** [What you learned from the failure]

### What I Would Do Differently
1. [Specific change you'd make]
2. [Approach you wish you'd tried earlier]

## Key Takeaways
- [Insight about the data]
- [Insight about the modeling approach]
- [Insight about the competition format]

## Final Pipeline
[Brief description of your final submission pipeline]

## Score Progression
| Approach | Public LB | Private LB |
|----------|-----------|------------|
| Baseline | X.XXX | - |
| + Feature Engineering | X.XXX | - |
| + Ensemble | X.XXX | X.XXX |

## Resources That Helped
- [Link to notebook/discussion that was useful]
- [Paper or blog post]

Congrats to the winners! Looking forward to reading the top solutions.
```

**Real example after CAFA 6 Protein Function Prediction (deadline Feb 2):**
"CAFA 6 Retrospective: What Worked for Protein Function Prediction & What I'd Change"

---

### Template 5: Resource Compilation Post

**Where to post:** Competition Discussion or Getting Started forums
**When to post:** Early in a competition or when a new domain-specific competition launches
**Why it works:** Resource compilations save people hours of searching. They become bookmarked reference posts with sustained upvote accumulation.

```
Title: [Competition Name] - Comprehensive Resource List (Papers, Tutorials, Baselines)

I've compiled resources that may be helpful for the [competition name]. I'll keep this updated as I find more.

## Official Resources
- [Competition overview page](link)
- [Host's explanation of evaluation metric](link)
- [Sample submission notebook](link)

## Papers
1. **[Paper Title]** ([year]) - [1-sentence summary of relevance]
   - Link: [arxiv/paper link]
2. **[Paper Title]** ([year]) - [1-sentence summary]
   - Link: [link]
3. **[Paper Title]** ([year]) - [1-sentence summary]
   - Link: [link]

## Tutorials & Blog Posts
- [Title](link) - [1-sentence description]
- [Title](link) - [1-sentence description]

## Kaggle Notebooks (from this & similar competitions)
- [Notebook Title](link) by @[author] - [brief description]
- [Notebook Title](link) by @[author] - [brief description]

## Datasets
- [Dataset name](link) - [how it could supplement competition data]

## Tools & Libraries
- [Library name](link) - [relevant feature]
- [Library name](link) - [relevant feature]

## Similar Past Competitions
- [Competition name](link) - [what to learn from it]

---

If you have additional resources, please share them in the comments and I'll add them to the list!
```

**Real example for Deep Past Akkadian Translation:**
"Akkadian Translation Challenge: NLP Resources, Cuneiform Datasets, and Ancient Language Models"

---

### Template 6: Competition Strategy Discussion

**Where to post:** Competition Discussion tab
**When to post:** 2-4 weeks into a competition
**Why it works:** Strategy discussions are where experienced Kagglers engage most deeply. They spark multi-comment threads.

```
Title: Strategy Discussion: [Specific Strategic Question]

I've been thinking about [specific strategic aspect] of this competition and wanted to discuss approaches.

## The Challenge
[Describe the specific challenge or strategic decision point]

## Approaches I've Considered

### Approach A: [Name]
- **Pros:** [list]
- **Cons:** [list]
- **My preliminary results:** [if applicable]

### Approach B: [Name]
- **Pros:** [list]
- **Cons:** [list]
- **My preliminary results:** [if applicable]

## Open Questions
1. [Specific question about validation strategy]
2. [Specific question about data augmentation]
3. [Specific question about ensemble methods]

What approaches are others taking? Has anyone had success with [specific technique]?
```

**Real example for Stanford RNA 3D Folding 2:**
"RNA 3D Folding: Single Large Model vs. Ensemble of Specialized Fold Predictors?"

---

### Template 7: Public Notebook Walkthrough

**Where to post:** Competition Discussion tab (linking to your notebook)
**When to post:** When you publish a notebook that achieves a meaningful result
**Why it works:** Discussion posts that accompany notebooks drive traffic to both, and the cross-pollination increases upvotes on both the discussion and the notebook.

```
Title: [Notebook Name] - Walkthrough & Key Design Decisions

I just published a notebook for this competition and wanted to walk through the key decisions and invite feedback.

**Notebook link:** [link]

## What This Notebook Does
[2-3 sentence summary of the notebook's purpose and result]

## Key Design Decisions

### 1. [Decision: e.g., Feature Engineering Strategy]
I chose [approach] because [reasoning]. Alternatives I considered: [list].

### 2. [Decision: e.g., Model Architecture]
I went with [model] because [reasoning]. The key hyperparameters are [list].

### 3. [Decision: e.g., Validation Strategy]
I used [validation method] to handle [specific data characteristic].

## Results
- Public LB score: [X]
- Local CV score: [Y]
- Training time: [Z]

## Known Limitations
- [Limitation 1]
- [Limitation 2]

## Questions for the Community
1. [Specific feedback request]
2. [Area where you're unsure]

The notebook is fully documented and should be easy to fork. Let me know what you think!
```

---

### Template 8: Dataset Announcement Post

**Where to post:** Datasets Discussion or relevant Competition Discussion
**When to post:** When you publish a new dataset, especially one relevant to an active competition
**Why it works:** Datasets that solve a need get upvotes. The announcement post catches people who don't browse the Datasets tab.

```
Title: [New Dataset] [Dataset Name] - [Brief value proposition]

I've published a new dataset that may be useful for [competition/task/domain]:

**Dataset link:** [link]

## What's Included
- [File 1]: [description, size, format]
- [File 2]: [description, size, format]
- Total size: [X MB/GB]
- Number of samples: [X]

## Why I Created This
[Explain the gap this fills or the problem it solves]

## How to Use It
```python
import pandas as pd
df = pd.read_csv('/kaggle/input/[dataset-slug]/[filename].csv')
print(df.shape)
print(df.head())
```

## Data Quality
- Missing values: [description]
- Preprocessing applied: [description]
- Known limitations: [description]

## Potential Applications
1. [Application 1]
2. [Application 2]
3. [Application 3, e.g., supplementary data for Competition X]

I also published a starter notebook with basic EDA: [link]

Feedback and suggestions welcome!
```

---

### Template 9: Smart Question Post (Discussion Starter)

**Where to post:** Competition Discussion or Getting Started forums
**When to post:** When you encounter a genuine challenge that others likely face too
**Why it works:** Well-framed questions that articulate a common pain point attract answers AND upvotes on the question itself. The question becomes a reference for others with the same issue.

```
Title: [Specific, searchable question title]

## Context
I'm working on [competition/task] and I've hit [specific challenge]. My current approach is [brief description].

## What I've Tried
1. **[Approach 1]:** [Result and why it didn't fully solve the problem]
2. **[Approach 2]:** [Result and why it didn't fully solve the problem]

## Specific Question
[Clear, focused question. Not "how do I do this?" but rather "Given X constraint, is Y or Z approach more appropriate for this type of data?"]

## Relevant Code (if applicable)
```python
# Minimal code showing the issue or current approach
```

## What I Think the Answer Might Be
[Your hypothesis -- this shows you've thought about it and invites correction/confirmation]

Thanks in advance for any insights!
```

**Real example for AIMO3:**
"Math Olympiad: How Are Others Handling Multi-Step Proofs Where LLM Reasoning Breaks Down?"

---

### Template 10: Competition Solution Sharing

**Where to post:** Competition Discussion tab (after competition ends or if sharing partial approach during)
**When to post:** Immediately after competition deadline
**Why it works:** Solution sharing posts are among the highest-upvoted content on all of Kaggle. Top solutions routinely earn gold medals.

```
Title: [Xth Place Solution] [Competition Name] - [Brief description of approach]

## Summary
- **Final Position:** [Xth place / Top Y%]
- **Public LB:** [score]
- **Private LB:** [score]
- **Shake up/down:** [+/- X positions]

## Solution Overview
[2-3 paragraph summary of the complete pipeline]

## Architecture / Pipeline
```
[ASCII diagram or description of the pipeline]
Input -> Preprocessing -> Feature Engineering -> Model(s) -> Post-processing -> Submission
```

## What Worked (with ablation)
| Component | Score Without | Score With | Delta |
|-----------|-------------|-----------|-------|
| [Component 1] | X.XXX | X.XXX | +X.XXX |
| [Component 2] | X.XXX | X.XXX | +X.XXX |

## What Didn't Work
- [Failed approach 1 and why]
- [Failed approach 2 and why]

## Hardware & Training
- GPU: [type]
- Training time: [X hours]
- Inference time: [X minutes]

## Code
[GitHub link or notebook link]

## Acknowledgments
Thanks to [people/resources that helped].

Congrats to the winners and all participants!
```

---

### Template 11: Weekly Competition Tips Thread

**Where to post:** Competition Discussion
**When to post:** Every Monday or at regular intervals during a competition

```
Title: [Competition Name] Week [X] Tips & Observations

Here are some things I've noticed this week that might help:

## Tip 1: [Title]
[Description with code/evidence]

## Tip 2: [Title]
[Description with code/evidence]

## Community Observations
- [Observation about leaderboard trends]
- [Observation about common approaches in public notebooks]

## My Progress This Week
- [What you tried and learned]

What tips do others have from this week?
```

---

### Template 12: Comparison / Benchmark Post

**Where to post:** Competition Discussion or Getting Started
**When to post:** When you've run systematic experiments

```
Title: [Competition Name] Benchmark: Comparing [X] Approaches / Models

I ran a systematic comparison of [X approaches/models] on the [competition] data with consistent preprocessing and validation.

## Setup
- Validation: [method]
- Hardware: [specs]
- Preprocessing: [consistent across all experiments]

## Results

| Model/Approach | CV Score | Public LB | Training Time | Inference Time |
|---------------|----------|-----------|---------------|----------------|
| [Model 1] | X.XXX | X.XXX | Xm | Xs |
| [Model 2] | X.XXX | X.XXX | Xm | Xs |
| [Model 3] | X.XXX | X.XXX | Xm | Xs |

## Analysis
[Discussion of results, surprising findings, trade-offs]

## Recommendations
- For quick iteration: [Model X]
- For best score: [Model Y]
- For production: [Model Z]

Notebook with full code: [link]
```

---

## 3. Comment Templates

Comments are often easier to earn bronze medals from than full posts, because you can write many more of them and the bar for a helpful comment is lower.

### Template C1: Helpful Answer to a Question

```
Great question! I ran into the same issue. Here's what worked for me:

```python
# Specific code solution
df['column'] = df['column'].fillna(df['column'].median())
```

The key insight is [explanation of WHY this works, not just what to do]. This happens because [technical reason].

If you're still stuck, check out [specific resource/notebook link] which covers this in more detail.
```

---

### Template C2: Constructive Feedback on a Notebook

```
Nice work on this notebook! A few observations:

1. **[Positive feedback]:** Your [specific section] is really well done, especially [specific element].

2. **Potential improvement:** In your [section], you might get better results by [specific suggestion]. I found that [technique] improved my score by [X] in a similar setup.

3. **Minor note:** On line [X], [specific code improvement or bug fix].

Have you tried [alternative approach]? It could complement your current pipeline nicely.
```

---

### Template C3: Adding Insight to an Existing Discussion

```
Building on what @[username] said, I've found that [additional insight or nuance].

In my experiments:
- [Specific finding 1]
- [Specific finding 2]

This suggests that [interpretation]. One thing to watch out for is [caveat or edge case that the original post didn't cover].
```

---

### Template C4: Sharing Relevant Resources

```
This is a great discussion. A few resources that might be relevant:

1. **[Resource name]** ([link]) - [1-sentence description of relevance]
2. **[Resource name]** ([link]) - [1-sentence description of relevance]
3. Similar approach was used in [past competition name] ([link]) - the [Xth place solution] used a similar technique.

The [first resource] is especially relevant because [specific connection to the discussion topic].
```

---

### Template C5: Code Improvement Suggestion

```
Nice approach! One optimization that might help:

Instead of:
```python
# Original code from the post
for i in range(len(df)):
    result.append(process(df.iloc[i]))
```

You could use:
```python
# Vectorized version
result = df.apply(process, axis=1)
# Or even better with native pandas:
result = df['column'].map(process_func)
```

This should be [X]x faster because [explanation of why vectorized operations are faster in pandas/numpy]. I tested this on [similar dataset] and saw [specific speedup].
```

---

### Template C6: Clarifying a Misunderstanding

```
I think there might be a small misunderstanding here. [Clarify the point politely.]

The [metric/concept/API] actually works by [correct explanation]. The confusion might come from [why the misunderstanding is common].

Here's a quick example to illustrate:
```python
# Code demonstrating the correct behavior
```

The documentation covers this at [link]. Hope that helps!
```

---

### Template C7: Sharing Experimental Results

```
I tried the approach mentioned in this thread. Here are my results:

- **Baseline (without this technique):** [score]
- **With this technique:** [score]
- **Improvement:** [+/- X.XXX]

Some notes:
- It worked best when [specific condition]
- It did NOT improve results when [specific condition]
- Key hyperparameters I tuned: [list]

My code for this experiment: [link or inline code]
```

---

### Template C8: Thanking and Extending

```
Thanks for sharing this @[username], this is really useful.

I extended your approach by [specific modification] and it [result]. Specifically:

```python
# Your modified version of their code
```

The key change was [explanation]. This gave me an additional [improvement/insight].
```

---

### Template C9: Asking a Follow-up Question (in a thread)

```
Really interesting findings. I have a follow-up question:

When you applied [technique], did you account for [specific consideration]? I ask because in [related context], I found that [observation], which suggests [potential issue or nuance].

Have you tested this with [variation]? I'm curious whether [specific hypothesis].
```

---

### Template C10: Providing Context from a Different Domain

```
Interesting approach! This reminds me of how [similar technique] is used in [different domain/field].

In [field], the standard approach to this type of problem is [description], which differs from what's commonly done here because [explanation]. The key insight that might transfer is [specific transferable idea].

[Author name]'s paper "[Paper Title]" ([link]) covers this well and might give some ideas for adaptation.
```

---

### Template C11: Debugging Help

```
I think I see the issue. The problem is likely in [specific line/section]:

The error occurs because [explanation]. To fix this:

```python
# Fixed code
```

I had the same error when [your experience]. The root cause was [explanation].

If that doesn't work, try adding `print(variable.shape)` before line [X] to check whether [diagnostic hypothesis].
```

---

### Template C12: Competition-Specific Observation

```
I noticed something interesting about this competition's evaluation metric:

[Specific observation, e.g., "The weighted F1 score penalizes false negatives on class X much more heavily than class Y because..."]

This means that optimizing for [specific sub-goal] might be more important than [what people might assume]. In my experiments, focusing on [specific class/aspect] improved my score more than improving overall accuracy.

Has anyone else noticed this trade-off?
```

---

## 4. Weekly Engagement Routine

### Daily Routine (15-30 minutes)

| Time | Activity | Details |
|------|----------|---------|
| Morning (10 min) | **Scan new discussions** | Check discussion tabs of 2-3 active competitions you're following. Look for questions you can answer or discussions where you can add value. |
| Morning (5 min) | **Respond to replies** | Check your notifications. Respond to anyone who replied to your posts or comments. This keeps threads active and visible. |
| Afternoon (10-15 min) | **Write 2-3 comments** | Post substantive comments on discussions where you can genuinely contribute. Prioritize: unanswered questions > active threads > new threads. |

**Daily targets:**
- Read 5-10 discussion posts
- Write 2-3 substantive comments
- Upvote genuinely helpful content (this builds goodwill and visibility)

### Weekly Routine

| Day | Activity | Template to Use |
|-----|----------|----------------|
| **Monday** | Create 1 original discussion post on most active competition | EDA Findings, Strategy Discussion, or Resource Compilation |
| **Tuesday** | Comment on 5+ discussions across different competitions | Mix of answer, insight, and resource-sharing templates |
| **Wednesday** | Publish or update a notebook, then create a walkthrough post | Notebook Walkthrough template |
| **Thursday** | Comment on 5+ discussions, focus on unanswered questions | Helpful Answer and Debugging templates |
| **Friday** | Create 1 original discussion post | Technique Tutorial or Comparison/Benchmark |
| **Saturday** | Deep work on competition (generates material for discussions) | -- |
| **Sunday** | Plan next week's discussion topics, review what earned medals | -- |

**Weekly targets:**
- 2-3 original discussion posts
- 12-18 substantive comments
- 1 notebook published or updated
- Expected medal yield: 4-8 bronze medals per week

### Monthly Routine

| Week | Monthly Activity |
|------|-----------------|
| Week 1 | Competition retrospective (if any competitions ended) |
| Week 2 | Publish a resource compilation post for upcoming deadlines |
| Week 3 | Write a technique tutorial based on your competition work |
| Week 4 | Review medal progress, adjust strategy based on what's working |

---

## 5. Content Calendar

### Phase 1: Weeks 1-4 (Foundation)

**Active competitions to engage with:**

#### Week 1 (Jan 27 - Feb 2)
| Day | Post/Activity | Competition | Template |
|-----|--------------|-------------|----------|
| Mon | "CAFA 6 Final Week: Common Pitfalls & Last-Minute Optimizations" | CAFA 6 (deadline Feb 2) | Strategy Discussion |
| Tue | Comment on 5 CAFA 6 discussions (last-minute tips) | CAFA 6 | Helpful Answers |
| Wed | Publish protein function EDA notebook + walkthrough post | CAFA 6 | Notebook Walkthrough |
| Thu | "Vesuvius Surface Detection: EDA of Training Labels & Tile Analysis" | Vesuvius (deadline Feb 13) | EDA Findings |
| Fri | Comment on Vesuvius and Med-Gemma discussions | Multiple | Mixed templates |
| Sat | Deep work on Vesuvius competition | -- | -- |
| Sun | "CAFA 6 Retrospective: What I Learned About Protein Function Prediction" | CAFA 6 | Lessons Learned |

#### Week 2 (Feb 3 - Feb 9)
| Day | Post/Activity | Competition | Template |
|-----|--------------|-------------|----------|
| Mon | "Vesuvius Challenge: Resource Compilation (Papers, Baselines, Tools)" | Vesuvius | Resource Compilation |
| Tue | Comment on 5+ Vesuvius discussions | Vesuvius | Mixed |
| Wed | "Med-Gemma Impact Challenge: Medical AI Evaluation Metrics Explained" | Med-Gemma | Technique Tutorial |
| Thu | Comment on Med-Gemma and AIMO3 discussions | Multiple | Mixed |
| Fri | Update Vesuvius notebook, post walkthrough | Vesuvius | Notebook Walkthrough |
| Sat | Deep work on Vesuvius + Med-Gemma | -- | -- |
| Sun | Plan next week, review medal count | -- | -- |

#### Week 3 (Feb 10 - Feb 16)
| Day | Post/Activity | Competition | Template |
|-----|--------------|-------------|----------|
| Mon | "Vesuvius Challenge: Final Push Tips & Validation Strategy" | Vesuvius (deadline Feb 13) | Strategy Discussion |
| Tue | Comment blitz on Vesuvius discussions (last days) | Vesuvius | Mixed |
| Wed | Submit Vesuvius entry + share approach | Vesuvius | Solution Sharing |
| Thu | "Vesuvius Challenge Retrospective: Surface Detection Lessons" | Vesuvius | Lessons Learned |
| Fri | "AIMO3 Math Olympiad: Resource List (LLM Reasoning, Math AI)" | AIMO3 | Resource Compilation |
| Sat | Deep work on AIMO3 or Akkadian | -- | -- |
| Sun | "Med-Gemma Challenge: Halfway Point Progress & Insights" | Med-Gemma | Strategy Discussion |

#### Week 4 (Feb 17 - Feb 23)
| Day | Post/Activity | Competition | Template |
|-----|--------------|-------------|----------|
| Mon | "Deep Past Akkadian: NLP Approaches for Ancient Language Translation" | Akkadian (deadline Mar 23) | Technique Tutorial |
| Tue | Comment on Akkadian and Med-Gemma discussions | Multiple | Mixed |
| Wed | "Med-Gemma Final Week: Key Findings & What Worked" | Med-Gemma (deadline Feb 24) | EDA Findings |
| Thu | Comment blitz on Med-Gemma discussions | Med-Gemma | Mixed |
| Fri | "Med-Gemma Impact Challenge: My Approach & Results" | Med-Gemma | Solution Sharing |
| Sat | Work on Akkadian competition | -- | -- |
| Sun | Month 1 review: count medals, assess what post types performed best | -- | -- |

**Expected medals after Phase 1: 15-25 bronze medals**

### Phase 2: Weeks 5-8 (Acceleration)

**Focus competitions:** Akkadian Translation, AIMO3, Stanford RNA, new competitions

#### Key Posts to Create
1. "Akkadian Translation: Comparing Seq2Seq vs. LLM-Based Translation Approaches" (Benchmark)
2. "AIMO3: How to Set Up a Local Math Evaluation Pipeline" (Technique Tutorial)
3. "Stanford RNA 3D Folding 2: Resource Compilation" (Resource List)
4. "Stanford RNA 3D Folding: EDA & Initial Structural Analysis" (EDA Findings)
5. "AIMO3 Week [X] Tips: What's Working on the Leaderboard" (Weekly Tips)
6. 2-3 Notebook Walkthroughs for new notebook publications
7. Bug report / data issue posts as discovered
8. "Akkadian Translation: Ancient Language Processing Resources for NLP Engineers" (Resource Compilation)

**Phase 2 also leverages your 17 existing notebooks:**
- Create discussion posts walking through your best-performing notebooks
- Post "behind the scenes" discussions about your notebook methodology
- Reference notebooks in competition discussions where relevant

**Expected medals after Phase 2: 30-45 cumulative bronze medals**

### Phase 3: Weeks 9-12 (Closing the Gap)

By this point, you should know which post types and competitions yield the best medal returns. Double down on what works.

#### Key Posts to Create
1. Competition retrospectives for any competitions that ended
2. Solution sharing posts
3. "Lessons from X Competitions" meta-post
4. Continue technique tutorials and EDA posts for active competitions
5. Engage with any new Featured competitions that launch
6. Cross-reference your 4 datasets in relevant competition discussions

**Expected medals after Phase 3: 50+ cumulative bronze medals = Discussion Expert**

### Leveraging Your Existing 17 Notebooks

Create discussion posts for your strongest notebooks:

| Notebook Topic | Discussion Post Angle |
|---------------|----------------------|
| EDA notebooks | "Deep Dive into [Dataset]: Key Patterns I Found" |
| Model notebooks | "Comparing [Models] on [Task]: Full Benchmark" |
| Tutorial notebooks | "[Technique] Explained with Code: Practical Guide" |
| Competition notebooks | "[Competition] Approach Walkthrough: From EDA to Submission" |

For each notebook, create at least one discussion post that adds context, explains decisions, and invites feedback. This can yield 10-15 additional medal opportunities from existing work.

### Trending Kaggle Topics to Post About

These topics consistently attract high engagement:

1. **LLM/Foundation Model Fine-tuning** -- Any discussion about fine-tuning LLMs for specific tasks
2. **Multimodal AI** -- Combining vision and language models (relevant to Med-Gemma)
3. **Efficient Inference** -- Techniques for faster inference within Kaggle's resource constraints
4. **Novel Architectures** -- Discussion of new papers and architectures (Mamba, State Space Models, etc.)
5. **Data-Centric AI** -- Data quality, labeling strategies, augmentation techniques
6. **AI for Science** -- Protein folding, drug discovery, materials science (relevant to CAFA 6, RNA Folding)
7. **Code LLMs** -- AI for code generation and understanding
8. **Kaggle Meta-Discussions** -- Tips for newcomers, competition strategy, platform features

---

## 6. Cross-Promotion Strategy

### Linking Notebooks in Discussions

**Do:**
- Reference your notebook when it directly addresses the discussion topic
- Include the link naturally within helpful content (not as the main point of the post)
- Add a brief summary of what the notebook contains so people know if it's worth clicking
- Use the format: "I explored this in my notebook [Title](link), specifically in the [section name] section"

**Don't:**
- Post comments that are just links to your notebook with no other content
- Spam your notebook link across unrelated discussions
- Make every discussion post a thinly-veiled notebook promotion

**Example of good cross-promotion:**
```
Great question about handling class imbalance in this competition. I found that
SMOTE combined with stratified k-fold gave the most stable results.

Here's the key code:
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

I go into more detail on this in my notebook [Competition Name: Handling
Imbalanced Medical Data](link) where I compare 5 different resampling strategies.
```

### Referencing Datasets in Competition Discussions

When your datasets are relevant to a competition:
```
For those looking for supplementary data, I published a dataset that might be
useful: [Dataset Name](link). It contains [X] samples of [description] which
could help with [specific aspect of the competition].

I've verified it's compatible with the competition data format. Here's how to
load it alongside the competition data:
```python
# Quick integration code
```
```

### Building a Recognizable Presence

1. **Consistent username and avatar:** Your profile at kaggle.com/lorenzoscaturchio should have a recognizable avatar and complete bio.

2. **Signature topics:** Develop expertise in 2-3 areas and become the "go-to person" for those topics in discussions. Given your portfolio:
   - Medical AI / Healthcare ML (Med-Gemma, medical datasets)
   - NLP for specialized domains (Akkadian translation, RAG)
   - Financial ML / Trading analysis

3. **Reply to comments on your posts:** Always engage with people who comment on your discussion posts. This builds relationships and keeps your threads active (more visibility = more potential upvotes).

4. **Credit others:** When you build on someone else's work, tag them with @username. They'll likely see your post and may upvote or engage.

5. **Consistent posting schedule:** People start to recognize and look for your posts when you maintain a regular cadence.

6. **Profile bio should mention your areas:** Include your focus areas so people landing on your profile understand your expertise.

---

## 7. Progress Tracking Checklist

### Medal Progress Tracker

Track each medal as you earn it. You need 50 bronze (or higher) discussion medals.

#### Bronze Discussion Medals (Target: 50)

**Batch 1 (Medals 1-10)**
- [ ] Medal 1: Post/Comment: _________________ Date: _______
- [ ] Medal 2: Post/Comment: _________________ Date: _______
- [ ] Medal 3: Post/Comment: _________________ Date: _______
- [ ] Medal 4: Post/Comment: _________________ Date: _______
- [ ] Medal 5: Post/Comment: _________________ Date: _______
- [ ] Medal 6: Post/Comment: _________________ Date: _______
- [ ] Medal 7: Post/Comment: _________________ Date: _______
- [ ] Medal 8: Post/Comment: _________________ Date: _______
- [ ] Medal 9: Post/Comment: _________________ Date: _______
- [ ] Medal 10: Post/Comment: ________________ Date: _______

**Batch 2 (Medals 11-20)**
- [ ] Medal 11: Post/Comment: ________________ Date: _______
- [ ] Medal 12: Post/Comment: ________________ Date: _______
- [ ] Medal 13: Post/Comment: ________________ Date: _______
- [ ] Medal 14: Post/Comment: ________________ Date: _______
- [ ] Medal 15: Post/Comment: ________________ Date: _______
- [ ] Medal 16: Post/Comment: ________________ Date: _______
- [ ] Medal 17: Post/Comment: ________________ Date: _______
- [ ] Medal 18: Post/Comment: ________________ Date: _______
- [ ] Medal 19: Post/Comment: ________________ Date: _______
- [ ] Medal 20: Post/Comment: ________________ Date: _______

**Batch 3 (Medals 21-30)**
- [ ] Medal 21: Post/Comment: ________________ Date: _______
- [ ] Medal 22: Post/Comment: ________________ Date: _______
- [ ] Medal 23: Post/Comment: ________________ Date: _______
- [ ] Medal 24: Post/Comment: ________________ Date: _______
- [ ] Medal 25: Post/Comment: ________________ Date: _______
- [ ] Medal 26: Post/Comment: ________________ Date: _______
- [ ] Medal 27: Post/Comment: ________________ Date: _______
- [ ] Medal 28: Post/Comment: ________________ Date: _______
- [ ] Medal 29: Post/Comment: ________________ Date: _______
- [ ] Medal 30: Post/Comment: ________________ Date: _______

**Batch 4 (Medals 31-40)**
- [ ] Medal 31: Post/Comment: ________________ Date: _______
- [ ] Medal 32: Post/Comment: ________________ Date: _______
- [ ] Medal 33: Post/Comment: ________________ Date: _______
- [ ] Medal 34: Post/Comment: ________________ Date: _______
- [ ] Medal 35: Post/Comment: ________________ Date: _______
- [ ] Medal 36: Post/Comment: ________________ Date: _______
- [ ] Medal 37: Post/Comment: ________________ Date: _______
- [ ] Medal 38: Post/Comment: ________________ Date: _______
- [ ] Medal 39: Post/Comment: ________________ Date: _______
- [ ] Medal 40: Post/Comment: ________________ Date: _______

**Batch 5 (Medals 41-50) -- HOME STRETCH**
- [ ] Medal 41: Post/Comment: ________________ Date: _______
- [ ] Medal 42: Post/Comment: ________________ Date: _______
- [ ] Medal 43: Post/Comment: ________________ Date: _______
- [ ] Medal 44: Post/Comment: ________________ Date: _______
- [ ] Medal 45: Post/Comment: ________________ Date: _______
- [ ] Medal 46: Post/Comment: ________________ Date: _______
- [ ] Medal 47: Post/Comment: ________________ Date: _______
- [ ] Medal 48: Post/Comment: ________________ Date: _______
- [ ] Medal 49: Post/Comment: ________________ Date: _______
- [ ] Medal 50: Post/Comment: ________________ Date: _______ **DISCUSSION EXPERT!**

### Weekly Activity Log

| Week | Posts Created | Comments Made | Medals Earned | Cumulative | Notes |
|------|-------------|---------------|---------------|------------|-------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |
| 9 | | | | | |
| 10 | | | | | |
| 11 | | | | | |
| 12 | | | | | |

### Content Type Performance Tracker

Track which types of posts/comments earn medals most reliably:

| Content Type | Posts/Comments Made | Medals Earned | Medal Rate |
|-------------|-------------------|---------------|------------|
| EDA Findings | | | |
| Bug Reports | | | |
| Technique Tutorials | | | |
| Lessons Learned | | | |
| Resource Compilations | | | |
| Strategy Discussions | | | |
| Notebook Walkthroughs | | | |
| Dataset Announcements | | | |
| Smart Questions | | | |
| Solution Sharing | | | |
| Helpful Answers (comments) | | | |
| Code Improvements (comments) | | | |
| Resource Sharing (comments) | | | |
| Insight Addition (comments) | | | |

### Monthly Review Questions

At the end of each month, answer these:

1. How many medals did I earn this month? ___
2. What type of content earned the most medals? ___
3. Which competition forum was most productive? ___
4. What should I do more of next month? ___
5. What should I stop doing? ___
6. Am I on track for 50 medals by target date? ___

---

## Quick Reference Card

### The 5 Highest-ROI Activities for Discussion Medals

1. **Answer unanswered questions** in active competition forums (2 min per comment, high medal rate)
2. **Post EDA findings** within the first week of a new competition (30-60 min, very high medal rate)
3. **Share bug reports / data issues** when you find them (10 min, very high medal rate)
4. **Write post-competition retrospectives** after deadlines (30-60 min, high medal rate)
5. **Comment with code improvements** on popular notebooks' discussion threads (5 min per comment, moderate medal rate)

### Rules of Thumb

- **1 substantive comment = 1 potential bronze medal.** You need 50. Do the math.
- **Quality over quantity.** One genuinely helpful answer beats ten "Great work!" comments.
- **Be early.** First-mover advantage is real on Kaggle. Post in the first days of a competition.
- **Be specific.** Vague advice gets ignored. Specific code, numbers, and examples get upvoted.
- **Be generous.** Share your findings freely. The more you give, the more you get back.
- **Never ask for upvotes.** It violates community norms and will get you downvoted.
- **Track your progress.** Use the checklist above. What gets measured gets managed.

---

*Strategy created: January 25, 2026*
*Profile: [kaggle.com/lorenzoscaturchio](https://www.kaggle.com/lorenzoscaturchio)*
*Target: 50 bronze discussion medals -> Discussion Expert tier*
