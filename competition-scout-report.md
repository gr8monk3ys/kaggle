# Kaggle Medal-Eligible Competition Scout Report

**Date:** January 25, 2026
**Scope:** Active Featured, Research, and Analytics competitions (medal-eligible only -- excludes Playground, Getting Started, InClass)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Medal Threshold Reference](#medal-threshold-reference)
3. [Active Medal-Eligible Competitions](#active-medal-eligible-competitions)
4. [Recently Closed Competitions](#recently-closed-competitions)
5. [Medal Accessibility Ranking](#medal-accessibility-ranking)
6. [Recommended Strategy](#recommended-strategy)
7. [Sources](#sources)

---

## Executive Summary

As of January 25, 2026, there are approximately **8-10 medal-eligible competitions** actively accepting submissions on Kaggle. The landscape spans diverse domains: computer vision, NLP/translation, bioinformatics, financial modeling, sports analytics, mathematical reasoning, and signal processing. Prize pools range from $50,000 to $2.2M+.

**Key finding:** The competitions with the most accessible medal opportunities for beginner-intermediate competitors are the **Deep Past Akkadian Translation Challenge** (niche NLP, early stage, fewer teams expected), the **Vesuvius Challenge - Surface Detection** (66 teams as of late January), and the **CAFA 6 Protein Function Prediction** (strong public notebooks/starter code available). The **Hull Tactical Market Prediction** also offers long runway (deadline June 2026) for iterative improvement.

---

## Medal Threshold Reference

Kaggle awards medals based on your final private leaderboard rank relative to the total number of teams:

| Team Count | Bronze | Silver | Gold |
|---|---|---|---|
| 0-99 (Small) | Top 40% | Top 20% | Top 10% |
| 100-249 (Medium) | Top 40% | Top 20% | Top 10 teams |
| 250-999 (Large) | Top 10% | Top 5% | Top 10 teams |
| 1000+ (Very Large) | Top 10% | Top 5% | Top 10 + 0.2% of teams |

**Important:** Only Featured, Research, and Analytics competitions award tier-progression medals. Playground, Getting Started, and InClass do NOT.

---

## Active Medal-Eligible Competitions

### 1. Hull Tactical - Market Prediction

| Field | Detail |
|---|---|
| **URL** | https://www.kaggle.com/competitions/hull-tactical-market-prediction |
| **Type** | Featured |
| **Prize** | $100,000 ($50K first place) |
| **Entry Deadline** | December 8, 2025 (CLOSED for new entries) |
| **Final Deadline** | June 16, 2026 |
| **Teams (est.)** | Several hundred (exact count unavailable) |
| **Medal Thresholds (est.)** | Large tier (250-999): Bronze top 10%, Silver top 5%, Gold top 10 |
| **Domain** | Time Series / Financial Modeling |
| **Evaluation** | Modified Sharpe Ratio |
| **Difficulty** | HIGH -- Requires quantitative finance knowledge; predicting S&P 500 excess returns with volatility constraints. Uses Hull Tactical proprietary signals. Live evaluation against market data. |

**Notes:** Entry deadline has passed but submitted models continue to be evaluated through June 2026. This is a "forecasting" style competition where models run on live data. The modified Sharpe ratio metric rewards risk-adjusted returns, making it challenging for pure ML approaches.

---

### 2. AI Mathematical Olympiad - Progress Prize 3 (AIMO3)

| Field | Detail |
|---|---|
| **URL** | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3 |
| **Type** | Featured |
| **Prize** | $2,207,152 + $110,000 additional prizes |
| **Entry Deadline** | April 8, 2026 |
| **Final Deadline** | April 15, 2026 |
| **Teams (est.)** | Very Large (1000+); AIMO2 had 2000+ teams |
| **Medal Thresholds (est.)** | Very Large tier: Bronze top 10%, Silver top 5%, Gold top 10 + 0.2% |
| **Domain** | Mathematical Reasoning / LLM |
| **Evaluation** | Accuracy on 110 original olympiad-level math problems |
| **Difficulty** | VERY HIGH -- Requires building/fine-tuning open-source LLMs for mathematical reasoning at IMO level. Problems span algebra, combinatorics, geometry, number theory. Up to 128 H100 GPUs available for select participants. |

**Notes:** The biggest prize pool of any active competition. However, this is an extremely competitive space dominated by well-funded teams from NVIDIA, major research labs, etc. Open-source requirement levels the playing field somewhat. All code must be publicly available. Winners present at AI Day at the 2026 IMO in Shanghai.

---

### 3. CAFA 6 Protein Function Prediction

| Field | Detail |
|---|---|
| **URL** | https://www.kaggle.com/competitions/cafa-6-protein-function-prediction |
| **Type** | Featured (Research-oriented) |
| **Prize** | $50,000 |
| **Entry Deadline** | January 26, 2026 (TOMORROW -- urgent!) |
| **Final Deadline** | February 2, 2026 |
| **Teams (est.)** | Large (CAFA 5 had 1,600+ teams) |
| **Medal Thresholds (est.)** | Very Large tier if similar to CAFA 5: Bronze top 10%, Silver top 5%, Gold top 10 + 0.2% |
| **Domain** | Bioinformatics / Protein Science |
| **Evaluation** | Prediction of Gene Ontology (GO) terms for proteins |
| **Kernel Only** | No (external submissions allowed) |
| **Difficulty** | MEDIUM-HIGH -- Protein language models (ESM2, ProtTrans) provide strong baselines. Domain knowledge in biology helps but public starter notebooks are available. Multi-label classification problem. |

**Notes:** Entry deadline is January 26, 2026 -- action needed immediately if interested. CAFA is a well-established bioinformatics challenge series. Leveraging pre-trained protein language model embeddings (ESM2, ProtBERT) with standard ML classifiers can produce competitive results even without deep biology expertise. Good starter notebooks available on Kaggle.

---

### 4. CSIRO - Image2Biomass Prediction

| Field | Detail |
|---|---|
| **URL** | https://www.kaggle.com/competitions/csiro-biomass |
| **Type** | Featured |
| **Prize** | $75,000 ($50K / $20K / $5K) |
| **Entry Deadline** | January 21, 2026 (CLOSED) |
| **Final Deadline** | January 28, 2026 (3 days away) |
| **Teams (est.)** | Medium-Large (hundreds) |
| **Medal Thresholds (est.)** | Large tier: Bronze top 10%, Silver top 5%, Gold top 10 |
| **Domain** | Computer Vision / Agriculture |
| **Evaluation** | Globally weighted R-squared |
| **Kernel Only** | Yes |
| **Difficulty** | MEDIUM -- Image regression task. 1,162 annotated top-view pasture images across 19 Australian locations. Additional metadata (species, season, location, height, NDVI) available. Standard CV pipelines (EfficientNet, ConvNeXt) with metadata fusion likely competitive. |

**Notes:** Entry is closed. Final submissions due January 28. Listed here for reference as results will be announced soon. This was a well-structured CV competition with strong starter code and clear evaluation.

---

### 5. Vesuvius Challenge - Surface Detection

| Field | Detail |
|---|---|
| **URL** | https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection |
| **Type** | Research (Code Competition) |
| **Prize** | $100,000 (to top 10 participants) |
| **Entry Deadline** | February 6, 2026 |
| **Final Deadline** | ~February 2026 |
| **Teams** | 66 teams (as of late January 2026) |
| **Medal Thresholds** | Small tier (0-99): Bronze top 40% (~26 teams), Silver top 20% (~13 teams), Gold top 10% (~7 teams) |
| **Domain** | Computer Vision / 3D Segmentation |
| **Evaluation** | Weighted average of Surface Dice, TopoScore, and VOI |
| **Difficulty** | MEDIUM-HIGH -- 3D CT scan segmentation of ancient scroll surfaces. Specialized domain but well-documented from prior Vesuvius challenges. Requires 3D medical/volumetric imaging experience. Relatively few teams = wider medal bands. |

**Notes:** With only 66 teams, this has the widest medal thresholds of any active competition. Bronze requires only top 40% (roughly top 26 teams). The domain is niche (3D CT segmentation of Herculaneum scrolls), which limits competition. Prior Vesuvius Challenge work provides substantial public resources and code. This is one of the best medal opportunities currently available.

---

### 6. Deep Past Challenge - Translate Akkadian to English

| Field | Detail |
|---|---|
| **URL** | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation |
| **Type** | Featured |
| **Prize** | $50,000 (first place up to $15,000) |
| **Entry Deadline** | March 23, 2026 |
| **Final Deadline** | March 2026 |
| **Teams (est.)** | Small-Medium (newly launched Dec 16, 2025) |
| **Medal Thresholds (est.)** | Depends on final team count; likely Medium tier |
| **Domain** | NLP / Machine Translation |
| **Evaluation** | Translation quality of Akkadian cuneiform to English |
| **Difficulty** | MEDIUM -- Niche NLP/translation task. Modern LLMs and transformer-based translation models provide strong starting points. The niche domain (ancient Akkadian) limits the field of competitors with domain expertise. Supported by XTX Markets. |

**Notes:** Relatively new competition with long runway (2+ months remaining). The niche domain of ancient cuneiform translation means fewer competitors with specialized knowledge, potentially making medals more accessible. Modern NLP techniques (fine-tuning LLMs, seq2seq models) should transfer well. This is a strong candidate for medal hunting.

---

### 7. NFL Big Data Bowl 2026 - Prediction

| Field | Detail |
|---|---|
| **URL** | https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction |
| **Type** | Featured |
| **Prize** | $50,000 (Prediction track); $100,000 total across both tracks |
| **Entry Deadline** | November 26, 2025 (CLOSED) |
| **Final Deadline** | Early 2026 (live evaluation on Weeks 14-18) |
| **Teams (est.)** | Large (hundreds; NFL BDB is popular) |
| **Medal Thresholds (est.)** | Large tier: Bronze top 10%, Silver top 5%, Gold top 10 |
| **Domain** | Sports Analytics / Time Series / Tabular |
| **Evaluation** | Predicted vs actual player locations (NGS data) |
| **Difficulty** | MEDIUM -- Predict player trajectories post-pass. Requires spatiotemporal modeling. NFL tracking data is rich and well-documented. Many public notebooks from prior BDB editions. |

**Notes:** Entry is closed but live evaluation continues. Finalists present at the NFL Scouting Combine. This competition blends tabular data, time series, and spatial prediction in an accessible sports domain.

---

### 8. NFL Big Data Bowl 2026 - Analytics

| Field | Detail |
|---|---|
| **URL** | https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics |
| **Type** | Analytics |
| **Prize** | Part of $100,000 total pool |
| **Entry Deadline** | Early 2026 |
| **Final Deadline** | Early 2026 |
| **Teams (est.)** | Medium (analytics track typically smaller than prediction) |
| **Medal Thresholds (est.)** | Medium tier: Bronze top 40%, Silver top 20%, Gold top 10 |
| **Domain** | Sports Analytics / Data Storytelling |
| **Evaluation** | Judged by NFL data analysts (not automated metric) |
| **Difficulty** | MEDIUM -- Notebook/report-based submission. Focus on analysis and storytelling rather than pure ML performance. Accessible for data analysts. University and Broadcast tracks available. |

**Notes:** Analytics competitions are judged by humans (NFL data analysts) rather than a leaderboard metric. This makes medal distribution less predictable but rewards strong analytical writing and visualization.

---

### 9. OFC 2026 ML Challenge

| Field | Detail |
|---|---|
| **URL** | https://www.kaggle.com/competitions/ofc-2026-ml-challenge |
| **Type** | Research (likely) |
| **Prize** | Publication in IEEE/Optica JOCN journal (no confirmed cash prize) |
| **Entry Deadline** | January 30, 2026 |
| **Final Deadline** | January 30, 2026 |
| **Teams (est.)** | Small (niche optical communications domain) |
| **Medal Thresholds (est.)** | Small tier if <100 teams: Bronze top 40%, Silver top 20%, Gold top 10% |
| **Domain** | Signal Processing / Optical Communications |
| **Evaluation** | Task-specific prediction accuracy |
| **Difficulty** | HIGH -- Very specialized domain (optical fiber communications). Requires domain expertise in photonics/optical networking. Winners present at OFC 2026 conference in LA (March 15-19). |

**Notes:** Extremely niche competition from the optical fiber communications community. Medal eligibility uncertain -- may be a Community competition rather than Featured/Research. If medal-eligible, the small field makes medals accessible for anyone with relevant domain knowledge.

---

### 10. Gemini 3 Hackathon (Vibe Code with Gemini 3 Pro)

| Field | Detail |
|---|---|
| **URL** | https://www.kaggle.com/competitions/gemini-3 |
| **Type** | Hackathon (new format) |
| **Prize** | $100,000 cash + $500,000 in API credits |
| **Entry Deadline** | February 10, 2026 |
| **Teams (est.)** | Very Large (Google competitions attract thousands) |
| **Domain** | Application Development / LLM |
| **Difficulty** | MEDIUM -- Build applications using Gemini 3 Pro API. Judged on technical execution (40%), impact (20%), innovation (30%), presentation (10%). |

**Notes:** This is a **Hackathon** format, which is a newer Kaggle competition type. Medal eligibility for hackathons may differ from traditional competitions -- verify before investing significant effort. Focus is on building applications, not ML models.

---

## Recently Closed Competitions (for context)

| Competition | Deadline | Prize | Type |
|---|---|---|---|
| PhysioNet - Digitization of ECG Images | Jan 15, 2026 | $50,000 | Featured (CV/Signal) |
| CSIRO - Image2Biomass Prediction | Jan 28, 2026 | $75,000 | Featured (CV) |
| Konwinski Prize (K Prize) Round 1 | Mar 12, 2025 | $100K-$1.225M | Featured (Code/AI Agents) |

---

## Medal Accessibility Ranking

Ranked from **most accessible** to **least accessible** for a beginner-intermediate ML competitor:

### Tier 1: Best Medal Opportunities

| Rank | Competition | Why |
|---|---|---|
| **1** | **Vesuvius Challenge - Surface Detection** | Only 66 teams. Bronze = top 40% (~26 teams). Small tier thresholds are extremely generous. Prior Vesuvius work provides extensive public resources. 3D segmentation is learnable with existing CV knowledge. |
| **2** | **Deep Past - Translate Akkadian** | Niche domain limits competitors. Long runway (March 23 deadline). Modern NLP/LLM techniques transfer well. Few people have Akkadian expertise, leveling the field. |
| **3** | **NFL Big Data Bowl 2026 - Analytics** | Human-judged (not leaderboard). Rewards analysis quality and storytelling. Accessible for data analysts without deep ML expertise. Sports data is intuitive. |

### Tier 2: Good Opportunities with More Effort

| Rank | Competition | Why |
|---|---|---|
| **4** | **CAFA 6 Protein Function Prediction** | Strong public baselines using protein language models. But entry deadline is TOMORROW (Jan 26). Large field expected. External submissions allowed (not kernel-only). |
| **5** | **Hull Tactical Market Prediction** | Long evaluation period (through June 2026). But financial prediction is inherently noisy and competitive. Entry is closed for new participants. |
| **6** | **OFC 2026 ML Challenge** | Very small field but highly specialized domain. Medal eligibility uncertain. |

### Tier 3: Challenging but Rewarding

| Rank | Competition | Why |
|---|---|---|
| **7** | **NFL Big Data Bowl 2026 - Prediction** | Popular competition with many teams. Entry closed. Sports domain is accessible but competition is fierce. |
| **8** | **AIMO3 (AI Math Olympiad)** | Massive prize pool attracts top talent globally. 2000+ teams expected. Requires LLM fine-tuning expertise and compute resources. But bronze at top 10% of 2000 teams = top 200, which is achievable with effort. |

### Tier 4: Expert Level

| Rank | Competition | Why |
|---|---|---|
| **9** | **Gemini 3 Hackathon** | Medal eligibility for hackathon format uncertain. Very large field from Google brand recognition. Judged subjectively. |

---

## Recommended Strategy

### For Immediate Action (next 1-2 days):
1. **CAFA 6 Protein Function Prediction** -- Entry deadline is January 26. Join NOW if interested. Use ESM2 embeddings + gradient boosting as a baseline.

### For Short-term (next 2 weeks):
2. **Vesuvius Challenge - Surface Detection** -- Best medal odds of any active competition. Entry deadline February 6. Study prior Vesuvius Challenge solutions and 3D segmentation techniques.

### For Medium-term (1-3 months):
3. **Deep Past - Translate Akkadian** -- Entry deadline March 23. Start with LLM fine-tuning approaches for low-resource translation. Niche field = fewer competitors.
4. **AIMO3** -- Entry deadline April 8. Only if you have LLM fine-tuning experience and access to significant compute.

### General Tips for Medal Hunting:
- **Start with public notebooks** -- Kaggle's community shares starter code. Fork, understand, then improve.
- **Ensemble aggressively** -- Blending multiple models is the single most effective technique for climbing leaderboards.
- **Focus on validation** -- Build a robust local CV setup that correlates with the public leaderboard to avoid overfitting.
- **Late-join advantage** -- Joining a competition 2-3 weeks before the deadline lets you leverage all published insights while avoiding burnout.
- **Target small competitions** -- Competitions with fewer teams have proportionally wider medal bands (especially at the small tier where bronze = top 40%).

---

## Sources

- [Kaggle Competitions Page](https://www.kaggle.com/competitions)
- [Kaggle Competition Types Documentation](https://www.kaggle.com/docs/competitions)
- [Kaggle Competition Progression System](https://www.kaggle.com/progression/competitions)
- [Kaggle Official Twitter/X](https://x.com/kaggle)
- [Kagoole Competition Tracker (Twitter/X)](https://x.com/kagoole)
- [CompeteHub - January 2026 Competitions](https://www.competehub.dev/en/monthly/2026-01)
- [ML Contests Aggregator](https://mlcontests.com/)
- [DataCamp - Kaggle Competitions Complete Guide](https://www.datacamp.com/blog/kaggle-competitions-the-complete-guide)
- [CSIRO Biomass Competition Announcement](https://www.csiro.au/en/news/all/news/2025/october/kaggle-competition)
- [AIMO Prize Updates](https://aimoprize.com/updates/2025-11-19-third-progress-prize-launched)
- [Hull Tactical Competition Announcement (Yahoo Finance)](https://finance.yahoo.com/news/hull-tactical-launches-kaggle-competition-120000562.html)
- [Deep Past Initiative](https://www.deeppast.org/)
- [Vesuvius Challenge / Scroll Prize](https://scrollprize.substack.com/p/back-to-the-challenge-100k-kaggle)
- [NFL Big Data Bowl 2026](https://operations.nfl.com/gameday/analytics/big-data-bowl/)
- [OFC 2026 ML Challenge Submission Guidelines](https://www.ofcconference.org/submit-a-paper/machine-learning-submission/)
- [ARC Prize 2025 Results](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [Paul Timothy Mooney - List of Every Kaggle Competition](https://www.kaggle.com/code/paultimothymooney/list-of-every-kaggle-competition-updated-monthly)

---

*Note: Team counts are estimates based on available data. Kaggle's website uses client-side rendering which prevented direct scraping of real-time team counts. The Kaggle CLI was attempted but could not be executed in this session. For the most accurate current team counts, visit each competition's leaderboard page directly on kaggle.com.*

*Report generated on January 25, 2026.*
