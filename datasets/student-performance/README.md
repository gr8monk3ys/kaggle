# Student Performance: Study Habits & GPA (10K Students)

> 10K students, 25 features, with a built-in sleep optimum and tutoring curve

**License:** GPL-3.0

**Kaggle:** [lorenzoscaturchio/student-academic-performance-dataset](https://www.kaggle.com/datasets/lorenzoscaturchio/student-academic-performance-dataset)

## Description

10,000 student records across 25 columns: study habits, demographics, family background, school resources, four subject scores (math, reading, writing, science), an overall GPA, and a binary pass/fail label.

What makes this table worth more than five minutes is that the structure in it is deliberately non-linear, so a correlation heatmap will not find it. The companion notebook measures every claim below.

- sleep_hours vs GPA is an inverted U. A degree-2 fit peaks at 7.44 hours: mean GPA is 3.27 at the peak, 2.67 in the 4-5.5 h band, and 2.93 in the 9-10 h band, so under-sleeping costs about 1.75x what over-sleeping costs. Pearson's r is only +0.18, which is the point - adding one squared term takes R2 from 0.031 to 0.124.
- tutoring_sessions rises and then reverses. The fitted curve peaks near 9-10 sessions (the 6-10 band averages 3.19 GPA against a 3.03 zero-session baseline), then falls back until the 16-20 band is statistically indistinguishable from zero sessions (+0.018 GPA, 95% CI [-0.039, +0.074]). That is a reversal, not a plateau. Earlier versions of this description called it "diminishing returns", which was the wrong label.
- The socioeconomic gradient is monotonic. Mean GPA rises at every step from parental_education "none" (2.93) to "phd" (3.37), and from low income (2.93) to high (3.30). Behind the income gap is an access gap - home internet 66% to 98%, laptop 55% to 95% - so the table supports mediation analysis, not just group means.
- gender, ethnicity, school_region, sports_participation and extracurricular_activities carry no engineered effect: each moves mean GPA by under 0.05 points end to end. They are clean negative controls for fairness tooling, which also means this dataset cannot demonstrate real demographic bias, because none was written into it.

Built for: GPA regression, pass/fail classification (note the 96% pass rate - use balanced accuracy or PR-AUC, not plain accuracy), non-linear feature engineering, mediation analysis, feature-importance teaching, and calibrating a fairness-audit pipeline against a known-null answer.

All data is synthetic, produced by the seeded create_dataset.py that ships with the dataset. Everything above is a measured property of that generator's output, not a finding about real students. Subject scores are clipped at 100 and GPA at 4.0, so 11-16% of subject scores and 4.5% of GPAs sit exactly at the cap and every effect quoted here is a lower bound.

## Tags

`education`, `classification`, `regression`, `clustering`, `beginner`

## Authors

- **Lorenzo Scaturchio**: Independent ML engineer building synthetic, education-first datasets for reproducible benchmarking and prototyping.

## Coverage

- Temporal: 2022-01-01 to 2025-12-31
- Geospatial: Global (synthetic)

## DOI and Citations

- DOI: Not assigned
- Scaturchio, Lorenzo (2026). Student Performance: Study Habits & GPA (10K Students). Kaggle Dataset. https://www.kaggle.com/datasets/lorenzoscaturchio/student-academic-performance-dataset

## Provenance

- Source: Synthetic data generation scripts in this repository
- Source: Public domain schemas and domain conventions for educational simulation
- Collection methodology: Programmatic synthetic generation using seeded statistical distributions and rule-based constraints to mimic realistic structure while avoiding direct personal data.

## students.csv

**Rows:** 10,000  |  **Columns:** 25  |  **Size:** 1,143.8 KB

| Column | Type | Null% | Unique | Sample values |
|--------|------|-------|--------|---------------|
| `student_id` | string | 0.0% | 10,000 | `STU00000`, `STU00001`, `STU00002` |
| `age` | integer | 0.0% | 11 | `19`, `23`, `22` |
| `gender` | string | 0.0% | 3 | `F`, `M`, `Non-binary` |
| `ethnicity` | string | 0.0% | 5 | `C`, `D`, `B` |
| `parental_education` | string | 0.0% | 6 | `high_school`, `bachelor`, `some_college` |
| `family_income` | string | 0.0% | 3 | `middle`, `high`, `low` |
| `school_type` | string | 0.0% | 3 | `public`, `private`, `charter` |
| `school_region` | string | 0.0% | 3 | `urban`, `suburban`, `rural` |
| `study_hours_per_week` | float | 0.0% | 288 | `2.0`, `14.2`, `3.3` |
| `attendance_rate` | float | 0.0% | 397 | `91.7`, `76.8`, `83.1` |
| `extracurricular_activities` | integer | 0.0% | 6 | `3`, `5`, `2` |
| `sports_participation` | boolean | 0.0% | 2 | `no`, `yes` |
| `tutoring_sessions` | integer | 0.0% | 21 | `17`, `15`, `16` |
| `parental_involvement` | string | 0.0% | 3 | `medium`, `high`, `low` |
| `internet_access` | boolean | 0.0% | 2 | `yes`, `no` |
| `has_laptop` | boolean | 0.0% | 2 | `yes`, `no` |
| `sleep_hours` | float | 0.0% | 61 | `7.3`, `8.2`, `5.4` |
| `stress_level` | float | 0.0% | 91 | `3.3`, `4.9`, `3.5` |
| `motivation_score` | float | 0.0% | 91 | `6.4`, `1.8`, `5.8` |
| `reading_score` | float | 0.0% | 654 | `70.7`, `66.1`, `51.3` |
| `writing_score` | float | 0.0% | 654 | `58.3`, `83.9`, `71.8` |
| `math_score` | float | 0.0% | 624 | `100.0`, `88.9`, `94.5` |
| `science_score` | float | 0.0% | 635 | `74.5`, `83.6`, `71.2` |
| `overall_gpa` | float | 0.0% | 269 | `2.8`, `3.29`, `2.93` |
| `passed` | integer | 0.0% | 2 | `1`, `0` |

## Suggested Use Cases

- GPA regression (drop all four subject scores first - `overall_gpa` is derived from them)
- Pass/fail classification with balanced metrics (the label is 96% positive)
- Non-linear feature engineering: recovering the sleep optimum and the tutoring reversal
- Mediation analysis: family_income -> internet/laptop access -> GPA
- Calibrating a fairness-audit pipeline against columns with a known-null answer

---
*Generated by `dataset_optimizer.py` — dataset_optimizer.py*
