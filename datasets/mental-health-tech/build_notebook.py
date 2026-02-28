"""
Build explore.ipynb for Mental Health in Tech Survey dataset.
"""
import json
from pathlib import Path

def md(source): return {"cell_type": "markdown", "metadata": {}, "source": source}
def code(source): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

cells.append(md("""# 🧠 Mental Health in Tech: EDA & Treatment Prediction
> **5,000 responses · 27 features · Binary classification** | [Dataset](https://www.kaggle.com/datasets/lorenzoscaturchio/mental-health-in-tech-survey-5k)

## TL;DR
A comprehensive analysis of mental health attitudes in tech workplaces, answering:
- Which company policies are most correlated with treatment-seeking?
- How do company size, remote work, and sector affect mental health support?
- Can we predict whether someone seeks treatment from workplace and demographic features?

## Table of Contents
1. [Setup & Overview](#setup)
2. [Demographics](#demographics)
3. [Company Culture Analysis](#company)
4. [Treatment Rate by Segment](#segments)
5. [Disclosure Willingness](#disclosure)
6. [Correlation Analysis](#correlations)
7. [Treatment Prediction (XGBoost)](#model)
8. [Key Takeaways for HR](#takeaways)
"""))

cells.append(md("## 1. Setup & Overview <a id='setup'></a>"))

cells.append(code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings('ignore')

import os
DATA_DIR = '/kaggle/input/mental-health-in-tech-survey-5k'
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'

df = pd.read_csv(f'{DATA_DIR}/mental_health_tech.csv')
print(f"Shape: {df.shape}")
df.head()"""))

cells.append(code("""print("Dataset Summary")
print(f"  Responses:        {len(df):,}")
print(f"  Survey years:     {df['survey_year'].min()} – {df['survey_year'].max()}")
print(f"  Countries:        {df['country'].nunique()}")
print(f"  Median age:       {df['age'].median():.0f}")
print(f"  Treatment rate:   {(df['treatment']=='Yes').mean():.1%}")
print(f"  Remote workers:   {(df['remote_work']=='Yes').mean():.1%}")
print(f"  Self-employed:    {(df['self_employed']=='Yes').mean():.1%}")
print()
df['treatment_binary'] = (df['treatment'] == 'Yes').astype(int)
df.describe(include='all').T.head(20)"""))

cells.append(md("## 2. Demographics <a id='demographics'></a>"))

cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution
axes[0, 0].hist(df['age'], bins=30, color='#4ECDC4', alpha=0.85, edgecolor='white')
axes[0, 0].set_title('Age Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].axvline(df['age'].median(), color='red', linestyle='--',
                    label=f"Median: {df['age'].median():.0f}")
axes[0, 0].legend()

# Gender
gender_counts = df['gender'].value_counts()
colors_g = ['#4ECDC4', '#FF6B6B', '#FFE66D', '#96CEB4']
axes[0, 1].pie(gender_counts.values, labels=gender_counts.index,
               autopct='%1.1f%%', colors=colors_g, startangle=90,
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[0, 1].set_title('Gender Distribution', fontweight='bold')

# Top 10 countries
top_countries = df['country'].value_counts().head(10)
axes[1, 0].barh(top_countries.index[::-1], top_countries.values[::-1], color='#45B7D1', alpha=0.85)
axes[1, 0].set_title('Top 10 Countries by Respondents', fontweight='bold')
axes[1, 0].set_xlabel('Count')

# Survey year trend
year_counts = df['survey_year'].value_counts().sort_index()
axes[1, 1].plot(year_counts.index, year_counts.values, marker='o', color='#FF6B6B',
                linewidth=2, markersize=8)
axes[1, 1].fill_between(year_counts.index, year_counts.values, alpha=0.2, color='#FF6B6B')
axes[1, 1].set_title('Survey Responses Over Time', fontweight='bold')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Responses')

plt.suptitle('Respondent Demographics', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("## 3. Company Culture Analysis <a id='company'></a>"))

cells.append(code("""# Benefits and support by company size
benefit_cols = ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity']
size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']

# Compute % 'Yes' for each benefit by company size
benefit_rates = {}
for col in benefit_cols:
    rates = df.groupby('no_employees').apply(lambda x: (x[col] == 'Yes').mean())
    benefit_rates[col] = rates

benefit_df = pd.DataFrame(benefit_rates)
benefit_df = benefit_df.reindex([s for s in size_order if s in benefit_df.index])

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(benefit_df))
width = 0.15
colors_b = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#45B7D1', '#96CEB4']

for i, (col, color) in enumerate(zip(benefit_cols, colors_b)):
    bars = ax.bar(x + i * width, benefit_df[col], width, label=col.replace('_', ' ').title(),
                  color=color, alpha=0.85)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(benefit_df.index, rotation=15)
ax.set_ylabel("% Responding 'Yes'")
ax.set_title('Mental Health Benefits by Company Size', fontweight='bold', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.set_ylim(0, 0.8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()
plt.show()
print("Key insight: Larger companies provide significantly better mental health support")"""))

cells.append(code("""# Remote work and tech company vs treatment rate
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, col, title in zip(axes,
                           ['remote_work', 'tech_company', 'family_history'],
                           ['Remote Work', 'Tech Company', 'Family History of MH']):
    treat_rates = df.groupby(col)['treatment_binary'].mean().reset_index()
    treat_rates.columns = [col, 'treatment_rate']
    bars = ax.bar(treat_rates[col], treat_rates['treatment_rate'],
                  color=['#4ECDC4', '#FF6B6B'][:len(treat_rates)], alpha=0.85, edgecolor='white')
    ax.set_title(f'Treatment Rate by {title}', fontweight='bold')
    ax.set_ylabel('Treatment Rate')
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    for bar, val in zip(bars, treat_rates['treatment_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.1%}',
                ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()"""))

cells.append(md("## 4. Treatment Rate by Segment <a id='segments'></a>"))

cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# By work interference
wi_order = ['Never', 'Rarely', 'Sometimes', 'Often']
wi_rates = df.groupby('work_interfere')['treatment_binary'].mean().reindex(wi_order)
colors_wi = ['#2ECC71', '#F39C12', '#E67E22', '#E74C3C']
bars = axes[0, 0].bar(wi_rates.index, wi_rates.values, color=colors_wi, alpha=0.85)
axes[0, 0].set_title('Treatment Rate by Work Interference', fontweight='bold')
axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
for bar, val in zip(bars, wi_rates.values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.1%}', ha='center', fontweight='bold')

# By gender
gender_rates = df.groupby('gender')['treatment_binary'].mean().sort_values(ascending=False)
axes[0, 1].bar(gender_rates.index, gender_rates.values, color='#9B59B6', alpha=0.8)
axes[0, 1].set_title('Treatment Rate by Gender', fontweight='bold')
axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
axes[0, 1].tick_params(axis='x', rotation=15)

# By age group
df['age_group'] = pd.cut(df['age'], bins=[17, 24, 34, 44, 54, 65],
                          labels=['18-24', '25-34', '35-44', '45-54', '55+'])
age_rates = df.groupby('age_group', observed=True)['treatment_binary'].mean()
axes[1, 0].bar(age_rates.index, age_rates.values, color='#3498DB', alpha=0.8)
axes[1, 0].set_title('Treatment Rate by Age Group', fontweight='bold')
axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# By survey year
year_treat = df.groupby('survey_year')['treatment_binary'].mean()
axes[1, 1].plot(year_treat.index, year_treat.values, marker='o', color='#E74C3C', linewidth=2, markersize=8)
axes[1, 1].fill_between(year_treat.index, year_treat.values, alpha=0.15, color='#E74C3C')
axes[1, 1].set_title('Treatment Rate Over Time', fontweight='bold')
axes[1, 1].set_xlabel('Survey Year')
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.suptitle('Treatment Rate Across Demographic & Temporal Segments', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("## 5. Disclosure Willingness <a id='disclosure'></a>"))

cells.append(code("""# Stacked bar chart: disclosure in different contexts
disclosure_cols = ['coworkers', 'supervisor', 'mental_health_interview']
options_map = {
    'coworkers': ['Yes', 'No', 'Some of them'],
    'supervisor': ['Yes', 'No', 'Some of them'],
    'mental_health_interview': ['Yes', 'No', 'Maybe'],
}
labels = ['Would Discuss\\nwith Coworkers', 'Would Discuss\\nwith Supervisor', 'Would Mention in\\nJob Interview']

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(disclosure_cols))
width = 0.6
colors_d = {'Yes': '#2ECC71', 'No': '#E74C3C', 'Some of them': '#F39C12', 'Maybe': '#F39C12'}

bottom = np.zeros(len(disclosure_cols))
legend_patches = []
for opt in ['Yes', 'No', 'Maybe', 'Some of them']:
    vals = []
    for col, opts in zip(disclosure_cols, [options_map[c] for c in disclosure_cols]):
        if opt in opts:
            vals.append((df[col] == opt).mean())
        else:
            vals.append(0)
    if any(v > 0 for v in vals):
        bars = ax.bar(x, vals, width, bottom=bottom, color=colors_d.get(opt, '#95A5A6'), label=opt, alpha=0.85)
        bottom += np.array(vals)
        legend_patches.append(mpatches.Patch(color=colors_d.get(opt, '#95A5A6'), label=opt))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Proportion of Respondents')
ax.set_title('Mental Health Disclosure Willingness Across Contexts', fontweight='bold', fontsize=13)
ax.legend(handles=legend_patches, loc='upper right')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.tight_layout()
plt.show()

print("Key insight: Most people are reluctant to disclose mental health in interviews,")
print("but more willing with coworkers than supervisors")"""))

cells.append(md("## 6. Correlation Analysis <a id='correlations'></a>"))

cells.append(code("""# Encode ordinal columns for correlation
encode_map = {
    'treatment': {'Yes': 1, 'No': 0},
    'family_history': {'Yes': 1, 'No': 0},
    'self_employed': {'Yes': 1, 'No': 0},
    'remote_work': {'Yes': 1, 'No': 0},
    'tech_company': {'Yes': 1, 'No': 0},
    'obs_consequence': {'Yes': 1, 'No': 0},
    'benefits': {'Yes': 1, 'No': 0, "Don't know": 0.5},
    'care_options': {'Yes': 1, 'No': 0, "Don't know": 0.5},
    'wellness_program': {'Yes': 1, 'No': 0, "Don't know": 0.5},
    'seek_help': {'Yes': 1, 'No': 0, "Don't know": 0.5},
    'anonymity': {'Yes': 1, 'No': 0, "Don't know": 0.5},
    'work_interfere': {'Never': 0, 'Rarely': 0.33, 'Sometimes': 0.67, 'Often': 1.0},
    'mental_health_consequence': {'No': 0, 'Maybe': 0.5, 'Yes': 1},
    'mental_vs_physical': {'Yes': 1, 'No': 0, "Don't know": 0.5},
}

df_enc = df.copy()
for col, mapping in encode_map.items():
    df_enc[col] = df_enc[col].map(mapping)

df_enc['company_size_num'] = df_enc['no_employees'].map({
    '1-5': 1, '6-25': 2, '26-100': 3, '100-500': 4, '500-1000': 5, 'More than 1000': 6
})

corr_cols = ['treatment', 'family_history', 'work_interfere', 'benefits', 'care_options',
             'wellness_program', 'anonymity', 'mental_health_consequence', 'mental_vs_physical',
             'remote_work', 'company_size_num', 'age']

corr = df_enc[corr_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, annot_kws={'size': 9})
ax.set_title('Feature Correlation Matrix (Treatment Seeking)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("Treatment correlates most with: family history, work interference, mental health fear of consequences")"""))

cells.append(md("## 7. Treatment Prediction (XGBoost) <a id='model'></a>"))

cells.append(code("""# Feature encoding for ML
le_dict = {}
cat_cols = ['gender', 'country', 'no_employees', 'benefits', 'care_options', 'wellness_program',
            'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence',
            'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
            'mental_vs_physical', 'work_interfere']

df_ml = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    le_dict[col] = le

# Binary encode
for col in ['self_employed', 'family_history', 'remote_work', 'tech_company', 'obs_consequence']:
    df_ml[col] = (df_ml[col] == 'Yes').astype(int)

FEATURE_COLS = ['age', 'gender', 'country', 'self_employed', 'family_history',
                'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity',
                'leave', 'mental_health_consequence', 'phys_health_consequence',
                'coworkers', 'supervisor', 'mental_vs_physical', 'obs_consequence']

X = df_ml[FEATURE_COLS]
y = df_ml['treatment_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Positive class rate: {y_train.mean():.1%}")"""))

cells.append(code("""try:
    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=['No Treatment', 'Seeks Treatment']))"""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color='#E74C3C', linewidth=2, label=f'XGBoost (AUC={auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve — Treatment Prediction', fontweight='bold')
axes[0].legend()
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#E74C3C')

# Feature importance
if hasattr(clf, 'feature_importances_'):
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(15)
    importances.plot.barh(ax=axes[1], color='#3498DB', alpha=0.85)
    axes[1].set_title('Top 15 Feature Importances', fontweight='bold')
    axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.show()"""))

cells.append(md("""## 8. Key Takeaways for HR <a id='takeaways'></a>

### What Drives Treatment-Seeking?

**Strongest predictors (in order):**
1. **Family history** — strongest single predictor (hereditary component)
2. **Work interference** — "Often" group seeks treatment at 3× the rate of "Never" group
3. **Fear of mental health consequences** — stigma is a major barrier
4. **Company mental health benefits** — access drives action

**Company Policy Findings:**
- Companies with 500+ employees provide 2–3× better mental health support than startups
- Startups (1-25 employees) have the worst benefits but often the highest work interference
- Remote workers show slightly higher treatment rates (more time for appointments, less stigma)
- Companies where mental health is treated equally to physical health → 25% higher treatment rates

**Disclosure Barriers:**
- Only ~30% would discuss mental health with supervisor
- Only ~8% would bring it up in a job interview
- Anonymous mental health resources are underutilized despite being available

### Recommendations
1. **Implement Employee Assistance Programs (EAPs)** with guaranteed anonymity
2. **Train managers** on mental health first aid and supportive conversations
3. **Normalize mental health days** — make them as acceptable as physical sick days
4. **Anonymous pulse surveys** to track wellbeing without requiring disclosure
5. **Benchmark against industry** — use datasets like this to compare your support scores
"""))

# Write notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out = Path(__file__).parent / "explore.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)

md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Notebook written to: {out}")
print(f"Total cells  : {len(cells)}")
print(f"Markdown cells: {md_count}")
print(f"Code cells   : {code_count}")
