#!/usr/bin/env python3
"""Generate cover images for all datasets using actual data.

Each cover image is a visually appealing chart saved as cover.png in the dataset dir.
Upload these to Kaggle dataset Settings to boost usability scores.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

DATASETS_DIR = Path(__file__).parent / "datasets"
FIG_SIZE = (12, 7)
DPI = 150
BG_COLOR = "#0f1116"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ba68c8",
                 "#4dd0e1", "#aed581", "#ff8a65", "#f06292", "#7986cb"]


def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333")


def save_fig(fig, path):
    fig.patch.set_facecolor(BG_COLOR)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {path}")


def gen_ai_research(data_dir):
    df = pd.read_csv(data_dir / "ai_research_papers.csv")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if "year" in df.columns:
        yearly = df.groupby("year").size()
        ax.fill_between(yearly.index, yearly.values, alpha=0.3, color=ACCENT_COLORS[0])
        ax.plot(yearly.index, yearly.values, color=ACCENT_COLORS[0], linewidth=2.5)
        style_ax(ax, "AI/ML Research Papers Published Per Year",
                 "Year", "Number of Papers")
    elif "category" in df.columns:
        top = df["category"].value_counts().head(10)
        ax.barh(top.index[::-1], top.values[::-1], color=ACCENT_COLORS[:len(top)])
        style_ax(ax, "AI/ML Research Papers by Category", "Count", "")
    save_fig(fig, data_dir / "cover.png")


def gen_credit_card_fraud(data_dir):
    df = pd.read_csv(data_dir / "credit_card_transactions.csv", nrows=50000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE, gridspec_kw={"width_ratios": [1, 2]})
    fraud_col = next((c for c in df.columns if "fraud" in c.lower() or "class" in c.lower()), None)
    if fraud_col:
        counts = df[fraud_col].value_counts()
        labels = ["Legitimate", "Fraud"] if len(counts) == 2 else [str(x) for x in counts.index]
        colors_pie = [ACCENT_COLORS[0], ACCENT_COLORS[3]]
        ax1.pie(counts.values, labels=labels, colors=colors_pie[:len(counts)],
                autopct="%1.1f%%", textprops={"color": TEXT_COLOR, "fontsize": 10},
                startangle=90, explode=[0, 0.1] if len(counts) == 2 else None)
        style_ax(ax1, "Class Distribution")
    amount_col = next((c for c in df.columns if "amount" in c.lower()), None)
    if amount_col and fraud_col:
        legit = df[df[fraud_col] == 0][amount_col].dropna()
        fraud = df[df[fraud_col] == 1][amount_col].dropna()
        bins = np.linspace(0, min(legit.quantile(0.99), 5000), 50)
        ax2.hist(legit, bins=bins, alpha=0.6, color=ACCENT_COLORS[0], label="Legitimate", density=True)
        ax2.hist(fraud, bins=bins, alpha=0.6, color=ACCENT_COLORS[3], label="Fraud", density=True)
        ax2.legend(facecolor=BG_COLOR, edgecolor="#333", labelcolor=TEXT_COLOR)
        style_ax(ax2, "Transaction Amount Distribution", "Amount ($)", "Density")
    fig.suptitle("Credit Card Fraud Detection Dataset", color=TEXT_COLOR, fontsize=18, fontweight="bold", y=1.02)
    save_fig(fig, data_dir / "cover.png")


def gen_ecommerce(data_dir):
    df = pd.read_csv(data_dir / "customers.csv")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    spend_col = next((c for c in df.columns if "spend" in c.lower() or "total" in c.lower() or "revenue" in c.lower()), None)
    seg_col = next((c for c in df.columns if "segment" in c.lower() or "cluster" in c.lower() or "category" in c.lower()), None)
    if spend_col and seg_col:
        grouped = df.groupby(seg_col)[spend_col].mean().sort_values(ascending=True)
        bars = ax.barh(grouped.index, grouped.values, color=ACCENT_COLORS[:len(grouped)])
        style_ax(ax, "Average Customer Spend by Segment", "Average Spend ($)", "")
    elif spend_col:
        ax.hist(df[spend_col].dropna(), bins=40, color=ACCENT_COLORS[0], alpha=0.7, edgecolor="#333")
        style_ax(ax, "Customer Spend Distribution", "Total Spend ($)", "Count")
    else:
        top_cols = [c for c in df.select_dtypes(include=["object"]).columns]
        if top_cols:
            counts = df[top_cols[0]].value_counts().head(8)
            ax.barh(counts.index[::-1], counts.values[::-1], color=ACCENT_COLORS[:len(counts)])
            style_ax(ax, f"Distribution of {top_cols[0]}", "Count", "")
    save_fig(fig, data_dir / "cover.png")


def gen_github(data_dir):
    df = pd.read_csv(data_dir / "github_repos.csv")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    stars_col = next((c for c in df.columns if "star" in c.lower()), None)
    forks_col = next((c for c in df.columns if "fork" in c.lower()), None)
    lang_col = next((c for c in df.columns if "lang" in c.lower()), None)
    if stars_col and forks_col and lang_col:
        top_langs = df[lang_col].value_counts().head(8).index
        for i, lang in enumerate(top_langs):
            subset = df[df[lang_col] == lang]
            ax.scatter(subset[stars_col], subset[forks_col],
                       alpha=0.5, s=20, color=ACCENT_COLORS[i % len(ACCENT_COLORS)], label=lang)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(facecolor=BG_COLOR, edgecolor="#333", labelcolor=TEXT_COLOR, fontsize=8, loc="upper left")
        style_ax(ax, "GitHub Repos: Stars vs Forks by Language", "Stars (log)", "Forks (log)")
    elif stars_col:
        ax.hist(df[stars_col].clip(upper=df[stars_col].quantile(0.95)), bins=50,
                color=ACCENT_COLORS[0], alpha=0.7, edgecolor="#333")
        style_ax(ax, "GitHub Repository Stars Distribution", "Stars", "Count")
    save_fig(fig, data_dir / "cover.png")


def gen_job_postings(data_dir):
    df = pd.read_csv(data_dir / "job_postings.csv")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    salary_col = next((c for c in df.columns if "salary" in c.lower() and ("max" in c.lower() or "avg" in c.lower() or "mid" in c.lower())), None)
    if not salary_col:
        salary_col = next((c for c in df.columns if "salary" in c.lower()), None)
    title_col = next((c for c in df.columns if "title" in c.lower() or "role" in c.lower()), None)
    if salary_col and title_col:
        avg_salary = df.groupby(title_col)[salary_col].median().dropna().sort_values(ascending=True).tail(12)
        colors = [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(len(avg_salary))]
        ax.barh(avg_salary.index, avg_salary.values, color=colors)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x/1000:.0f}K"))
        style_ax(ax, "Median Salary by Job Title (Top 12)", "Salary", "")
    elif salary_col:
        ax.hist(df[salary_col].dropna(), bins=40, color=ACCENT_COLORS[0], alpha=0.7, edgecolor="#333")
        style_ax(ax, "Salary Distribution", "Salary ($)", "Count")
    save_fig(fig, data_dir / "cover.png")


def gen_mental_health(data_dir):
    df = pd.read_csv(data_dir / "mental_health_tech.csv")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    treat_col = next((c for c in df.columns if "treatment" in c.lower() or "sought" in c.lower()), None)
    age_col = next((c for c in df.columns if "age" in c.lower()), None)
    if treat_col and age_col:
        df["_treat_num"] = df[treat_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(float)
        bins = [18, 25, 30, 35, 40, 45, 50, 60, 70]
        df["age_group"] = pd.cut(df[age_col].clip(18, 70), bins=bins)
        rates = df.groupby("age_group", observed=True)["_treat_num"].mean() * 100
        ax.bar([str(x) for x in rates.index], rates.values, color=ACCENT_COLORS[:len(rates)], alpha=0.8)
        style_ax(ax, "Mental Health Treatment Rates by Age Group",
                 "Age Group", "Sought Treatment (%)")
    else:
        cols = [c for c in df.select_dtypes(include=["object"]).columns if df[c].nunique() < 10]
        if cols:
            counts = df[cols[0]].value_counts().head(8)
            ax.barh(counts.index[::-1], counts.values[::-1], color=ACCENT_COLORS[:len(counts)])
            style_ax(ax, f"Distribution of {cols[0]}", "Count", "")
    save_fig(fig, data_dir / "cover.png")


def gen_ml_interview(data_dir):
    df = pd.read_csv(data_dir / "ml_interview_questions.csv")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    cat_col = next((c for c in df.columns if "category" in c.lower() or "topic" in c.lower()), None)
    diff_col = next((c for c in df.columns if "difficulty" in c.lower() or "level" in c.lower()), None)
    if cat_col and diff_col:
        ct = pd.crosstab(df[cat_col], df[diff_col])
        ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).tail(10).index]
        ct.plot(kind="barh", stacked=True, ax=ax, color=ACCENT_COLORS[:ct.shape[1]])
        ax.legend(facecolor=BG_COLOR, edgecolor="#333", labelcolor=TEXT_COLOR, fontsize=9)
        style_ax(ax, "ML Interview Questions: Category x Difficulty", "Count", "")
    elif cat_col:
        counts = df[cat_col].value_counts().head(10)
        ax.barh(counts.index[::-1], counts.values[::-1], color=ACCENT_COLORS[:len(counts)])
        style_ax(ax, "ML Interview Questions by Category", "Count", "")
    save_fig(fig, data_dir / "cover.png")


def gen_programming_benchmarks(data_dir):
    df = pd.read_csv(data_dir / "language_benchmarks.csv")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    lang_col = next((c for c in df.columns if "lang" in c.lower()), None)
    time_col = next((c for c in df.columns if "time" in c.lower() or "speed" in c.lower() or "perf" in c.lower()), None)
    if lang_col and time_col:
        median_perf = df.groupby(lang_col)[time_col].median().sort_values().head(12)
        colors = [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(len(median_perf))]
        ax.barh(median_perf.index, median_perf.values, color=colors)
        style_ax(ax, "Median Benchmark Time by Language (Top 12)", "Time (s)", "")
    elif lang_col:
        counts = df[lang_col].value_counts().head(12)
        ax.barh(counts.index[::-1], counts.values[::-1], color=ACCENT_COLORS[:len(counts)])
        style_ax(ax, "Benchmarks by Programming Language", "Count", "")
    save_fig(fig, data_dir / "cover.png")


def gen_spotify(data_dir):
    df = pd.read_csv(data_dir / "spotify_tracks.csv", nrows=50000)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    energy_col = next((c for c in df.columns if "energy" in c.lower()), None)
    dance_col = next((c for c in df.columns if "dance" in c.lower()), None)
    pop_col = next((c for c in df.columns if "popular" in c.lower()), None)
    if energy_col and dance_col and pop_col:
        filtered = df.dropna(subset=[energy_col, dance_col, pop_col])
        if not filtered.empty:
            sample = filtered.sample(min(2000, len(filtered)))
            scatter = ax.scatter(sample[energy_col], sample[dance_col],
                                 c=sample[pop_col], cmap="coolwarm", alpha=0.5, s=15)
            fig.colorbar(scatter, ax=ax, label="Popularity")
            style_ax(ax, "Spotify Tracks: Energy vs Danceability",
                     "Energy", "Danceability")
    elif energy_col:
        ax.hist(df[energy_col].dropna(), bins=50, color=ACCENT_COLORS[1], alpha=0.7, edgecolor="#333")
        style_ax(ax, "Spotify Track Energy Distribution", "Energy", "Count")
    save_fig(fig, data_dir / "cover.png")


def gen_student_performance(data_dir):
    df = pd.read_csv(data_dir / "students.csv")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    gpa_col = next((c for c in df.columns if "gpa" in c.lower() or "grade" in c.lower() or "score" in c.lower()), None)
    study_col = next((c for c in df.columns if "study" in c.lower() or "hours" in c.lower()), None)
    if gpa_col and study_col:
        filtered = df.dropna(subset=[gpa_col, study_col])
        if not filtered.empty:
            sample = filtered.sample(min(2000, len(filtered)))
            ax.scatter(sample[study_col], sample[gpa_col], alpha=0.3, s=15, color=ACCENT_COLORS[0])
            z = np.polyfit(sample[study_col], sample[gpa_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sample[study_col].min(), sample[study_col].max(), 100)
            ax.plot(x_line, p(x_line), color=ACCENT_COLORS[3], linewidth=2.5, linestyle="--")
            style_ax(ax, "Study Hours vs GPA (with Trend Line)",
                     "Study Hours per Week", "GPA")
    elif gpa_col:
        ax.hist(df[gpa_col].dropna(), bins=40, color=ACCENT_COLORS[0], alpha=0.7, edgecolor="#333")
        style_ax(ax, "GPA Distribution", "GPA", "Count")
    save_fig(fig, data_dir / "cover.png")


GENERATORS = {
    "ai-research-trends": gen_ai_research,
    "credit-card-fraud": gen_credit_card_fraud,
    "ecommerce-behavior": gen_ecommerce,
    "github-repo-metrics": gen_github,
    "job-postings": gen_job_postings,
    "mental-health-tech": gen_mental_health,
    "ml-interview-qa": gen_ml_interview,
    "programming-benchmarks": gen_programming_benchmarks,
    "spotify-tracks": gen_spotify,
    "student-performance": gen_student_performance,
}


def main():
    success = 0
    errors = []
    for name, gen_func in GENERATORS.items():
        data_dir = DATASETS_DIR / name
        if not data_dir.exists():
            print(f"  SKIP: {name} (directory not found)")
            continue
        print(f"Generating cover for: {name}")
        try:
            gen_func(data_dir)
            success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((name, str(e)))

    print(f"\nDone: {success} covers generated, {len(errors)} errors")
    if errors:
        for name, err in errors:
            print(f"  FAILED: {name} — {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
