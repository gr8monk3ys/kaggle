#!/usr/bin/env python3
"""
Synthetic GitHub Repository Metrics Dataset Generator
======================================================
Generates realistic data for 5K+ GitHub repositories with correlated
features including stars, forks, issues, contributors, language trends,
and repository health indicators.

Usage:
    python create_dataset.py   # writes github_repos.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_REPOS = 5500
OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Language distribution (realistic based on GitHub trends)
# ---------------------------------------------------------------------------
LANGUAGES = {
    "Python":       {"frac": 0.18, "star_mult": 1.2, "topic_affinity": ["machine-learning", "data-science", "automation", "web", "api"]},
    "JavaScript":   {"frac": 0.16, "star_mult": 1.1, "topic_affinity": ["web", "frontend", "react", "nodejs", "typescript"]},
    "TypeScript":   {"frac": 0.10, "star_mult": 1.15, "topic_affinity": ["web", "frontend", "react", "angular", "api"]},
    "Java":         {"frac": 0.09, "star_mult": 0.9, "topic_affinity": ["enterprise", "android", "spring", "microservices", "backend"]},
    "Go":           {"frac": 0.07, "star_mult": 1.1, "topic_affinity": ["cloud", "devops", "cli", "kubernetes", "microservices"]},
    "Rust":         {"frac": 0.06, "star_mult": 1.3, "topic_affinity": ["systems", "cli", "performance", "wasm", "embedded"]},
    "C++":          {"frac": 0.05, "star_mult": 0.8, "topic_affinity": ["systems", "performance", "game-engine", "embedded", "graphics"]},
    "C#":           {"frac": 0.04, "star_mult": 0.85, "topic_affinity": ["dotnet", "game-engine", "enterprise", "unity", "desktop"]},
    "Ruby":         {"frac": 0.03, "star_mult": 0.9, "topic_affinity": ["web", "rails", "devops", "automation", "api"]},
    "PHP":          {"frac": 0.04, "star_mult": 0.7, "topic_affinity": ["web", "laravel", "wordpress", "cms", "backend"]},
    "Swift":        {"frac": 0.03, "star_mult": 0.95, "topic_affinity": ["ios", "macos", "mobile", "swiftui", "apple"]},
    "Kotlin":       {"frac": 0.03, "star_mult": 0.9, "topic_affinity": ["android", "mobile", "backend", "spring", "multiplatform"]},
    "Scala":        {"frac": 0.02, "star_mult": 0.75, "topic_affinity": ["data-science", "big-data", "spark", "functional", "jvm"]},
    "C":            {"frac": 0.03, "star_mult": 0.7, "topic_affinity": ["systems", "embedded", "linux", "networking", "performance"]},
    "Shell":        {"frac": 0.02, "star_mult": 0.6, "topic_affinity": ["devops", "automation", "linux", "cli", "docker"]},
    "Dart":         {"frac": 0.02, "star_mult": 0.95, "topic_affinity": ["flutter", "mobile", "cross-platform", "ui", "frontend"]},
    "Julia":        {"frac": 0.01, "star_mult": 1.0, "topic_affinity": ["machine-learning", "data-science", "scientific-computing", "numerical", "optimization"]},
    "Lua":          {"frac": 0.01, "star_mult": 0.7, "topic_affinity": ["game-engine", "embedded", "neovim", "scripting", "modding"]},
    "R":            {"frac": 0.01, "star_mult": 0.6, "topic_affinity": ["data-science", "statistics", "visualization", "bioinformatics", "ggplot"]},
}

LICENSES = {
    "MIT":        0.35,
    "Apache-2.0": 0.20,
    "GPL-3.0":    0.10,
    "BSD-3":      0.08,
    "ISC":        0.05,
    "MPL-2.0":    0.03,
    "LGPL-3.0":   0.02,
    "Unlicense":  0.02,
    "AGPL-3.0":   0.02,
    "None":       0.13,
}

ALL_TOPICS = [
    "machine-learning", "data-science", "web", "api", "cli", "devops",
    "cloud", "kubernetes", "docker", "frontend", "backend", "mobile",
    "react", "nodejs", "typescript", "automation", "testing", "security",
    "database", "microservices", "serverless", "graphql", "rest-api",
    "deep-learning", "nlp", "computer-vision", "pytorch", "tensorflow",
    "transformers", "llm", "rag", "fine-tuning", "mlops", "data-pipeline",
    "etl", "streaming", "monitoring", "observability", "performance",
    "systems", "embedded", "game-engine", "graphics", "wasm",
    "cross-platform", "desktop", "ios", "android", "flutter",
    "rust", "go", "python", "javascript",
    "open-source", "education", "awesome-list", "hacktoberfest",
    "documentation", "framework", "library", "tool",
]

# Repo name parts for generation
NAME_PREFIXES = [
    "awesome", "go", "py", "react", "fast", "super", "mini", "micro",
    "turbo", "hyper", "nano", "ultra", "smart", "auto", "easy", "simple",
    "open", "free", "pro", "next", "meta", "deep", "quick", "magic",
]
NAME_ROOTS = [
    "flow", "hub", "lab", "kit", "forge", "lens", "sync", "dash",
    "scan", "bot", "pipe", "stack", "craft", "form", "graph", "link",
    "bench", "serve", "agent", "track", "guard", "pilot", "spark",
    "wave", "core", "edge", "storm", "vault", "bridge", "light",
]
NAME_SUFFIXES = ["", "-ai", "-ml", "-dev", "-io", "-js", "-rs", "-go", "-cli", "-app", "-api", "-ui"]

# Description templates
DESC_TEMPLATES = [
    "A {adj} {noun} for {use_case}",
    "{adj} {noun} built with {lang}",
    "Open-source {noun} for {use_case}",
    "Fast and lightweight {noun} for {use_case}",
    "{noun} framework for building {use_case}",
    "Production-ready {noun} for {use_case}",
    "Modern {noun} written in {lang}",
    "Batteries-included {noun} for {use_case}",
    "Scalable {noun} for {use_case}",
    "High-performance {noun} for {use_case}",
]
DESC_ADJS = ["blazing-fast", "production-grade", "lightweight", "extensible", "modular",
             "type-safe", "async", "distributed", "real-time", "cloud-native"]
DESC_NOUNS = ["framework", "library", "toolkit", "platform", "engine", "tool",
              "SDK", "CLI", "dashboard", "pipeline", "service", "server"]
DESC_USE_CASES = [
    "web applications", "data processing", "ML model deployment",
    "API development", "microservices", "data visualization",
    "log analysis", "container orchestration", "CI/CD pipelines",
    "real-time analytics", "message queuing", "file management",
    "authentication", "database management", "task scheduling",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_repo_name(i):
    """Generate a plausible GitHub repo name."""
    style = rng.integers(0, 4)
    if style == 0:
        return f"{rng.choice(NAME_PREFIXES)}-{rng.choice(NAME_ROOTS)}{rng.choice(NAME_SUFFIXES)}"
    elif style == 1:
        return f"{rng.choice(NAME_ROOTS)}{rng.choice(NAME_SUFFIXES)}"
    elif style == 2:
        return f"{rng.choice(NAME_PREFIXES)}{rng.choice(NAME_ROOTS)}"
    else:
        return f"{rng.choice(NAME_PREFIXES)}-{rng.choice(NAME_ROOTS)}-{rng.choice(NAME_ROOTS)}"


def generate_description(lang):
    """Generate a plausible repo description."""
    template = rng.choice(DESC_TEMPLATES)
    return template.format(
        adj=rng.choice(DESC_ADJS),
        noun=rng.choice(DESC_NOUNS),
        use_case=rng.choice(DESC_USE_CASES),
        lang=lang,
    )


# ---------------------------------------------------------------------------
# Generate dataset
# ---------------------------------------------------------------------------
def generate_repos():
    """Create the GitHub repos DataFrame with realistic correlations."""
    lang_names = list(LANGUAGES.keys())
    lang_fracs = [LANGUAGES[l]["frac"] for l in lang_names]

    # Assign languages
    languages = rng.choice(lang_names, size=N_REPOS, p=lang_fracs)

    # Repository age (days since creation)
    # Newer repos are more common
    ages_days = rng.exponential(scale=800, size=N_REPOS).astype(int).clip(7, 3650)
    created_dates = pd.Timestamp("2025-01-01") - pd.to_timedelta(ages_days, unit="D")

    # --- Stars (power-law distribution with language multiplier) ---
    # Most repos have few stars, few repos have many
    raw_stars = rng.pareto(a=1.2, size=N_REPOS) * 10
    star_mults = np.array([LANGUAGES[l]["star_mult"] for l in languages])
    # Older repos accumulate more stars
    age_factor = np.log1p(ages_days) / np.log1p(365)
    stars = (raw_stars * star_mults * age_factor).astype(int).clip(0, 200_000)

    # --- Forks (correlated with stars, ~10-30% of stars) ---
    fork_ratio = rng.uniform(0.05, 0.35, N_REPOS)
    forks = (stars * fork_ratio + rng.normal(0, 5, N_REPOS)).astype(int).clip(0, None)

    # --- Watchers (correlated with stars) ---
    watchers = (stars * rng.uniform(0.01, 0.15, N_REPOS) + rng.normal(0, 3, N_REPOS)).astype(int).clip(0, None)

    # --- Open issues (correlated with stars and age) ---
    issue_rate = rng.exponential(0.02, N_REPOS)
    open_issues = (stars * issue_rate + rng.exponential(5, N_REPOS)).astype(int).clip(0, None)

    # --- Closed issues (typically more than open) ---
    closed_issues = (open_issues * rng.uniform(1.5, 8.0, N_REPOS) + rng.exponential(10, N_REPOS)).astype(int)

    # --- Pull requests ---
    pr_rate = rng.uniform(0.1, 0.5, N_REPOS)
    open_prs = (open_issues * pr_rate).astype(int).clip(0, None)
    merged_prs = (closed_issues * rng.uniform(0.3, 0.8, N_REPOS)).astype(int)

    # --- Contributors (correlated with stars and age) ---
    contrib_base = np.log1p(stars) * rng.uniform(0.5, 3.0, N_REPOS)
    contributors = contrib_base.astype(int).clip(1, 5000)

    # --- Commits (correlated with contributors and age) ---
    commits = (contributors * rng.uniform(10, 200, N_REPOS) * np.sqrt(ages_days / 365)).astype(int).clip(1, None)

    # --- Releases ---
    releases = (ages_days / 365 * rng.uniform(0.5, 12, N_REPOS)).astype(int).clip(0, 200)

    # --- Code metrics ---
    readme_length = rng.lognormal(7, 1.2, N_REPOS).astype(int).clip(100, 50000)
    # Popular repos tend to have longer READMEs
    readme_length = (readme_length * (1 + np.log1p(stars) / 20)).astype(int)

    # CI/CD (more likely for popular repos)
    has_ci_prob = 0.3 + 0.4 * np.minimum(stars / 1000, 1)
    has_ci = (rng.random(N_REPOS) < has_ci_prob).astype(int)

    # Test coverage (only if has CI, correlated with maturity)
    test_coverage = np.where(
        has_ci,
        rng.beta(3, 2, N_REPOS) * 100,  # skewed toward higher coverage
        np.nan,
    )
    test_coverage = np.round(test_coverage, 1)

    # Code of conduct
    has_coc = (rng.random(N_REPOS) < (0.1 + 0.3 * np.minimum(stars / 500, 1))).astype(int)

    # Contributing guide
    has_contributing = (rng.random(N_REPOS) < (0.15 + 0.35 * np.minimum(stars / 500, 1))).astype(int)

    # Last commit date (more active repos have more recent commits)
    days_since_commit = rng.exponential(scale=60, size=N_REPOS).astype(int).clip(0, ages_days)
    # Popular repos are more active
    days_since_commit = (days_since_commit / (1 + np.log1p(stars) / 10)).astype(int).clip(0, None)
    last_commit_dates = pd.Timestamp("2025-01-01") - pd.to_timedelta(days_since_commit, unit="D")

    # Is archived (more likely for old repos with no recent commits)
    archive_prob = np.where(days_since_commit > 365, 0.3, 0.02)
    is_archived = (rng.random(N_REPOS) < archive_prob).astype(int)

    # Is fork
    is_fork = (rng.random(N_REPOS) < 0.15).astype(int)

    # Size (KB) - correlated with commits and language
    size_kb = (commits * rng.uniform(0.5, 5, N_REPOS) + rng.lognormal(8, 2, N_REPOS)).astype(int).clip(10, None)

    # License
    license_names = list(LICENSES.keys())
    license_probs = list(LICENSES.values())
    licenses = rng.choice(license_names, size=N_REPOS, p=license_probs)

    # Topics (2-5 topics per repo, biased by language)
    topics_list = []
    for i in range(N_REPOS):
        lang = languages[i]
        affinity = LANGUAGES[lang]["topic_affinity"]
        n_topics = rng.integers(2, 6)
        # Mix affinity topics with random topics
        n_affinity = min(n_topics, len(affinity), rng.integers(1, n_topics + 1))
        chosen = list(rng.choice(affinity, size=n_affinity, replace=False))
        remaining = [t for t in ALL_TOPICS if t not in chosen]
        if n_topics - n_affinity > 0:
            chosen += list(rng.choice(remaining, size=n_topics - n_affinity, replace=False))
        topics_list.append("|".join(chosen))

    # Repo names and descriptions
    repo_names = []
    seen_names = set()
    for i in range(N_REPOS):
        while True:
            name = generate_repo_name(i)
            if name not in seen_names:
                seen_names.add(name)
                repo_names.append(name)
                break

    descriptions = [generate_description(languages[i]) for i in range(N_REPOS)]

    # Default branch
    default_branch = rng.choice(["main", "master", "main", "main", "develop"], N_REPOS)

    # Has wiki, has pages, has discussions
    has_wiki = (rng.random(N_REPOS) < 0.4).astype(int)
    has_pages = (rng.random(N_REPOS) < 0.15).astype(int)
    has_discussions = (rng.random(N_REPOS) < (0.1 + 0.2 * np.minimum(stars / 1000, 1))).astype(int)

    df = pd.DataFrame({
        "repo_name": repo_names,
        "language": languages,
        "description": descriptions,
        "stars": stars,
        "forks": forks,
        "watchers": watchers,
        "open_issues": open_issues,
        "closed_issues": closed_issues,
        "open_pull_requests": open_prs,
        "merged_pull_requests": merged_prs,
        "contributors": contributors,
        "commits": commits,
        "releases": releases,
        "license": licenses,
        "topics": topics_list,
        "created_date": created_dates.strftime("%Y-%m-%d"),
        "last_commit_date": last_commit_dates.strftime("%Y-%m-%d"),
        "readme_length": readme_length,
        "has_ci": has_ci,
        "test_coverage": test_coverage,
        "has_code_of_conduct": has_coc,
        "has_contributing_guide": has_contributing,
        "has_wiki": has_wiki,
        "has_pages": has_pages,
        "has_discussions": has_discussions,
        "default_branch": default_branch,
        "is_archived": is_archived,
        "is_fork": is_fork,
        "size_kb": size_kb,
    })

    return df


def main():
    df = generate_repos()
    csv_path = OUTPUT_DIR / "github_repos.csv"
    df.to_csv(csv_path, index=False)

    print(f"Generated {len(df)} repositories")
    print(f"\nLanguage distribution (top 10):")
    print(df["language"].value_counts().head(10).to_string())
    print(f"\nStar statistics:")
    print(df["stars"].describe().to_string())
    print(f"\nLicense distribution:")
    print(df["license"].value_counts().to_string())
    print(f"\nSaved to: {csv_path}")


if __name__ == "__main__":
    main()
