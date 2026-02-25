#!/usr/bin/env python3
"""
Synthetic Job Postings NLP & Salary Prediction Dataset Generator
=================================================================
Generates 15,000 realistic job postings spanning 10 job titles, 8 industries,
and 5 experience levels with salary ranges, required skills, company context,
and natural-language job descriptions.

Notable correlations encoded:
  - Senior/Director roles earn more than Entry/Mid
  - Enterprise companies pay 20% more than the baseline; startups pay 15% less
  - Data Scientist and ML Engineer roles carry a 15% salary premium
  - Remote roles carry a 10% salary premium
  - Healthcare and Finance industries pay above average; Education/Media pay less
  - Experience level controls required-skills list length

Usage:
    python create_dataset.py   # writes job_postings.csv to the same directory
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
N_JOBS = 15_000

DATE_START = pd.Timestamp("2023-01-01")
DATE_END = pd.Timestamp("2025-12-31")

OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Domain data
# ---------------------------------------------------------------------------
JOB_TITLES = [
    "Data Scientist",
    "Software Engineer",
    "ML Engineer",
    "Data Engineer",
    "Product Manager",
    "UX Designer",
    "DevOps Engineer",
    "Data Analyst",
    "Research Scientist",
    "Backend Engineer",
]

INDUSTRIES = ["tech", "finance", "healthcare", "retail", "media", "education", "manufacturing", "government"]

COMPANY_SIZES = ["startup", "small", "medium", "large", "enterprise"]
COMPANY_SIZE_WEIGHTS = [0.15, 0.20, 0.25, 0.25, 0.15]

EXPERIENCE_LEVELS = ["entry", "mid", "senior", "lead", "director"]
EXPERIENCE_WEIGHTS = [0.20, 0.30, 0.28, 0.14, 0.08]

REMOTE_TYPES = ["onsite", "hybrid", "remote"]
REMOTE_WEIGHTS = [0.30, 0.40, 0.30]

EDUCATION_OPTIONS = ["none", "bachelor", "master", "phd"]

# Education distribution per experience level
EDUCATION_DIST = {
    "entry":    [0.05, 0.65, 0.25, 0.05],
    "mid":      [0.05, 0.55, 0.30, 0.10],
    "senior":   [0.05, 0.45, 0.35, 0.15],
    "lead":     [0.05, 0.40, 0.35, 0.20],
    "director": [0.05, 0.35, 0.35, 0.25],
}

# ---------------------------------------------------------------------------
# 200 fake companies across industries
# ---------------------------------------------------------------------------
COMPANIES = {
    "tech": [
        "Axiom Technologies", "Cloudify Labs", "NexGen Systems", "ByteForge Inc",
        "Aether Computing", "Luminary AI", "StackEdge Corp", "Polarity Networks",
        "Pinnacle Software", "Circadian Tech", "Quantum Leap Digital", "Orion Dev Studios",
        "Codex Innovations", "SynapticIO", "TerraBytes Inc", "NovaStream Tech",
        "DataBridge Systems", "VectorLogic", "Meridian Cloud", "Apex AI Labs",
        "Prism Solutions", "ZeroPoint Systems", "Helix Platforms", "Crestline Software",
    ],
    "finance": [
        "Meridian Capital Group", "Vanguard Analytics", "Pinnacle Financial", "Summit Wealth Partners",
        "Atlas Investment Group", "Ironclad Fintech", "Sterling Data Finance", "Harbor Capital",
        "Crestview Asset Management", "Acme Financial AI", "Keystone Payments", "Northgate Capital",
        "Eclipse Trading Systems", "Apex Risk Analytics", "FinEdge Corp", "Broadfield Investments",
        "Horizon Fintech", "Cerberus Analytics", "Lighthouse Capital", "Trident Finance",
        "Cobalt Banking Tech", "Zenith Asset Advisors", "Redwood Financial", "Clearwater Markets",
    ],
    "healthcare": [
        "Helix Health Systems", "MedVault Technologies", "CureAI Labs", "Asclepius Analytics",
        "Nexus Medical AI", "BioPath Solutions", "ClinIQ Technologies", "Genome Insight",
        "Pulse Health Data", "Vital Analytics Inc", "CareEdge Systems", "Medi-Stream Corp",
        "Omega Health Tech", "Synapse Medical", "MedBridge Analytics", "Aurora Health Data",
        "Heartcore Systems", "Apex Clinical AI", "LifeLine Analytics", "DataCure Corp",
        "Orion Health Labs", "Vertex Medical Tech", "Radiant Health AI", "ZenHealth Solutions",
    ],
    "retail": [
        "Apex Commerce Solutions", "MomentumRetail AI", "Spectrum Consumer Tech", "GreenShelf Analytics",
        "PricePoint Systems", "Compass Retail Data", "Grid Commerce Corp", "Mercury Retail AI",
        "Trilliant Consumer Insights", "SkyLine Shopping Tech", "Urban Commerce Labs", "TrueCart Solutions",
        "Catalyst Retail Analytics", "Ember Commerce", "Flare Consumer Data", "Vortex Retail Systems",
        "Nimbus Commerce AI", "ClearPath Retail", "Stellar Shopping Tech", "ProximaRetail Corp",
        "Outreach Commerce", "Gravity Consumer Analytics", "ShopEdge Technologies", "Nexus Retail Insights",
    ],
    "media": [
        "Luminary Media AI", "WaveForm Analytics", "Prism Content Labs", "Signal Media Corp",
        "Vistas Content Tech", "Echo Analytics Group", "Frequency Media Systems", "Radiant Broadcasting AI",
        "Spectrum Content Corp", "Horizon Media Labs", "Pulse Digital Media", "Canvas Content Solutions",
        "Starstream Analytics", "Frame Media Tech", "Clarity Content Systems", "Vivid Analytics Corp",
        "Conduit Media AI", "Aria Content Labs", "SoundWave Analytics", "Strata Media Solutions",
        "Beacon Digital Corp", "Praxis Media AI", "Vox Content Tech", "Blueprint Media Labs",
    ],
    "education": [
        "Akademia AI", "EduBridge Technologies", "Scholar Analytics", "LearnEdge Systems",
        "CampusAI Corp", "MindPath Learning Tech", "AcademIQ Analytics", "TeachFlow Solutions",
        "Beacon Learning Labs", "Insight Education AI", "Skillify Technologies", "Cognify Corp",
        "Pathway Learning Analytics", "Edu-Pulse Systems", "OpenMind Technologies", "SkillBridge AI",
        "Atlas Education Corp", "Vertex Learning Labs", "Enlighten Analytics", "Cascade Education Tech",
        "Turing Learning Systems", "Compass Education AI", "Apex Edtech", "Mosaic Learning Corp",
    ],
    "manufacturing": [
        "IronCore Analytics", "Fabricate AI", "Precision Data Systems", "Alloy Intelligence Corp",
        "Foundry Analytics Group", "MachineEdge AI", "Torque Data Systems", "MetaForge Analytics",
        "Steelpath Solutions", "Circadian Manufacturing AI", "ProCast Analytics", "Nexus Fabrication Tech",
        "Apex Industrial AI", "Delta Manufacturing Systems", "GearShift Analytics", "Tensile Data Corp",
        "Forge AI Solutions", "Catalyst Industrial Tech", "Vertex Manufacturing Analytics", "Iron Summit AI",
        "Calibrate Systems", "Precision Works Analytics", "Atlas Industrial Corp", "Crest Manufacturing AI",
    ],
    "government": [
        "Civic Analytics Corp", "PublicData Systems", "NationAI Labs", "GovEdge Solutions",
        "Infrastructure Analytics Group", "CivilIQ Technologies", "PolicyData Corp", "Apex Civic AI",
        "Meridian Public Tech", "Urban Analytics Group", "Compass Government AI", "Strata Policy Systems",
        "Atlas Civic Data", "Federal Analytics Corp", "Horizon Public Solutions", "DataGov Systems",
        "Insight Policy Tech", "Nexus Civic Analytics", "Civic Pulse Systems", "BenchmarkGov AI",
        "Pioneer Policy Corp", "Keystone Public Analytics", "Prism Government Tech", "Orion Civic Solutions",
    ],
}

# Flatten for lookup
ALL_COMPANIES_BY_INDUSTRY = COMPANIES  # industry -> list of company names

# ---------------------------------------------------------------------------
# US cities (50)
# ---------------------------------------------------------------------------
US_CITIES = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX", "Boston, MA",
    "Chicago, IL", "Los Angeles, CA", "Denver, CO", "Atlanta, GA", "Washington, DC",
    "Dallas, TX", "San Jose, CA", "Portland, OR", "Miami, FL", "Minneapolis, MN",
    "San Diego, CA", "Philadelphia, PA", "Phoenix, AZ", "Nashville, TN", "Detroit, MI",
    "Raleigh, NC", "Salt Lake City, UT", "Charlotte, NC", "Columbus, OH", "Indianapolis, IN",
    "Pittsburgh, PA", "Baltimore, MD", "Kansas City, MO", "Tampa, FL", "St. Louis, MO",
    "Sacramento, CA", "Oakland, CA", "Louisville, KY", "Richmond, VA", "Cincinnati, OH",
    "Orlando, FL", "San Antonio, TX", "Houston, TX", "Cleveland, OH", "Memphis, TN",
    "New Orleans, LA", "Buffalo, NY", "Hartford, CT", "Providence, RI", "Albany, NY",
    "Boise, ID", "Madison, WI", "Des Moines, IA", "Omaha, NE", "Albuquerque, NM",
]

# ---------------------------------------------------------------------------
# Skills pool by role
# ---------------------------------------------------------------------------
ROLE_SKILLS = {
    "Data Scientist": [
        "Python", "R", "SQL", "Machine Learning", "Statistics", "TensorFlow", "PyTorch",
        "Spark", "Tableau", "scikit-learn", "Deep Learning", "NLP", "A/B Testing", "Jupyter",
    ],
    "Software Engineer": [
        "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "Docker",
        "Kubernetes", "AWS", "REST APIs", "Git", "SQL", "Redis", "PostgreSQL",
    ],
    "ML Engineer": [
        "Python", "TensorFlow", "PyTorch", "MLflow", "Airflow", "Kubernetes", "Docker",
        "AWS SageMaker", "Spark", "Feature Stores", "CI/CD", "Databricks",
    ],
    "Data Engineer": [
        "Python", "SQL", "Spark", "Airflow", "dbt", "Kafka", "Snowflake", "BigQuery",
        "Redshift", "AWS", "Azure", "Terraform", "PostgreSQL",
    ],
    "Product Manager": [
        "Product Strategy", "Roadmapping", "SQL", "A/B Testing", "User Research",
        "Agile", "Stakeholder Management", "Data Analysis", "JIRA", "Figma",
    ],
    "UX Designer": [
        "Figma", "User Research", "Prototyping", "Wireframing", "Usability Testing",
        "Design Systems", "Sketch", "Adobe XD", "HTML/CSS",
    ],
    "DevOps Engineer": [
        "Kubernetes", "Docker", "Terraform", "AWS", "Azure", "GCP", "CI/CD",
        "Jenkins", "Prometheus", "Grafana", "Linux", "Python", "Ansible",
    ],
    "Data Analyst": [
        "SQL", "Python", "Tableau", "Power BI", "Excel", "Statistics",
        "Google Analytics", "A/B Testing", "Looker", "dbt",
    ],
    "Research Scientist": [
        "Python", "PyTorch", "TensorFlow", "Machine Learning", "Deep Learning",
        "Statistics", "LaTeX", "Experimentation", "Publication Record",
    ],
    "Backend Engineer": [
        "Python", "Java", "Go", "SQL", "PostgreSQL", "Redis", "Kafka",
        "Docker", "Kubernetes", "REST APIs", "gRPC", "AWS",
    ],
}

# ---------------------------------------------------------------------------
# Salary base ranges (min, max) per experience level
# ---------------------------------------------------------------------------
SALARY_BASE = {
    "entry":    (55_000,  90_000),
    "mid":      (85_000, 130_000),
    "senior":   (120_000, 180_000),
    "lead":     (150_000, 220_000),
    "director": (180_000, 300_000),
}

# Company-size salary multipliers
COMPANY_SIZE_MULT = {
    "startup":    0.85,
    "small":      0.92,
    "medium":     1.00,
    "large":      1.10,
    "enterprise": 1.20,
}

# Industry salary multipliers
INDUSTRY_MULT = {
    "tech":          1.10,
    "finance":       1.15,
    "healthcare":    1.08,
    "retail":        0.95,
    "media":         0.92,
    "education":     0.88,
    "manufacturing": 0.97,
    "government":    0.93,
}

# High-paying role multipliers
HIGH_PAY_ROLES = {"Data Scientist", "ML Engineer", "Research Scientist"}
HIGH_PAY_ROLE_MULT = 1.15

# Remote premium
REMOTE_MULT = 1.10

# ---------------------------------------------------------------------------
# Description templates
# ---------------------------------------------------------------------------
DESCRIPTION_TEMPLATES = [
    (
        "We are seeking a {exp_level} {title} to join our {industry} team at {company}. "
        "You will work with {skills_preview} to drive {goal}. "
        "{culture_value_sentence}"
    ),
    (
        "{company} is looking for a passionate {title} with expertise in {skills_preview}. "
        "The ideal candidate will have {exp_level}-level experience in the {industry} space. "
        "{goal_sentence}"
    ),
    (
        "Join {company} as a {exp_level} {title} and help us build {goal} using {skills_preview}. "
        "We value {culture_value} and foster an inclusive environment where everyone can thrive."
    ),
    (
        "As a {exp_level} {title} at {company}, you will leverage {skills_preview} to deliver "
        "impactful solutions in the {industry} industry. {goal_sentence} {culture_value_sentence}"
    ),
]

GOALS = [
    "scalable data pipelines",
    "next-generation ML systems",
    "product analytics infrastructure",
    "real-time recommendation systems",
    "fraud detection systems",
    "user-facing features",
    "cloud infrastructure",
    "data-driven insights",
    "high-availability microservices",
    "intelligent automation workflows",
    "customer personalization engines",
    "enterprise data platforms",
]

CULTURE_VALUES = [
    "ownership and autonomy",
    "data-driven decision making",
    "collaboration and transparency",
    "continuous learning",
    "technical excellence",
    "customer obsession",
    "diversity and inclusion",
    "fast iteration and experimentation",
    "psychological safety",
    "impact at scale",
]

GOAL_SENTENCE_TEMPLATES = [
    "This role will focus on delivering {goal} that measurably improves our {industry} outcomes.",
    "You will own the design and delivery of {goal} end to end.",
    "Your work will directly shape our approach to {goal} across a global user base.",
]

CULTURE_SENTENCE_TEMPLATES = [
    "We champion {culture_value} as a core part of how we operate.",
    "Our team is built on {culture_value} — you will feel it on day one.",
    "We deeply believe in {culture_value} at every level of the organization.",
]

# ---------------------------------------------------------------------------
# Days-to-fill distributions per experience level
# ---------------------------------------------------------------------------
DAYS_TO_FILL_PARAMS = {
    "entry":    (15, 45),
    "mid":      (20, 60),
    "senior":   (30, 90),
    "lead":     (45, 120),
    "director": (60, 180),
}

# Poisson lambda for applications per company size
APPLICATIONS_LAMBDA = {
    "startup":    25,
    "small":      45,
    "medium":     80,
    "large":      140,
    "enterprise": 220,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_dates(start: pd.Timestamp, end: pd.Timestamp, n: int) -> list:
    """Return n random date strings (YYYY-MM-DD) between start and end."""
    ts_start = int(start.timestamp())
    ts_end = int(end.timestamp())
    timestamps = rng.integers(ts_start, ts_end, size=n)
    return pd.to_datetime(timestamps, unit="s").strftime("%Y-%m-%d").tolist()


def pick_skills(title: str, exp_level: str) -> str:
    """Return a pipe-separated skills string appropriate for experience level."""
    pool = ROLE_SKILLS[title]
    skill_count_range = {
        "entry":    (2, 4),
        "mid":      (4, 6),
        "senior":   (6, 8),
        "lead":     (7, 9),
        "director": (6, 10),
    }
    lo, hi = skill_count_range[exp_level]
    n = int(rng.integers(lo, hi + 1))
    n = min(n, len(pool))
    chosen = rng.choice(pool, size=n, replace=False)
    return "|".join(chosen)


def compute_salary(title: str, exp_level: str, company_size: str,
                   industry: str, remote_type: str) -> tuple:
    """Return (salary_min, salary_max) after applying all multipliers."""
    base_lo, base_hi = SALARY_BASE[exp_level]

    # Draw a random salary_min inside the base range
    raw_min = int(rng.integers(base_lo, base_hi))
    # salary_max is 15-35% above salary_min
    spread_pct = rng.uniform(0.15, 0.35)
    raw_max = int(raw_min * (1 + spread_pct))

    mult = 1.0
    mult *= COMPANY_SIZE_MULT[company_size]
    mult *= INDUSTRY_MULT[industry]
    if title in HIGH_PAY_ROLES:
        mult *= HIGH_PAY_ROLE_MULT
    if remote_type == "remote":
        mult *= REMOTE_MULT

    salary_min = int(round(raw_min * mult / 1_000) * 1_000)
    salary_max = int(round(raw_max * mult / 1_000) * 1_000)
    return salary_min, salary_max


def build_description(title: str, exp_level: str, company: str,
                       industry: str, skills_str: str) -> str:
    """Generate a 2-4 sentence job description from templates."""
    template = rng.choice(DESCRIPTION_TEMPLATES)
    skills_list = skills_str.split("|")
    # Preview: first 3 skills joined naturally
    if len(skills_list) >= 3:
        skills_preview = ", ".join(skills_list[:2]) + f", and {skills_list[2]}"
    elif len(skills_list) == 2:
        skills_preview = f"{skills_list[0]} and {skills_list[1]}"
    else:
        skills_preview = skills_list[0]

    goal = str(rng.choice(GOALS))
    culture_value = str(rng.choice(CULTURE_VALUES))

    goal_sentence = str(rng.choice(GOAL_SENTENCE_TEMPLATES)).format(
        goal=goal, industry=industry
    )
    culture_value_sentence = str(rng.choice(CULTURE_SENTENCE_TEMPLATES)).format(
        culture_value=culture_value
    )

    description = template.format(
        exp_level=exp_level,
        title=title,
        company=company,
        industry=industry,
        skills_preview=skills_preview,
        goal=goal,
        culture_value=culture_value,
        goal_sentence=goal_sentence,
        culture_value_sentence=culture_value_sentence,
    )
    return description.strip()


def pick_education(exp_level: str) -> str:
    return str(rng.choice(EDUCATION_OPTIONS, p=EDUCATION_DIST[exp_level]))


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_job_postings() -> pd.DataFrame:
    print("Sampling job attributes...")

    # Core categorical fields
    titles = rng.choice(JOB_TITLES, size=N_JOBS)
    industries = rng.choice(INDUSTRIES, size=N_JOBS)
    company_sizes = rng.choice(COMPANY_SIZES, size=N_JOBS, p=COMPANY_SIZE_WEIGHTS)
    exp_levels = rng.choice(EXPERIENCE_LEVELS, size=N_JOBS, p=EXPERIENCE_WEIGHTS)
    remote_types = rng.choice(REMOTE_TYPES, size=N_JOBS, p=REMOTE_WEIGHTS)
    locations = rng.choice(US_CITIES, size=N_JOBS)
    posted_dates = random_dates(DATE_START, DATE_END, N_JOBS)

    # Company names (sampled from industry bucket)
    companies = []
    for ind in industries:
        bucket = ALL_COMPANIES_BY_INDUSTRY[ind]
        companies.append(str(rng.choice(bucket)))

    print("Computing salaries...")
    salary_mins = []
    salary_maxs = []
    for i in range(N_JOBS):
        s_min, s_max = compute_salary(
            str(titles[i]), str(exp_levels[i]), str(company_sizes[i]),
            str(industries[i]), str(remote_types[i])
        )
        salary_mins.append(s_min)
        salary_maxs.append(s_max)

    print("Generating required skills...")
    required_skills_list = [
        pick_skills(str(titles[i]), str(exp_levels[i])) for i in range(N_JOBS)
    ]

    print("Generating education requirements...")
    education_list = [pick_education(str(exp_levels[i])) for i in range(N_JOBS)]

    print("Generating job descriptions...")
    descriptions = [
        build_description(
            str(titles[i]), str(exp_levels[i]), companies[i],
            str(industries[i]), required_skills_list[i]
        )
        for i in range(N_JOBS)
    ]

    print("Computing applications and days-to-fill...")
    applications = []
    days_to_fill = []
    for i in range(N_JOBS):
        lam = APPLICATIONS_LAMBDA[str(company_sizes[i])]
        applications.append(int(rng.poisson(lam)))

        lo, hi = DAYS_TO_FILL_PARAMS[str(exp_levels[i])]
        days_to_fill.append(int(rng.integers(lo, hi + 1)))

    print("Assembling DataFrame...")
    df = pd.DataFrame({
        "job_id":            [f"JOB{str(i).zfill(5)}" for i in range(N_JOBS)],
        "title":             titles,
        "company":           companies,
        "location":          locations,
        "remote_type":       remote_types,
        "industry":          industries,
        "company_size":      company_sizes,
        "experience_level":  exp_levels,
        "salary_min":        salary_mins,
        "salary_max":        salary_maxs,
        "required_skills":   required_skills_list,
        "education_required": education_list,
        "description":       descriptions,
        "posted_date":       posted_dates,
        "applications":      applications,
        "days_to_fill":      days_to_fill,
    })

    # Sort chronologically
    df = df.sort_values("posted_date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total job postings : {len(df):>8,}")
    print(f"  Columns            : {df.shape[1]}")
    print(f"  Date range         : {df['posted_date'].min()} to {df['posted_date'].max()}")
    print(f"  Salary range       : ${df['salary_min'].min():,.0f} — ${df['salary_max'].max():,.0f}")
    print()
    print("Title distribution:")
    for title, cnt in df["title"].value_counts().items():
        print(f"  {title:<22s}: {cnt:>5,} ({cnt/len(df)*100:.1f}%)")
    print()
    print("Experience level distribution:")
    for lvl, cnt in df["experience_level"].value_counts().items():
        print(f"  {lvl:<12s}: {cnt:>5,} ({cnt/len(df)*100:.1f}%)")
    print()
    print("Remote type distribution:")
    for rt, cnt in df["remote_type"].value_counts().items():
        print(f"  {rt:<10s}: {cnt:>5,} ({cnt/len(df)*100:.1f}%)")
    print()
    print("Median salary_min by experience level:")
    medians = df.groupby("experience_level")["salary_min"].median()
    for lvl in EXPERIENCE_LEVELS:
        if lvl in medians.index:
            print(f"  {lvl:<12s}: ${medians[lvl]:>9,.0f}")
    print()
    print("Median salary_min by company size:")
    for size in COMPANY_SIZES:
        med = df[df["company_size"] == size]["salary_min"].median()
        print(f"  {size:<12s}: ${med:>9,.0f}")
    print()
    print(f"  Missing values     : {df.isnull().sum().sum()}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating job postings dataset (15,000 rows)...")
    df = generate_job_postings()

    output_path = OUTPUT_DIR / "job_postings.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    print_summary(df)


if __name__ == "__main__":
    main()
