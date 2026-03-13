#!/usr/bin/env python3
"""
AI & Data Jobs Market Dataset Generator
======================================

Builds a current, multi-table synthetic jobs dataset for AI and data roles.

Outputs:
  - jobs.csv
  - companies.csv
  - job_skills.csv
  - salary_benchmarks.csv
  - skill_demand_monthly.csv

The generator is intentionally opinionated:
  - GenAI / LLM roles become more common in 2025-2026
  - Salary is driven by role, seniority, company size, region, remote policy,
    and industry
  - Startups offer equity more often but lower cash compensation
  - Enterprise and finance roles carry higher bonus targets
  - Skills are exposed both as pipe-separated strings and as a normalized table
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
rng = np.random.default_rng(SEED)
OUTPUT_DIR = Path(__file__).resolve().parent

N_COMPANIES = 280
N_JOBS = 36_000
DATE_START = pd.Timestamp("2024-01-01")
DATE_END = pd.Timestamp("2026-03-31")


LOCATIONS = [
    {"city": "San Francisco, CA", "country": "United States", "region": "North America", "salary_mult": 1.28},
    {"city": "New York, NY", "country": "United States", "region": "North America", "salary_mult": 1.24},
    {"city": "Seattle, WA", "country": "United States", "region": "North America", "salary_mult": 1.18},
    {"city": "Austin, TX", "country": "United States", "region": "North America", "salary_mult": 1.07},
    {"city": "Boston, MA", "country": "United States", "region": "North America", "salary_mult": 1.16},
    {"city": "Toronto, ON", "country": "Canada", "region": "North America", "salary_mult": 0.92},
    {"city": "Vancouver, BC", "country": "Canada", "region": "North America", "salary_mult": 0.95},
    {"city": "London, UK", "country": "United Kingdom", "region": "Europe", "salary_mult": 1.04},
    {"city": "Dublin, Ireland", "country": "Ireland", "region": "Europe", "salary_mult": 1.00},
    {"city": "Berlin, Germany", "country": "Germany", "region": "Europe", "salary_mult": 0.97},
    {"city": "Amsterdam, Netherlands", "country": "Netherlands", "region": "Europe", "salary_mult": 1.00},
    {"city": "Paris, France", "country": "France", "region": "Europe", "salary_mult": 0.96},
    {"city": "Zurich, Switzerland", "country": "Switzerland", "region": "Europe", "salary_mult": 1.18},
    {"city": "Bangalore, India", "country": "India", "region": "APAC", "salary_mult": 0.50},
    {"city": "Hyderabad, India", "country": "India", "region": "APAC", "salary_mult": 0.47},
    {"city": "Singapore", "country": "Singapore", "region": "APAC", "salary_mult": 0.98},
    {"city": "Sydney, Australia", "country": "Australia", "region": "APAC", "salary_mult": 0.96},
    {"city": "Melbourne, Australia", "country": "Australia", "region": "APAC", "salary_mult": 0.93},
]

LOCATION_WEIGHTS = np.array(
    [11, 10, 8, 7, 6, 4, 3, 8, 3, 4, 4, 3, 2, 8, 5, 5, 3, 2],
    dtype=float,
)
LOCATION_WEIGHTS /= LOCATION_WEIGHTS.sum()

INDUSTRIES = {
    "software": {"weight": 0.22, "salary_mult": 1.10, "company_types": ["product", "platform", "developer-tools"]},
    "finance": {"weight": 0.14, "salary_mult": 1.16, "company_types": ["fintech", "bank", "asset-manager"]},
    "healthcare": {"weight": 0.10, "salary_mult": 1.05, "company_types": ["healthtech", "provider", "biotech"]},
    "cloud": {"weight": 0.10, "salary_mult": 1.11, "company_types": ["cloud-platform", "infra", "security"]},
    "retail": {"weight": 0.10, "salary_mult": 0.95, "company_types": ["marketplace", "commerce", "consumer"]},
    "consulting": {"weight": 0.10, "salary_mult": 1.02, "company_types": ["consulting", "services", "advisory"]},
    "media": {"weight": 0.08, "salary_mult": 0.93, "company_types": ["media", "streaming", "adtech"]},
    "public-sector": {"weight": 0.06, "salary_mult": 0.92, "company_types": ["public-sector", "research-consortium"]},
    "education": {"weight": 0.05, "salary_mult": 0.90, "company_types": ["edtech", "learning-platform"]},
    "manufacturing": {"weight": 0.05, "salary_mult": 0.97, "company_types": ["industrial", "automation"]},
}

COMPANY_SIZE_BANDS = {
    "startup": (15, 80),
    "small": (80, 250),
    "medium": (250, 1_000),
    "large": (1_000, 5_000),
    "enterprise": (5_000, 40_000),
}
COMPANY_SIZE_WEIGHTS = np.array([0.18, 0.24, 0.24, 0.20, 0.14], dtype=float)
COMPANY_SIZE_WEIGHTS /= COMPANY_SIZE_WEIGHTS.sum()
COMPANY_SIZE_MULT = {
    "startup": 0.92,
    "small": 0.97,
    "medium": 1.00,
    "large": 1.08,
    "enterprise": 1.14,
}

FUNDING_STAGE_BY_SIZE = {
    "startup": ["seed", "series-a", "series-b"],
    "small": ["series-a", "series-b", "series-c"],
    "medium": ["series-c", "series-d", "late-stage"],
    "large": ["public", "late-stage", "subsidiary"],
    "enterprise": ["public", "public", "subsidiary"],
}

ROLE_CONFIGS = {
    "Data Scientist": {
        "weight": 0.14,
        "family": "modeling",
        "focuses": ["experimentation", "forecasting", "recommendation", "classification", "pricing"],
        "salary_mult": 1.05,
        "bonus_base": 0.10,
        "skills": [
            "Python", "SQL", "Statistics", "scikit-learn", "Pandas", "A/B Testing", "XGBoost",
            "Feature Engineering", "Experiment Design", "Jupyter",
        ],
    },
    "ML Engineer": {
        "weight": 0.12,
        "family": "ml-engineering",
        "focuses": ["training-platforms", "recommenders", "ranking", "serving", "feature-stores"],
        "salary_mult": 1.10,
        "bonus_base": 0.10,
        "skills": [
            "Python", "PyTorch", "TensorFlow", "Docker", "Kubernetes", "MLflow", "Airflow",
            "Feature Store", "CI/CD", "Spark",
        ],
    },
    "AI Engineer": {
        "weight": 0.08,
        "family": "genai-apps",
        "focuses": ["rag", "agents", "copilots", "knowledge-search", "automation"],
        "salary_mult": 1.13,
        "bonus_base": 0.09,
        "skills": [
            "Python", "FastAPI", "RAG", "Prompt Engineering", "Vector Databases", "LangChain",
            "LLM Evaluation", "Structured Outputs", "Docker", "API Design",
        ],
    },
    "LLM Engineer": {
        "weight": 0.06,
        "family": "genai-apps",
        "focuses": ["fine-tuning", "evaluation", "rag", "agents", "guardrails"],
        "salary_mult": 1.16,
        "bonus_base": 0.09,
        "skills": [
            "Python", "Prompt Engineering", "RAG", "LLM Evaluation", "Fine-Tuning", "Vector Databases",
            "Guardrails", "Synthetic Data", "Transformers", "FastAPI",
        ],
    },
    "Research Scientist": {
        "weight": 0.06,
        "family": "research",
        "focuses": ["nlp", "multimodal", "rl", "evaluation", "alignment"],
        "salary_mult": 1.14,
        "bonus_base": 0.11,
        "skills": [
            "Python", "PyTorch", "Transformers", "Deep Learning", "Experiment Tracking",
            "Distributed Training", "Research Methods", "Statistics", "Scientific Writing", "RLHF",
        ],
    },
    "Data Engineer": {
        "weight": 0.12,
        "family": "data-platform",
        "focuses": ["batch-pipelines", "streaming", "warehousing", "elt", "governance"],
        "salary_mult": 1.03,
        "bonus_base": 0.08,
        "skills": [
            "Python", "SQL", "Spark", "Airflow", "dbt", "Snowflake", "BigQuery",
            "Kafka", "Databricks", "Data Modeling",
        ],
    },
    "Analytics Engineer": {
        "weight": 0.08,
        "family": "analytics",
        "focuses": ["semantic-layer", "elt", "metrics", "experimentation", "self-serve-bi"],
        "salary_mult": 1.00,
        "bonus_base": 0.08,
        "skills": [
            "SQL", "dbt", "Data Modeling", "Looker", "Snowflake", "BigQuery", "Git", "Metrics Layer", "Python",
        ],
    },
    "MLOps Engineer": {
        "weight": 0.07,
        "family": "ml-operations",
        "focuses": ["deployment", "monitoring", "platform", "feature-stores", "governance"],
        "salary_mult": 1.08,
        "bonus_base": 0.09,
        "skills": [
            "Python", "Docker", "Kubernetes", "Terraform", "CI/CD", "MLflow", "Model Monitoring",
            "Feature Store", "AWS", "Prometheus",
        ],
    },
    "Data Analyst": {
        "weight": 0.12,
        "family": "analytics",
        "focuses": ["dashboards", "reporting", "operations", "experimentation", "insights"],
        "salary_mult": 0.93,
        "bonus_base": 0.07,
        "skills": [
            "SQL", "Python", "Tableau", "Excel", "Statistics", "Looker", "A/B Testing", "Pandas",
        ],
    },
    "BI Analyst": {
        "weight": 0.07,
        "family": "analytics",
        "focuses": ["dashboards", "finance-reporting", "kpis", "ops", "self-serve-bi"],
        "salary_mult": 0.90,
        "bonus_base": 0.07,
        "skills": [
            "SQL", "Power BI", "Tableau", "Excel", "Dashboarding", "Stakeholder Management", "Looker",
        ],
    },
    "Applied Scientist": {
        "weight": 0.05,
        "family": "modeling",
        "focuses": ["ranking", "nlp", "forecasting", "personalization", "causal-inference"],
        "salary_mult": 1.12,
        "bonus_base": 0.10,
        "skills": [
            "Python", "PyTorch", "scikit-learn", "XGBoost", "Statistics", "Feature Engineering",
            "Causal Inference", "Experiment Design", "SQL",
        ],
    },
    "AI Product Manager": {
        "weight": 0.03,
        "family": "product",
        "focuses": ["copilots", "workflow-automation", "evaluation", "ai-platform", "analytics"],
        "salary_mult": 1.02,
        "bonus_base": 0.12,
        "skills": [
            "Product Strategy", "Experiment Design", "SQL", "Prompt Engineering", "Analytics", "Roadmapping",
            "Stakeholder Management", "User Research",
        ],
    },
}

ROLE_NAMES = list(ROLE_CONFIGS)
ROLE_WEIGHTS = np.array([ROLE_CONFIGS[name]["weight"] for name in ROLE_NAMES], dtype=float)
ROLE_WEIGHTS /= ROLE_WEIGHTS.sum()

SKILL_CATEGORY = {
    "Python": ("programming", False),
    "SQL": ("analytics", False),
    "Statistics": ("analytics", False),
    "scikit-learn": ("ml", False),
    "Pandas": ("analytics", False),
    "A/B Testing": ("analytics", False),
    "XGBoost": ("ml", False),
    "Feature Engineering": ("ml", False),
    "Experiment Design": ("analytics", False),
    "Jupyter": ("tooling", False),
    "PyTorch": ("ml", False),
    "TensorFlow": ("ml", False),
    "Docker": ("platform", False),
    "Kubernetes": ("platform", False),
    "MLflow": ("mlops", False),
    "Airflow": ("data-platform", False),
    "Feature Store": ("mlops", False),
    "CI/CD": ("platform", False),
    "Spark": ("data-platform", False),
    "FastAPI": ("backend", False),
    "RAG": ("llm", True),
    "Prompt Engineering": ("llm", True),
    "Vector Databases": ("llm", True),
    "LangChain": ("llm", True),
    "LLM Evaluation": ("llm", True),
    "Structured Outputs": ("llm", True),
    "API Design": ("backend", False),
    "Fine-Tuning": ("llm", True),
    "Guardrails": ("llm", True),
    "Synthetic Data": ("llm", True),
    "Transformers": ("ml", True),
    "Deep Learning": ("ml", False),
    "Experiment Tracking": ("mlops", False),
    "Distributed Training": ("mlops", False),
    "Research Methods": ("research", False),
    "Scientific Writing": ("research", False),
    "RLHF": ("llm", True),
    "dbt": ("data-platform", False),
    "Snowflake": ("data-platform", False),
    "BigQuery": ("data-platform", False),
    "Kafka": ("data-platform", False),
    "Databricks": ("data-platform", False),
    "Data Modeling": ("analytics", False),
    "Looker": ("bi", False),
    "Git": ("tooling", False),
    "Metrics Layer": ("analytics", False),
    "Terraform": ("platform", False),
    "Model Monitoring": ("mlops", False),
    "AWS": ("cloud", False),
    "Prometheus": ("platform", False),
    "Tableau": ("bi", False),
    "Excel": ("bi", False),
    "Power BI": ("bi", False),
    "Dashboarding": ("bi", False),
    "Stakeholder Management": ("product", False),
    "Causal Inference": ("analytics", False),
    "Product Strategy": ("product", False),
    "Analytics": ("analytics", False),
    "Roadmapping": ("product", False),
    "User Research": ("product", False),
    "AWS SageMaker": ("cloud", False),
    "GCP": ("cloud", False),
    "Azure": ("cloud", False),
    "OpenAI API": ("llm", True),
    "Anthropic API": ("llm", True),
    "Gemini API": ("llm", True),
    "vLLM": ("llm", True),
    "Pinecone": ("llm", True),
    "Weaviate": ("llm", True),
    "pgvector": ("llm", True),
    "DSPy": ("llm", True),
}

CLOUD_STACK_OPTIONS = [
    ["AWS"],
    ["GCP"],
    ["Azure"],
    ["AWS", "Snowflake"],
    ["GCP", "BigQuery"],
    ["Azure", "Databricks"],
]
LLM_STACK_OPTIONS = [
    ["OpenAI API", "LangChain", "Pinecone"],
    ["Anthropic API", "DSPy", "pgvector"],
    ["Gemini API", "RAG", "Weaviate"],
    ["OpenAI API", "vLLM", "LLM Evaluation"],
]

SENIORITY_LEVELS = ["entry", "mid", "senior", "staff", "principal"]
SENIORITY_WEIGHTS = np.array([0.18, 0.34, 0.28, 0.14, 0.06], dtype=float)
SENIORITY_WEIGHTS /= SENIORITY_WEIGHTS.sum()
SENIORITY_YEARS = {
    "entry": (0, 2),
    "mid": (2, 5),
    "senior": (5, 8),
    "staff": (8, 11),
    "principal": (10, 15),
}
SENIORITY_MULT = {
    "entry": 0.72,
    "mid": 1.00,
    "senior": 1.24,
    "staff": 1.48,
    "principal": 1.72,
}

EMPLOYMENT_TYPES = ["full_time", "contract", "internship"]
EMPLOYMENT_WEIGHTS = np.array([0.87, 0.10, 0.03], dtype=float)
EMPLOYMENT_WEIGHTS /= EMPLOYMENT_WEIGHTS.sum()

EDUCATION_BY_FAMILY = {
    "analytics": ["none", "bachelor", "master"],
    "product": ["bachelor", "master"],
    "data-platform": ["bachelor", "master"],
    "ml-engineering": ["bachelor", "master"],
    "genai-apps": ["bachelor", "master"],
    "research": ["master", "phd"],
    "ml-operations": ["bachelor", "master"],
    "modeling": ["bachelor", "master", "phd"],
}

DESCRIPTION_TEMPLATES = [
    "Join {company_name} as a {seniority} {job_title} working on {focus_area} problems in {industry}. "
    "You will partner with cross-functional teams to ship production work using {skills_preview}.",
    "We are hiring a {job_title} to help scale {company_name}'s {focus_area} roadmap. "
    "The role blends hands-on delivery, stakeholder collaboration, and modern tooling across {skills_preview}.",
    "{company_name} is looking for a {seniority} {job_title} to improve {focus_area} outcomes for customers in {industry}. "
    "You will own high-impact projects spanning experimentation, delivery, and operational excellence with {skills_preview}.",
]


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace("&", "and")
        .replace("/", "-")
        .replace(" ", "-")
        .replace(",", "")
        .replace(".", "")
    )


def round_salary(value: float) -> int:
    return int(round(value / 1000.0) * 1000)


def pick_location() -> dict[str, object]:
    idx = int(rng.choice(len(LOCATIONS), p=LOCATION_WEIGHTS))
    return LOCATIONS[idx]


def month_weighted_dates(n: int) -> pd.DatetimeIndex:
    months = pd.period_range(DATE_START, DATE_END, freq="M")
    month_weights = []
    for month in months:
        month_start = month.to_timestamp()
        age_index = (month_start.year - 2024) * 12 + (month_start.month - 1)
        weight = 1.0 + 0.06 * age_index
        month_weights.append(weight)
    month_weights = np.asarray(month_weights, dtype=float)
    month_weights /= month_weights.sum()
    chosen_months = rng.choice(months.astype(str), size=n, p=month_weights)
    dates = []
    for month_str in chosen_months:
        month = pd.Period(month_str, freq="M")
        start = month.to_timestamp()
        end = (month + 1).to_timestamp() - pd.Timedelta(days=1)
        day = int(rng.integers(0, (end - start).days + 1))
        dates.append(start + pd.Timedelta(days=day))
    return pd.DatetimeIndex(dates)


def build_company_name(industry: str, seq: int) -> str:
    prefixes = [
        "Apex", "Northstar", "Signal", "Vector", "Nimbus", "Atlas", "Cobalt", "Meridian",
        "Beacon", "Vertex", "Pioneer", "Orbit", "Foundry", "Helix", "Summit", "Prism",
    ]
    suffixes = {
        "software": ["Labs", "Systems", "Cloud", "Works", "AI", "Platform"],
        "finance": ["Capital", "Markets", "Finance", "Analytics", "Investments"],
        "healthcare": ["Health", "Bio", "Care", "Clinical", "Medical"],
        "cloud": ["Compute", "Infra", "Ops", "Cloud", "Security"],
        "retail": ["Commerce", "Retail", "Consumer", "Marketplace", "Insights"],
        "consulting": ["Advisory", "Partners", "Consulting", "Solutions"],
        "media": ["Media", "Streaming", "Content", "Studios", "Signals"],
        "public-sector": ["Civic", "Public Data", "GovTech", "Policy"],
        "education": ["Learning", "Edu", "Academy", "Scholar"],
        "manufacturing": ["Industrial", "Factory", "Automation", "Works"],
    }
    prefix = prefixes[seq % len(prefixes)]
    suffix = suffixes[industry][seq % len(suffixes[industry])]
    return f"{prefix} {suffix}"


def generate_companies() -> pd.DataFrame:
    industries = list(INDUSTRIES)
    industry_weights = np.array([INDUSTRIES[name]["weight"] for name in industries], dtype=float)
    industry_weights /= industry_weights.sum()

    rows = []
    used_names: set[str] = set()
    for idx in range(N_COMPANIES):
        industry = str(rng.choice(industries, p=industry_weights))
        size = str(rng.choice(list(COMPANY_SIZE_BANDS), p=COMPANY_SIZE_WEIGHTS))
        funding_stage = str(rng.choice(FUNDING_STAGE_BY_SIZE[size]))
        location = pick_location()
        company_type = str(rng.choice(INDUSTRIES[industry]["company_types"]))
        name = build_company_name(industry, idx)
        if name in used_names:
            name = f"{name} {idx + 1}"
        used_names.add(name)
        min_emp, max_emp = COMPANY_SIZE_BANDS[size]
        employee_count = int(rng.integers(min_emp, max_emp + 1))
        remote_first = bool(rng.random() < (0.45 if size in {"startup", "small"} else 0.24))
        ai_maturity = float(np.clip(rng.normal(69 if industry in {"software", "cloud", "finance"} else 58, 12), 20, 98))
        benefits_score = float(np.clip(rng.normal(72 if size in {"large", "enterprise"} else 63, 10), 35, 96))
        glassdoor_rating = float(np.clip(rng.normal(3.9 if size in {"large", "enterprise"} else 3.7, 0.35), 2.7, 4.9))
        hiring_velocity = int(np.clip(rng.normal(68 if remote_first else 56, 14), 20, 99))
        rows.append(
            {
                "company_id": f"COMP{idx + 1:04d}",
                "company_name": name,
                "industry": industry,
                "company_type": company_type,
                "company_size": size,
                "funding_stage": funding_stage,
                "hq_city": location["city"],
                "country": location["country"],
                "region": location["region"],
                "employee_count_estimate": employee_count,
                "remote_first": int(remote_first),
                "ai_maturity_score": round(ai_maturity, 1),
                "benefits_score": round(benefits_score, 1),
                "glassdoor_like_rating": round(glassdoor_rating, 2),
                "hiring_velocity_index": hiring_velocity,
            }
        )
    return pd.DataFrame(rows)


def adjusted_role_weights(posted_date: pd.Timestamp) -> np.ndarray:
    weights = ROLE_WEIGHTS.copy()
    if posted_date.year >= 2025:
        for role_name in ("AI Engineer", "LLM Engineer", "AI Product Manager"):
            weights[ROLE_NAMES.index(role_name)] *= 1.25
    if posted_date.year >= 2026:
        for role_name in ("AI Engineer", "LLM Engineer"):
            weights[ROLE_NAMES.index(role_name)] *= 1.15
    weights /= weights.sum()
    return weights


def pick_cloud_stack() -> list[str]:
    idx = int(rng.integers(0, len(CLOUD_STACK_OPTIONS)))
    return list(CLOUD_STACK_OPTIONS[idx])


def pick_llm_stack() -> list[str]:
    idx = int(rng.integers(0, len(LLM_STACK_OPTIONS)))
    return list(LLM_STACK_OPTIONS[idx])


def pick_skills(job_title: str, family: str, focus_area: str, cloud_stack: list[str], llm_stack: list[str]) -> list[str]:
    role_skills = list(ROLE_CONFIGS[job_title]["skills"])
    pool = role_skills + cloud_stack
    if family in {"genai-apps", "research"} or focus_area in {"rag", "agents", "fine-tuning", "evaluation"}:
        pool += llm_stack
    if family in {"data-platform", "ml-engineering", "ml-operations"}:
        pool += ["AWS", "GCP", "Azure"]
    unique_pool = []
    for skill in pool:
        if skill not in unique_pool:
            unique_pool.append(skill)
    n_skills = int(rng.integers(6, min(11, len(unique_pool)) + 1))
    picked = list(rng.choice(unique_pool, size=n_skills, replace=False))
    role_anchor = ROLE_CONFIGS[job_title]["skills"][:2]
    for anchor in reversed(role_anchor):
        if anchor not in picked:
            picked.insert(0, anchor)
    deduped = []
    for skill in picked:
        if skill not in deduped:
            deduped.append(skill)
    return deduped[:12]


def generate_jobs(companies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    posted_dates = month_weighted_dates(N_JOBS)
    jobs = []
    skill_rows = []
    company_idx = companies.set_index("company_id")
    for idx in range(N_JOBS):
        posted_date = pd.Timestamp(posted_dates[idx])
        role = str(rng.choice(ROLE_NAMES, p=adjusted_role_weights(posted_date)))
        role_cfg = ROLE_CONFIGS[role]
        seniority = str(rng.choice(SENIORITY_LEVELS, p=SENIORITY_WEIGHTS))
        company_id = str(rng.choice(companies["company_id"]))
        company = company_idx.loc[company_id]
        employment_type = str(rng.choice(EMPLOYMENT_TYPES, p=EMPLOYMENT_WEIGHTS))
        if seniority != "entry" and employment_type == "internship":
            employment_type = "full_time"

        remote_probs = np.array([0.25, 0.45, 0.30], dtype=float)
        if int(company["remote_first"]) == 1:
            remote_probs = np.array([0.10, 0.32, 0.58], dtype=float)
        remote_type = str(rng.choice(["onsite", "hybrid", "remote"], p=remote_probs / remote_probs.sum()))
        focus_area = str(rng.choice(role_cfg["focuses"]))
        cloud_stack = pick_cloud_stack()
        llm_stack = pick_llm_stack() if role_cfg["family"] in {"genai-apps", "research"} or focus_area in {"rag", "agents", "fine-tuning", "evaluation"} else []
        skills = pick_skills(role, role_cfg["family"], focus_area, cloud_stack, llm_stack)

        min_years, max_years = SENIORITY_YEARS[seniority]
        exp_min = max(0, min_years + int(rng.integers(0, 2)))
        exp_max = max(exp_min + 1, max_years + int(rng.integers(0, 2)))

        base_mid = 118_000
        base_mid *= role_cfg["salary_mult"]
        base_mid *= SENIORITY_MULT[seniority]
        base_mid *= INDUSTRIES[str(company["industry"])]["salary_mult"]
        base_mid *= COMPANY_SIZE_MULT[str(company["company_size"])]
        location_match = next(item for item in LOCATIONS if item["city"] == company["hq_city"])
        base_mid *= float(location_match["salary_mult"])
        if remote_type == "remote":
            base_mid *= 1.03
        if posted_date.year == 2025:
            base_mid *= 1.03
        elif posted_date.year >= 2026:
            base_mid *= 1.06
        if role_cfg["family"] == "genai-apps":
            base_mid *= 1.07
        if str(company["company_size"]) == "startup":
            base_mid *= 0.96
        volatility = float(rng.normal(1.0, 0.06))
        base_mid *= volatility

        equity_offered = int(
            rng.random()
            < (0.72 if str(company["company_size"]) in {"startup", "small"} and employment_type == "full_time" else 0.18)
        )
        bonus_target = role_cfg["bonus_base"] + (0.03 if company["industry"] == "finance" else 0.0) + (0.02 if seniority in {"staff", "principal"} else 0.0)
        bonus_target = float(np.clip(bonus_target + rng.normal(0.0, 0.01), 0.03, 0.30))

        visa_sponsorship = int(
            rng.random()
            < (
                0.34
                if str(company["company_size"]) in {"large", "enterprise"} and company["country"] in {"United States", "Canada", "United Kingdom", "Germany"}
                else 0.12
            )
        )

        salary_mid = round_salary(base_mid)
        salary_min = round_salary(salary_mid * float(rng.uniform(0.82, 0.90)))
        salary_max = round_salary(salary_mid * float(rng.uniform(1.10, 1.22)))
        salary_max = max(salary_max, salary_min + 5_000)

        education_pool = EDUCATION_BY_FAMILY[role_cfg["family"]]
        education_required = str(rng.choice(education_pool))
        primary_skill_cluster = {
            "analytics": "analytics",
            "product": "product",
            "data-platform": "data-platform",
            "ml-engineering": "ml-engineering",
            "genai-apps": "llm",
            "research": "research",
            "ml-operations": "mlops",
            "modeling": "ml",
        }[role_cfg["family"]]

        description = str(rng.choice(DESCRIPTION_TEMPLATES)).format(
            company_name=company["company_name"],
            job_title=role,
            seniority=seniority,
            industry=company["industry"],
            focus_area=focus_area.replace("-", " "),
            skills_preview=", ".join(skills[:3]),
        )

        applications = int(
            np.clip(
                rng.normal(
                    40
                    + (10 if remote_type == "remote" else 0)
                    + (6 if role in {"Data Analyst", "Data Scientist"} else 0)
                    + (4 if role_cfg["family"] == "genai-apps" else 0),
                    14,
                ),
                4,
                220,
            )
        )
        days_open = int(
            np.clip(
                rng.normal(
                    32
                    + (10 if seniority in {"staff", "principal"} else 0)
                    + (8 if role_cfg["family"] == "research" else 0)
                    - (0.06 * applications),
                    9,
                ),
                6,
                120,
            )
        )
        is_filled = int(applications >= 18 and days_open >= 12 and rng.random() < 0.82)
        hiring_urgency = "high" if days_open <= 18 else ("medium" if days_open <= 40 else "low")

        job_id = f"JOB{idx + 1:06d}"
        jobs.append(
            {
                "job_id": job_id,
                "company_id": company_id,
                "company_name": company["company_name"],
                "posted_date": posted_date.date().isoformat(),
                "job_title": role,
                "role_family": role_cfg["family"],
                "seniority": seniority,
                "employment_type": employment_type,
                "remote_type": remote_type,
                "city": company["hq_city"],
                "country": company["country"],
                "region": company["region"],
                "industry": company["industry"],
                "company_size": company["company_size"],
                "funding_stage": company["funding_stage"],
                "company_type": company["company_type"],
                "salary_currency": "USD",
                "salary_min_usd": salary_min,
                "salary_max_usd": salary_max,
                "salary_mid_usd": salary_mid,
                "bonus_target_pct": round(bonus_target, 3),
                "equity_offered": equity_offered,
                "experience_min_years": exp_min,
                "experience_max_years": exp_max,
                "education_required": education_required,
                "visa_sponsorship": visa_sponsorship,
                "ai_focus_area": focus_area,
                "primary_skill_cluster": primary_skill_cluster,
                "cloud_stack": "|".join(cloud_stack),
                "llm_stack": "|".join(llm_stack) if llm_stack else "",
                "required_skills": "|".join(skills),
                "skills_count": len(skills),
                "applications_30d": applications,
                "days_open": days_open,
                "is_filled": is_filled,
                "hiring_urgency": hiring_urgency,
                "description": description,
            }
        )

        for skill_idx, skill in enumerate(skills):
            skill_category, is_genai_skill = SKILL_CATEGORY.get(skill, ("other", False))
            importance = "core" if skill_idx < 3 else ("strong" if skill_idx < 6 else "nice_to_have")
            proficiency = "advanced" if skill_idx < 3 else ("intermediate" if skill_idx < 6 else "basic")
            skill_rows.append(
                {
                    "job_id": job_id,
                    "skill": skill,
                    "skill_category": skill_category,
                    "importance": importance,
                    "proficiency_level": proficiency,
                    "is_genai_skill": int(is_genai_skill),
                }
            )

    return pd.DataFrame(jobs), pd.DataFrame(skill_rows)


def build_salary_benchmarks(jobs: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        jobs.groupby(["job_title", "country", "seniority", "remote_type"], observed=False)
        .agg(
            sample_jobs=("job_id", "count"),
            salary_p10_usd=("salary_mid_usd", lambda s: int(round(np.quantile(s, 0.10)))),
            salary_median_usd=("salary_mid_usd", lambda s: int(round(np.median(s)))),
            salary_p90_usd=("salary_mid_usd", lambda s: int(round(np.quantile(s, 0.90)))),
            bonus_target_pct_median=("bonus_target_pct", lambda s: round(float(np.median(s)), 3)),
            visa_share=("visa_sponsorship", lambda s: round(float(np.mean(s)), 3)),
        )
        .reset_index()
    )
    return grouped[grouped["sample_jobs"] >= 15].sort_values(
        ["job_title", "country", "seniority", "remote_type"]
    ).reset_index(drop=True)


def build_skill_demand_monthly(jobs: pd.DataFrame, job_skills: pd.DataFrame) -> pd.DataFrame:
    jobs_with_month = jobs[["job_id", "posted_date", "salary_mid_usd", "remote_type"]].copy()
    jobs_with_month["year_month"] = pd.to_datetime(jobs_with_month["posted_date"]).dt.to_period("M").astype(str)
    merged = job_skills.merge(jobs_with_month, on="job_id", how="left")
    counts = merged["skill"].value_counts()
    top_skills = counts.head(40).index
    merged = merged[merged["skill"].isin(top_skills)].copy()

    monthly_jobs = jobs_with_month.groupby("year_month", observed=False)["job_id"].nunique().rename("month_jobs")
    grouped = (
        merged.groupby(["year_month", "skill", "skill_category"], observed=False)
        .agg(
            job_count=("job_id", "nunique"),
            median_salary_mid_usd=("salary_mid_usd", lambda s: int(round(np.median(s)))),
            remote_share=("remote_type", lambda s: round(float((s == "remote").mean()), 3)),
        )
        .reset_index()
    )
    grouped = grouped.merge(monthly_jobs.reset_index(), on="year_month", how="left")
    grouped["share_of_postings"] = (grouped["job_count"] / grouped["month_jobs"]).round(4)
    grouped = grouped.drop(columns=["month_jobs"])
    return grouped.sort_values(["year_month", "job_count", "skill"], ascending=[True, False, True]).reset_index(drop=True)


def main() -> None:
    companies = generate_companies()
    jobs, job_skills = generate_jobs(companies)
    salary_benchmarks = build_salary_benchmarks(jobs)
    skill_demand = build_skill_demand_monthly(jobs, job_skills)

    companies.to_csv(OUTPUT_DIR / "companies.csv", index=False)
    jobs.to_csv(OUTPUT_DIR / "jobs.csv", index=False)
    job_skills.to_csv(OUTPUT_DIR / "job_skills.csv", index=False)
    salary_benchmarks.to_csv(OUTPUT_DIR / "salary_benchmarks.csv", index=False)
    skill_demand.to_csv(OUTPUT_DIR / "skill_demand_monthly.csv", index=False)

    print("Generated AI & Data Jobs Market dataset")
    print(f"  companies.csv            {len(companies):>8,} rows x {companies.shape[1]} cols")
    print(f"  jobs.csv                 {len(jobs):>8,} rows x {jobs.shape[1]} cols")
    print(f"  job_skills.csv           {len(job_skills):>8,} rows x {job_skills.shape[1]} cols")
    print(f"  salary_benchmarks.csv    {len(salary_benchmarks):>8,} rows x {salary_benchmarks.shape[1]} cols")
    print(f"  skill_demand_monthly.csv {len(skill_demand):>8,} rows x {skill_demand.shape[1]} cols")


if __name__ == "__main__":
    main()
