"""
Generate synthetic mental health in tech survey dataset.
5,000 responses inspired by the OSMI Mental Health in Tech Survey.
"""

import numpy as np
import pandas as pd
import random
from pathlib import Path

np.random.seed(42)
random.seed(42)

N = 5_000
OUTPUT = Path(__file__).parent / "mental_health_tech.csv"

COUNTRIES = {
    "United States": 0.45,
    "United Kingdom": 0.12,
    "Canada": 0.07,
    "Germany": 0.06,
    "Netherlands": 0.04,
    "Australia": 0.04,
    "Ireland": 0.03,
    "India": 0.03,
    "Sweden": 0.02,
    "France": 0.02,
    "Brazil": 0.02,
    "Other": 0.10,
}
COUNTRY_LIST = list(COUNTRIES.keys())
COUNTRY_W = np.array(list(COUNTRIES.values()))
COUNTRY_W /= COUNTRY_W.sum()

GENDERS = ["Male", "Female", "Non-binary", "Other / Prefer not to say"]
GENDER_W = [0.60, 0.31, 0.06, 0.03]

COMPANY_SIZES = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
COMPANY_W = [0.08, 0.15, 0.22, 0.25, 0.12, 0.18]

YES_NO = ["Yes", "No"]
YES_NO_SOMETIMES = ["Yes", "No", "Sometimes"]
YES_NO_DONT_KNOW = ["Yes", "No", "Don't know"]
WORK_INTERFERE = ["Never", "Rarely", "Sometimes", "Often"]
LEAVE_EASE = ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]
DISCUSS_OPTIONS = ["Yes", "No", "Some of them"]
INTERVIEW_OPTIONS = ["Yes", "No", "Maybe"]
MV_PHYSICAL_OPTIONS = ["Yes", "No", "Don't know"]
YN_SOME = ["Yes", "No", "Some of them"]

SURVEY_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
YEAR_W = [0.12, 0.13, 0.15, 0.18, 0.22, 0.20]

COMMENTS_POOL = [
    "", "", "", "", "", "", "", "",
    "Better awareness programs needed.",
    "Management needs training on mental health.",
    "Glad to see more companies taking this seriously.",
    "Stigma is still a major issue in my workplace.",
    "Remote work has been a double-edged sword.",
    "HR resources are available but rarely used.",
    "Open conversations help reduce stigma.",
    "More anonymous support options would help.",
    "Burnout is rampant in my team.",
    "We have EAP but nobody talks about it.",
    "Leadership support makes all the difference.",
    "Anonymous reporting would increase participation.",
]


def weighted_choice(options, weights):
    return np.random.choice(options, p=np.array(weights) / sum(weights))


def yn(prob_yes):
    return "Yes" if random.random() < prob_yes else "No"


rows = []
for i in range(1, N + 1):
    year = int(np.random.choice(SURVEY_YEARS, p=YEAR_W))
    country = np.random.choice(COUNTRY_LIST, p=COUNTRY_W)
    gender = np.random.choice(GENDERS, p=GENDER_W)

    # Age: realistic tech distribution peaking 25-35
    age = int(np.clip(np.random.normal(30, 8), 18, 65))

    self_employed = yn(0.12)
    family_history = yn(0.30)

    # Treatment more likely with family history, work interference
    base_treat = 0.40
    treat_boost = 0.20 if family_history == "Yes" else 0
    treatment = yn(base_treat + treat_boost)

    # Work interference correlated with treatment-seeking
    if treatment == "Yes":
        wi = np.random.choice(WORK_INTERFERE, p=[0.10, 0.20, 0.40, 0.30])
    else:
        wi = np.random.choice(WORK_INTERFERE, p=[0.35, 0.35, 0.20, 0.10])

    company_size = weighted_choice(COMPANY_SIZES, COMPANY_W)
    remote_work = yn(0.35 if year < 2020 else 0.55)
    tech_company = yn(0.72)

    # Larger companies tend to have better benefits
    big_company = company_size in ["100-500", "500-1000", "More than 1000"]
    benefits = np.random.choice(YES_NO_DONT_KNOW,
                                p=[0.55, 0.25, 0.20] if big_company else [0.25, 0.45, 0.30])
    care_options = np.random.choice(YES_NO_DONT_KNOW,
                                    p=[0.50, 0.20, 0.30] if big_company else [0.20, 0.50, 0.30])
    wellness_program = np.random.choice(YES_NO_DONT_KNOW,
                                        p=[0.48, 0.30, 0.22] if big_company else [0.18, 0.52, 0.30])
    seek_help = np.random.choice(YES_NO_DONT_KNOW,
                                 p=[0.52, 0.22, 0.26] if big_company else [0.22, 0.48, 0.30])
    anonymity = np.random.choice(YES_NO_DONT_KNOW,
                                 p=[0.45, 0.18, 0.37] if big_company else [0.20, 0.40, 0.40])
    leave = np.random.choice(LEAVE_EASE, p=[0.12, 0.28, 0.30, 0.20, 0.10])

    # Consequence perceptions
    mh_consequence = np.random.choice(["Yes", "No", "Maybe"], p=[0.25, 0.45, 0.30])
    ph_consequence = np.random.choice(["Yes", "No", "Maybe"], p=[0.10, 0.72, 0.18])

    coworkers = np.random.choice(["Yes", "No", "Some of them"], p=[0.30, 0.42, 0.28])
    supervisor = np.random.choice(["Yes", "No", "Some of them"], p=[0.32, 0.44, 0.24])

    # Interview disclosures — most people say no/maybe
    mh_interview = np.random.choice(INTERVIEW_OPTIONS, p=[0.08, 0.62, 0.30])
    ph_interview = np.random.choice(INTERVIEW_OPTIONS, p=[0.20, 0.50, 0.30])

    mental_vs_phys = np.random.choice(MV_PHYSICAL_OPTIONS, p=[0.35, 0.28, 0.37])
    obs_consequence = yn(0.22)

    comment = random.choice(COMMENTS_POOL)

    rows.append({
        "respondent_id": i,
        "survey_year": year,
        "age": age,
        "gender": gender,
        "country": country,
        "self_employed": self_employed,
        "family_history": family_history,
        "treatment": treatment,
        "work_interfere": wi,
        "no_employees": company_size,
        "remote_work": remote_work,
        "tech_company": tech_company,
        "benefits": benefits,
        "care_options": care_options,
        "wellness_program": wellness_program,
        "seek_help": seek_help,
        "anonymity": anonymity,
        "leave": leave,
        "mental_health_consequence": mh_consequence,
        "phys_health_consequence": ph_consequence,
        "coworkers": coworkers,
        "supervisor": supervisor,
        "mental_health_interview": mh_interview,
        "phys_health_interview": ph_interview,
        "mental_vs_physical": mental_vs_phys,
        "obs_consequence": obs_consequence,
        "comments": comment,
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(f"Saved {len(df):,} responses to {OUTPUT}")
print(f"Treatment rate: {(df['treatment']=='Yes').mean():.1%}")
print(f"Gender split:\n{df['gender'].value_counts()}")
