#!/usr/bin/env python3
"""
Synthetic Student Academic Performance Dataset Generator
=========================================================
Generates 10,000 student records across 25 features covering demographics,
study habits, family background, school resources, and academic outcomes.

Includes realistic correlations:
  - study_hours is the strongest predictor of scores
  - attendance_rate has a strong positive effect
  - sleep_hours has a quadratic optimum at ~7.5 hours
  - tutoring_sessions shows diminishing returns
  - stress_level negatively impacts performance
  - motivation_score positively impacts performance
  - parental_education and family_income have modest positive effects
  - private/charter schools show a small positive premium
  - internet_access and has_laptop give a small boost

Usage:
    python create_dataset.py   # writes students.csv
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
N_STUDENTS = 10_000
OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Lookup tables / encodings
# ---------------------------------------------------------------------------
PARENTAL_EDU_LEVELS = ["none", "high_school", "some_college", "bachelor", "master", "phd"]
PARENTAL_EDU_WEIGHTS = [0.05, 0.30, 0.25, 0.25, 0.10, 0.05]
PARENTAL_EDU_NUMERIC = {lvl: i for i, lvl in enumerate(PARENTAL_EDU_LEVELS)}  # 0-5

INCOME_LEVELS = ["low", "middle", "high"]
INCOME_WEIGHTS = [0.25, 0.50, 0.25]
INCOME_BONUS = {"low": -3, "middle": 0, "high": 3}

SCHOOL_TYPE_BONUS = {"public": 0, "private": 3, "charter": 1.5}

PARENTAL_INVOLVEMENT_BONUS = {"low": -2, "medium": 0, "high": 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sleep_bonus(hours: np.ndarray) -> np.ndarray:
    """Quadratic bonus peaking at 7.5 hours; penalises both under- and over-sleep."""
    return -2.0 * (hours - 7.5) ** 2 + 4.0


# ---------------------------------------------------------------------------
# Generate STUDENTS
# ---------------------------------------------------------------------------
def generate_students() -> pd.DataFrame:
    """Build the full students DataFrame with realistic feature correlations."""

    # --- Demographics ---
    ages = rng.integers(15, 26, N_STUDENTS)  # 15-25 inclusive

    genders = rng.choice(
        ["M", "F", "Non-binary"],
        size=N_STUDENTS,
        p=[0.48, 0.48, 0.04],
    )

    ethnicities = rng.choice(
        ["A", "B", "C", "D", "E"],
        size=N_STUDENTS,
        p=[0.15, 0.20, 0.30, 0.25, 0.10],
    )

    parental_education = rng.choice(
        PARENTAL_EDU_LEVELS,
        size=N_STUDENTS,
        p=PARENTAL_EDU_WEIGHTS,
    )
    parental_edu_numeric = np.array([PARENTAL_EDU_NUMERIC[e] for e in parental_education])

    family_income = rng.choice(
        INCOME_LEVELS,
        size=N_STUDENTS,
        p=INCOME_WEIGHTS,
    )
    income_bonus_arr = np.array([INCOME_BONUS[i] for i in family_income])

    # --- School ---
    school_type = rng.choice(
        ["public", "private", "charter"],
        size=N_STUDENTS,
        p=[0.65, 0.20, 0.15],
    )
    school_bonus_arr = np.array([SCHOOL_TYPE_BONUS[s] for s in school_type])

    school_region = rng.choice(
        ["urban", "suburban", "rural"],
        size=N_STUDENTS,
        p=[0.40, 0.40, 0.20],
    )

    # --- Study habits ---
    # Higher-income/higher-parental-edu students study slightly more
    study_base = 10 + parental_edu_numeric * 0.5 + income_bonus_arr * 0.3
    study_hours_per_week = np.clip(
        rng.normal(study_base, 6), 0, 40
    ).round(1)

    # Attendance: higher-motivation students attend more
    motivation_score = np.clip(rng.normal(5.5, 2.0, N_STUDENTS), 1, 10).round(1)
    attendance_base = 75 + motivation_score * 1.5
    attendance_rate = np.clip(
        rng.normal(attendance_base, 8), 50, 100
    ).round(1)

    extracurricular_activities = rng.integers(0, 6, N_STUDENTS)  # 0-5

    sports_participation = rng.choice(
        ["yes", "no"],
        size=N_STUDENTS,
        p=[0.40, 0.60],
    )

    tutoring_sessions = rng.integers(0, 21, N_STUDENTS)  # 0-20

    parental_involvement = rng.choice(
        ["low", "medium", "high"],
        size=N_STUDENTS,
        p=[0.25, 0.45, 0.30],
    )
    involvement_bonus_arr = np.array([PARENTAL_INVOLVEMENT_BONUS[p] for p in parental_involvement])

    # --- Resources ---
    # Higher income -> more likely to have internet and laptop
    internet_prob = np.where(family_income == "high", 0.97,
                    np.where(family_income == "middle", 0.85, 0.65))
    internet_access = (rng.random(N_STUDENTS) < internet_prob).astype(int)

    laptop_prob = np.where(family_income == "high", 0.95,
                  np.where(family_income == "middle", 0.80, 0.55))
    has_laptop = (rng.random(N_STUDENTS) < laptop_prob).astype(int)

    resource_bonus = internet_access * 1.5 + has_laptop * 1.5

    # --- Psychological ---
    sleep_hours = np.clip(rng.normal(7.0, 1.2, N_STUDENTS), 4, 10).round(1)
    sleep_bonus_arr = sleep_bonus(sleep_hours)

    # Stress: negatively correlated with sleep and positively with study hours
    stress_base = 5.0 - sleep_bonus_arr * 0.3 + study_hours_per_week * 0.04
    stress_level = np.clip(
        rng.normal(stress_base, 1.5), 1, 10
    ).round(1)

    # Motivation: positively correlated with parental involvement and income
    motivation_score = np.clip(
        motivation_score
        + involvement_bonus_arr * 0.3
        + income_bonus_arr * 0.1,
        1, 10,
    ).round(1)

    # ---------------------------------------------------------------------------
    # Score generation
    # ---------------------------------------------------------------------------
    # Shared base score (captures overall academic tendency)
    tutoring_effect = tutoring_sessions * 1.0 - tutoring_sessions ** 2 * 0.05

    base_score = (
        40
        + study_hours_per_week * 1.5
        + (attendance_rate - 70) * 0.5
        + tutoring_effect
        + parental_edu_numeric * 2.0
        + motivation_score * 1.5
        + sleep_bonus_arr
        - stress_level * 1.0
        + income_bonus_arr
        + school_bonus_arr
        + resource_bonus
        + involvement_bonus_arr
    )

    # Per-subject scores: common component + subject-specific noise
    # Students who do well overall tend to do well in all subjects,
    # but not perfectly (individual subject affinity adds variance)
    subject_noise_std = 10.0
    common_noise = rng.normal(0, 8, N_STUDENTS)  # shared random component

    reading_score = np.clip(
        base_score + common_noise + rng.normal(0, subject_noise_std, N_STUDENTS),
        0, 100,
    ).round(1)

    writing_score = np.clip(
        base_score + common_noise * 0.85 + rng.normal(0, subject_noise_std, N_STUDENTS),
        0, 100,
    ).round(1)

    # Math and science have a slight additional boost from study hours
    math_score = np.clip(
        base_score + study_hours_per_week * 0.3
        + common_noise * 0.70
        + rng.normal(0, subject_noise_std, N_STUDENTS),
        0, 100,
    ).round(1)

    science_score = np.clip(
        base_score + study_hours_per_week * 0.2
        + common_noise * 0.75
        + rng.normal(0, subject_noise_std, N_STUDENTS),
        0, 100,
    ).round(1)

    # GPA: weighted average of all four subjects mapped to 0-4 scale, plus small noise
    avg_score = (reading_score * 0.25 + writing_score * 0.25
                 + math_score * 0.25 + science_score * 0.25)
    overall_gpa = np.clip(
        avg_score / 25.0 + rng.normal(0, 0.15, N_STUDENTS),
        0.0, 4.0,
    ).round(2)

    # Binary pass/fail: GPA >= 2.0
    passed = (overall_gpa >= 2.0).astype(int)

    # ---------------------------------------------------------------------------
    # Assemble DataFrame
    # ---------------------------------------------------------------------------
    df = pd.DataFrame({
        "student_id":               [f"STU{str(i).zfill(5)}" for i in range(N_STUDENTS)],
        "age":                      ages,
        "gender":                   genders,
        "ethnicity":                ethnicities,
        "parental_education":       parental_education,
        "family_income":            family_income,
        "school_type":              school_type,
        "school_region":            school_region,
        "study_hours_per_week":     study_hours_per_week,
        "attendance_rate":          attendance_rate,
        "extracurricular_activities": extracurricular_activities,
        "sports_participation":     sports_participation,
        "tutoring_sessions":        tutoring_sessions,
        "parental_involvement":     parental_involvement,
        "internet_access":          np.where(internet_access == 1, "yes", "no"),
        "has_laptop":               np.where(has_laptop == 1, "yes", "no"),
        "sleep_hours":              sleep_hours,
        "stress_level":             stress_level,
        "motivation_score":         motivation_score,
        "reading_score":            reading_score,
        "writing_score":            writing_score,
        "math_score":               math_score,
        "science_score":            science_score,
        "overall_gpa":              overall_gpa,
        "passed":                   passed,
    })

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating student academic performance dataset...")
    students = generate_students()
    out_path = OUTPUT_DIR / "students.csv"
    students.to_csv(out_path, index=False)

    print(f"\nAll files saved to: {OUTPUT_DIR}")
    print("\nSummary:")
    print(f"  students.csv : {len(students):>8,} rows x {students.shape[1]} cols")
    print(f"\nColumn overview:")
    print(students.dtypes.to_string())
    print(f"\nFirst few rows:")
    print(students.head(3).to_string())
    print(f"\nNumeric summary:")
    print(students.describe(include="all").round(2).to_string())
    print(f"\nPass rate: {students['passed'].mean():.1%}")
    print(f"Mean GPA : {students['overall_gpa'].mean():.2f}")


if __name__ == "__main__":
    main()
