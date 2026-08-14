"""
Loaders for the three raw datasets. Each function returns a pandas DataFrame
with source-specific columns intact -- cleaning/renaming happens in preprocess.py.
"""
import json
import pandas as pd
from src.config import RESUME_CSV, NAUKRI_LDJSON, LINKEDIN_POSTINGS_CSV, LINKEDIN_SKILLS_CSV


def load_resumes() -> pd.DataFrame:
    """Resume Dataset (Kaggle, Snehaan Bhawal): ID, Resume_str, Resume_html, Category."""
    df = pd.read_csv(RESUME_CSV)
    return df


def load_naukri_jobs() -> pd.DataFrame:
    """Naukri job listings, stored as line-delimited JSON (one JSON object per line)."""
    records = []
    with open(NAUKRI_LDJSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(records)
    return df


def load_linkedin_jobs(nrows: int = None) -> pd.DataFrame:
    """LinkedIn Job Postings 2023-2024. `nrows` lets callers sample during dev."""
    df = pd.read_csv(LINKEDIN_POSTINGS_CSV, low_memory=False, nrows=nrows)
    return df


def load_linkedin_skills() -> pd.DataFrame:
    """job_id -> skill_abr mapping, one row per (job, skill) pair. Needs aggregation."""
    df = pd.read_csv(LINKEDIN_SKILLS_CSV)
    return df


if __name__ == "__main__":
    resumes = load_resumes()
    print(f"Resumes: {len(resumes)} rows, categories={resumes['Category'].nunique()}")

    naukri = load_naukri_jobs()
    print(f"Naukri jobs: {len(naukri)} rows, columns={list(naukri.columns)}")

    linkedin = load_linkedin_jobs(nrows=5000)
    print(f"LinkedIn jobs (sample): {len(linkedin)} rows, columns={list(linkedin.columns)}")
