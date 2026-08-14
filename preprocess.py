"""
Cleaning + merging logic.

Two outputs:
  1. clean_resumes()   -> tidy resume dataframe (id, text, category)
  2. build_job_corpus() -> single job corpus with common columns:
        title, company, location, skills, description, experience, source
     pulled from Naukri + LinkedIn postings.
"""
import re
import html
import pandas as pd

from src.config import (
    MAX_JOBS_PER_SOURCE,
    LINKEDIN_SKILLS_CSV,
    LINKEDIN_SKILLS_MAPPING_CSV,
)
from src.data.load_data import (
    load_resumes,
    load_naukri_jobs,
    load_linkedin_jobs,
    load_linkedin_skills,
)

# ---------------------------------------------------------------------------
# Generic text cleaning
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_NONALNUM_RE = re.compile(r"[^a-z0-9\s\+\#\.\-]")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags/entities, collapse whitespace, drop junk chars.
    Keeps +, #, ., - since they matter in skill tokens (C++, C#, Node.js)."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = text.lower()
    text = _NONALNUM_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Resumes
# ---------------------------------------------------------------------------

def clean_resumes() -> pd.DataFrame:
    df = load_resumes()
    df = df.rename(columns={"ID": "id", "Resume_str": "resume_text", "Category": "category"})
    df = df[["id", "resume_text", "category"]].dropna(subset=["resume_text", "category"])
    df["resume_text_clean"] = df["resume_text"].apply(clean_text)
    df["category"] = df["category"].str.strip().str.title()
    df = df[df["resume_text_clean"].str.len() > 50].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Naukri
# ---------------------------------------------------------------------------

_EXPERIENCE_RE = re.compile(r"Work\s*Experience\s*(\d+)\s*-\s*(\d+)\s*Years", re.IGNORECASE)
_KEY_SKILLS_RE = re.compile(r"Key\s*Skills\s*(.+?)(?:$)", re.IGNORECASE | re.DOTALL)


def _extract_naukri_experience(desc: str) -> str:
    if not isinstance(desc, str):
        return ""
    m = _EXPERIENCE_RE.search(desc)
    return f"{m.group(1)}-{m.group(2)} years" if m else ""


def _extract_naukri_skills(desc: str) -> str:
    if not isinstance(desc, str):
        return ""
    m = _KEY_SKILLS_RE.search(desc)
    if not m:
        return ""
    raw = m.group(1)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return raw.strip()[:500]


def clean_naukri() -> pd.DataFrame:
    df = load_naukri_jobs()
    df["experience"] = df["job_description"].apply(_extract_naukri_experience)
    df["skills"] = df["job_description"].apply(_extract_naukri_skills)

    out = pd.DataFrame({
        "title": df.get("job_title"),
        "company": df.get("company_name"),
        "location": df.get("city").fillna("") + ", " + df.get("country").fillna(""),
        "skills": df["skills"],
        "description": df.get("job_description"),
        "experience": df["experience"],
        "source": "naukri",
    })
    out = out.dropna(subset=["title", "description"])
    if len(out) > MAX_JOBS_PER_SOURCE:
        out = out.sample(n=MAX_JOBS_PER_SOURCE, random_state=42)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# LinkedIn
# ---------------------------------------------------------------------------

def _build_linkedin_skills_lookup() -> pd.Series:
    """job_id -> comma-joined skill names."""
    skills = load_linkedin_skills()
    mapping = pd.read_csv(LINKEDIN_SKILLS_MAPPING_CSV)
    skills = skills.merge(mapping, on="skill_abr", how="left")
    grouped = skills.groupby("job_id")["skill_name"].apply(
        lambda s: ", ".join(sorted(set(s.dropna())))
    )
    return grouped


def clean_linkedin() -> pd.DataFrame:
    nrows = None
    df = load_linkedin_jobs(nrows=nrows)
    if len(df) > MAX_JOBS_PER_SOURCE:
        df = df.sample(n=MAX_JOBS_PER_SOURCE, random_state=42)

    skills_lookup = _build_linkedin_skills_lookup()
    df["skills"] = df["job_id"].map(skills_lookup).fillna("")

    out = pd.DataFrame({
        "title": df.get("title"),
        "company": df.get("company_name"),
        "location": df.get("location"),
        "skills": df["skills"],
        "description": df.get("description"),
        "experience": df.get("formatted_experience_level").fillna(""),
        "source": "linkedin",
    })
    out = out.dropna(subset=["title", "description"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Merge into one job corpus
# ---------------------------------------------------------------------------

def build_job_corpus() -> pd.DataFrame:
    naukri = clean_naukri()
    linkedin = clean_linkedin()
    corpus = pd.concat([naukri, linkedin], ignore_index=True)

    for col in ["title", "company", "location", "skills", "description", "experience"]:
        corpus[col] = corpus[col].fillna("").astype(str)

    corpus["description_clean"] = corpus["description"].apply(clean_text)
    corpus["skills_clean"] = corpus["skills"].apply(clean_text)
    corpus["search_text"] = (
        corpus["title"] + " " + corpus["skills_clean"] + " " + corpus["description_clean"]
    ).apply(clean_text)

    corpus = corpus[corpus["search_text"].str.len() > 30].reset_index(drop=True)
    corpus["job_id"] = corpus.index
    return corpus


if __name__ == "__main__":
    resumes = clean_resumes()
    print(f"Cleaned resumes: {len(resumes)} rows")
    print(resumes["category"].value_counts())

    corpus = build_job_corpus()
    print(f"\nJob corpus: {len(corpus)} rows")
    print(corpus["source"].value_counts())
    print(corpus.head(3))
