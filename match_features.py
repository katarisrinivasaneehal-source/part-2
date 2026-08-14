"""
Match features for a (resume, job) pair -- used by the optional fit/shortlisting
predictor (Model B). Combines skill overlap, a crude education-keyword match,
job-side experience signal, and text similarity (via the job TF-IDF space).
"""
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.models.skill_gap import extract_skills

_DEGREE_TERMS = ["bachelor", "master", "phd", "doctorate", "degree", "diploma", "mba"]
_EXPERIENCE_YEARS_RE = re.compile(r"(\d+)\s*(?:-|to)\s*(\d+)\s*years?")


def skill_overlap_features(resume_text_clean: str, job_skills_text: str) -> dict:
    resume_skills = extract_skills(resume_text_clean)
    job_skills = extract_skills(job_skills_text)
    if not resume_skills or not job_skills:
        overlap = 0
        jaccard = 0.0
    else:
        overlap = len(resume_skills & job_skills)
        jaccard = overlap / len(resume_skills | job_skills)
    return {
        "skill_overlap_count": overlap,
        "skill_jaccard": jaccard,
        "resume_skill_count": len(resume_skills),
        "job_skill_count": len(job_skills),
    }


def education_match_feature(resume_text_clean: str, job_text_clean: str) -> int:
    resume_has_degree = any(term in resume_text_clean for term in _DEGREE_TERMS)
    job_wants_degree = any(term in job_text_clean for term in _DEGREE_TERMS)
    return int(resume_has_degree and job_wants_degree)


def experience_required_years(job_experience_text: str) -> float:
    """Rough midpoint of a 'X-Y years' style requirement; 0 if not found."""
    if not isinstance(job_experience_text, str):
        return 0.0
    m = _EXPERIENCE_YEARS_RE.search(job_experience_text.lower())
    if not m:
        return 0.0
    lo, hi = int(m.group(1)), int(m.group(2))
    return (lo + hi) / 2


def text_similarity(resume_vec, job_vec) -> float:
    """Cosine similarity between two pre-transformed TF-IDF vectors (job vector space)."""
    return float(cosine_similarity(resume_vec, job_vec)[0][0])


def build_feature_row(resume_text_clean, job_row, resume_vec, job_vec) -> dict:
    features = {}
    features.update(skill_overlap_features(resume_text_clean, job_row["skills_clean"]))
    features["education_match"] = education_match_feature(
        resume_text_clean, job_row["description_clean"]
    )
    features["job_experience_years"] = experience_required_years(job_row["experience"])
    features["text_similarity"] = text_similarity(resume_vec, job_vec)
    return features
