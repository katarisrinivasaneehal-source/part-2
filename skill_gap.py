"""
Insight -- Skill-Gap Report.

For a candidate resume:
  1. Find the resume's nearest job cluster (via the clustering model).
  2. Extract the cluster's top skill terms (from the job corpus `skills_clean`
     column, restricted to that cluster -- not the noisier full-description
     vocabulary the clustering itself was fit on).
  3. Extract candidate skills from the resume text via a curated skill vocabulary.
  4. Report: skills the candidate already has (matched), and skills they're missing
     (the gap) that are common in their target cluster.
"""
import re
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from src.config import JOB_CORPUS_PARQUET, MODELS_DIR
from src.data.preprocess import clean_text

SVD_PATH = MODELS_DIR / "job_svd.pkl"
KMEANS_PATH = MODELS_DIR / "job_kmeans.pkl"
CLUSTER_ASSIGNMENTS_PATH = MODELS_DIR / "job_clusters.parquet"

# A curated, reasonably broad technical + professional skill vocabulary.
# Kept as a flat list (not per-category) since resumes span 24 very different
# domains (Chef, Aviation, Accountant, IT...) -- a single shared vocabulary
# with substring matching generalizes better than 24 hand-tuned lists.
SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "nosql",
    "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
    "spring", "hibernate", "aws", "azure", "gcp", "docker", "kubernetes",
    "git", "linux", "excel", "powerpoint", "word", "tableau", "power bi",
    "machine learning", "deep learning", "nlp", "data analysis", "data science",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "project management", "agile", "scrum", "jira", "leadership", "communication",
    "customer service", "sales", "marketing", "digital marketing", "seo",
    "social media", "content writing", "accounting", "auditing", "taxation",
    "bookkeeping", "financial analysis", "budgeting", "quickbooks", "sap",
    "recruitment", "onboarding", "payroll", "hr policies", "employee relations",
    "nursing", "patient care", "clinical", "phlebotomy", "cpr",
    "teaching", "curriculum development", "classroom management",
    "civil engineering", "mechanical engineering", "electrical engineering",
    "autocad", "solidworks", "construction management", "welding",
    "food safety", "menu planning", "inventory management", "logistics",
    "supply chain", "quality control", "manufacturing", "cnc",
    "legal research", "contract drafting", "litigation", "compliance",
    "public relations", "graphic design", "adobe photoshop", "illustrator",
    "figma", "ui/ux", "video editing", "photography",
    "network security", "cybersecurity", "cloud computing", "devops",
    "salesforce", "crm", "negotiation", "public speaking", "event planning",
]


_SKILL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(SKILL_VOCAB, key=len, reverse=True)) + r")\b"
)


def extract_skills(text_clean: str, vocab=SKILL_VOCAB) -> set:
    """Single compiled alternation regex instead of looping per-skill --
    ~100x fewer regex engine invocations, matters when called per (resume, job) pair."""
    if not isinstance(text_clean, str) or not text_clean:
        return set()
    return set(_SKILL_PATTERN.findall(text_clean))


def _top_cluster_skills(cluster_id: int, top_n: int = 15) -> list:
    corpus = pd.read_parquet(JOB_CORPUS_PARQUET)
    assignments = pd.read_parquet(CLUSTER_ASSIGNMENTS_PATH)
    corpus = corpus.merge(assignments[["job_id", "cluster"]], on="job_id")

    cluster_jobs = corpus[corpus["cluster"] == cluster_id]

    # Prefer title + description over the scraped `skills_clean` field: a
    # sizeable share of Naukri postings share templated/mismatched "Key
    # Skills" sections (a scraping artifact -- generic terms like
    # "manufacturing"/"sales" dominate regardless of the actual posting),
    # which drowned out real signal for e.g. nursing/healthcare clusters.
    # Title + description is noisier per-row but far more reliable in
    # aggregate across thousands of postings.
    counter = Counter()
    combined_text = (cluster_jobs["title"].fillna("") + " " + cluster_jobs["description_clean"].fillna(""))
    sample_n = min(1500, len(cluster_jobs))
    for text in combined_text.sample(sample_n, random_state=42):
        counter.update(extract_skills(clean_text(text) if not isinstance(text, str) else text))

    return [skill for skill, _ in counter.most_common(top_n)]


def predict_cluster(resume_text_clean: str) -> int:
    from src.models.recommender import _load_artifacts
    vectorizer, _, _ = _load_artifacts()
    svd = joblib.load(SVD_PATH)
    kmeans = joblib.load(KMEANS_PATH)

    vec = vectorizer.transform([resume_text_clean])
    reduced = svd.transform(vec)
    cluster = kmeans.predict(reduced)[0]
    return int(cluster)


def skill_gap_report(resume_text_clean: str, top_n: int = 15) -> dict:
    candidate_skills = extract_skills(resume_text_clean)
    cluster_id = predict_cluster(resume_text_clean)
    target_skills = _top_cluster_skills(cluster_id, top_n=top_n)

    matched = sorted(candidate_skills & set(target_skills))
    missing = [s for s in target_skills if s not in candidate_skills]

    return {
        "cluster_id": cluster_id,
        "candidate_skills": sorted(candidate_skills),
        "target_cluster_top_skills": target_skills,
        "matched_skills": matched,
        "missing_skills": missing,
    }


if __name__ == "__main__":
    resumes = pd.read_parquet("data/processed/resumes_clean.parquet")
    sample = resumes.iloc[0]
    print(f"Sample resume category: {sample['category']}")

    report = skill_gap_report(sample["resume_text_clean"])
    print(f"\nAssigned cluster: {report['cluster_id']}")
    print(f"Candidate skills found: {report['candidate_skills']}")
    print(f"Target cluster top skills: {report['target_cluster_top_skills']}")
    print(f"\nMatched (candidate already has): {report['matched_skills']}")
    print(f"Missing (skill gap): {report['missing_skills']}")
