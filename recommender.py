"""
Unsupervised Core Engine -- Job Recommendation.

Vectorizes the whole job corpus with TF-IDF once (fit + save), then for any
resume, transforms it into the same vector space and ranks jobs by cosine
similarity. This is the heart of the portal's "search" feature.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    JOB_CORPUS_PARQUET,
    JOB_TFIDF_VECTORIZER_PATH,
    JOB_TFIDF_MATRIX_PATH,
    TFIDF_MAX_FEATURES,
)
from src.features.text_features import build_tfidf_vectorizer


def fit_job_vectorizer():
    """Fit TF-IDF on the full job corpus and persist vectorizer + matrix."""
    corpus = pd.read_parquet(JOB_CORPUS_PARQUET)
    vectorizer = build_tfidf_vectorizer(max_features=TFIDF_MAX_FEATURES)
    matrix = vectorizer.fit_transform(corpus["search_text"])

    joblib.dump(vectorizer, JOB_TFIDF_VECTORIZER_PATH)
    joblib.dump(matrix, JOB_TFIDF_MATRIX_PATH)
    print(f"Fitted job TF-IDF: {matrix.shape[0]} jobs x {matrix.shape[1]} features")
    print(f"Saved -> {JOB_TFIDF_VECTORIZER_PATH}, {JOB_TFIDF_MATRIX_PATH}")
    return vectorizer, matrix


_ARTIFACT_CACHE = {}


def _load_artifacts():
    """Cache vectorizer/matrix/corpus in-process -- these are read on every
    recommendation call (e.g. from the Streamlit app), and reloading ~90MB
    of pickled matrix from disk each time is the dominant cost."""
    if not _ARTIFACT_CACHE:
        _ARTIFACT_CACHE["vectorizer"] = joblib.load(JOB_TFIDF_VECTORIZER_PATH)
        _ARTIFACT_CACHE["matrix"] = joblib.load(JOB_TFIDF_MATRIX_PATH)
        _ARTIFACT_CACHE["corpus"] = pd.read_parquet(JOB_CORPUS_PARQUET)
    return _ARTIFACT_CACHE["vectorizer"], _ARTIFACT_CACHE["matrix"], _ARTIFACT_CACHE["corpus"]


def recommend_jobs(resume_text_clean: str, top_n: int = 10) -> pd.DataFrame:
    """Return top-N matching jobs for a cleaned resume text, ranked by cosine similarity."""
    vectorizer, matrix, corpus = _load_artifacts()

    resume_vec = vectorizer.transform([resume_text_clean])
    sims = cosine_similarity(resume_vec, matrix).flatten()

    top_idx = np.argsort(sims)[::-1][:top_n]
    results = corpus.iloc[top_idx].copy()
    results["match_score"] = sims[top_idx]
    cols = ["job_id", "title", "company", "location", "skills", "experience",
            "source", "match_score"]
    return results[cols].reset_index(drop=True)


def precision_at_k(resume_category: str, resume_text_clean: str, k: int = 10) -> float:
    """
    Qualitative/quantitative recommender check: of the top-K recommended jobs,
    what fraction have a title that plausibly matches the resume's known category?
    Uses simple keyword overlap between category name and job title as a proxy
    label, since the job corpus has no ground-truth category field.
    """
    recs = recommend_jobs(resume_text_clean, top_n=k)
    category_tokens = set(resume_category.lower().replace("-", " ").split())
    hits = 0
    for title in recs["title"].fillna(""):
        title_tokens = set(title.lower().split())
        if category_tokens & title_tokens:
            hits += 1
    return hits / k


if __name__ == "__main__":
    fit_job_vectorizer()

    # quick smoke test using a resume from the training set
    resumes = pd.read_parquet("data/processed/resumes_clean.parquet")
    sample = resumes.iloc[0]
    print(f"\nSample resume category: {sample['category']}")
    recs = recommend_jobs(sample["resume_text_clean"], top_n=5)
    print(recs[["title", "company", "match_score"]])

    p_at_10 = precision_at_k(sample["category"], sample["resume_text_clean"], k=10)
    print(f"\nPrecision@10 (keyword-overlap proxy): {p_at_10:.2f}")
