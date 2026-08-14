"""
Central configuration: paths, constants, and shared parameters.
Every other module imports paths from here instead of hardcoding strings.
"""
from pathlib import Path

# ---- Directory layout ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for d in [INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- Raw file locations ----
RESUME_CSV = RAW_DIR / "resume_dataset.csv"
NAUKRI_LDJSON = RAW_DIR / "naukri_jobs.ldjson"
LINKEDIN_POSTINGS_CSV = RAW_DIR / "linkedin_postings.csv"
LINKEDIN_SKILLS_CSV = RAW_DIR / "linkedin_job_skills.csv"
LINKEDIN_SKILLS_MAPPING_CSV = RAW_DIR / "linkedin_skills_mapping.csv"

# ---- Interim / processed outputs ----
JOB_CORPUS_PARQUET = PROCESSED_DIR / "job_corpus.parquet"
JOB_CORPUS_CSV = PROCESSED_DIR / "job_corpus_sample.csv"  # small sample for quick inspection
RESUME_CLEAN_PARQUET = PROCESSED_DIR / "resumes_clean.parquet"

# ---- Model artifacts ----
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
JOB_TFIDF_VECTORIZER_PATH = MODELS_DIR / "job_tfidf_vectorizer.pkl"
JOB_TFIDF_MATRIX_PATH = MODELS_DIR / "job_tfidf_matrix.pkl"

# ---- Params ----
RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE = (1, 2)
MIN_DF = 2

# Cap on how many LinkedIn/Naukri rows to keep in the merged job corpus.
# Full LinkedIn set alone is 120k+ rows; this keeps things fast for a 3-week
# project while still giving a rich corpus. Raise/remove once infra allows.
MAX_JOBS_PER_SOURCE = 40000
