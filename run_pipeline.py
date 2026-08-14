"""Run the cleaning + merge pipeline once and persist outputs to data/processed/."""
from src.config import RESUME_CLEAN_PARQUET, JOB_CORPUS_PARQUET, JOB_CORPUS_CSV
from src.data.preprocess import clean_resumes, build_job_corpus


def main():
    print("Cleaning resumes...")
    resumes = clean_resumes()
    resumes.to_parquet(RESUME_CLEAN_PARQUET, index=False)
    print(f"  saved {len(resumes)} rows -> {RESUME_CLEAN_PARQUET}")

    print("Building job corpus...")
    corpus = build_job_corpus()
    corpus.to_parquet(JOB_CORPUS_PARQUET, index=False)
    corpus.sample(min(2000, len(corpus)), random_state=42).to_csv(JOB_CORPUS_CSV, index=False)
    print(f"  saved {len(corpus)} rows -> {JOB_CORPUS_PARQUET}")
    print(f"  sample csv -> {JOB_CORPUS_CSV}")


if __name__ == "__main__":
    main()
