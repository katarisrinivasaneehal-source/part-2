# SmartHire — Resume-to-Job Matching & Career Guidance Engine

A classical ML system that takes a resume, predicts its job category, ranks the best-matching
jobs from a merged job corpus, clusters jobs into role families, produces a skill-gap report,
and (optionally) scores fit/shortlisting likelihood — all via a Streamlit portal.
No LLMs / generative AI anywhere in the pipeline.

## Status: Complete (Weeks 1-3)

- [x] **Week 1** — Datasets downloaded & cleaned, merged job corpus, Model A (resume classifier)
- [x] **Week 2** — Job recommender (TF-IDF + cosine similarity), job clustering (K-Means),
      skill-gap report
- [x] **Week 3** — Model B (fit/shortlisting predictor, optional), Streamlit portal, final report

## Datasets

| Source | Rows | Notes |
|---|---|---|
| Resume Dataset (Kaggle, Snehaan Bhawal) | 2,483 (cleaned) | 24 job categories |
| Naukri Job Listings | ~30,000 | India-focused postings |
| LinkedIn Job Postings 2023-2024 | ~40,000 (sampled from 123k) | Global postings, incl. skills + salary |

Merged job corpus: **~70,000 postings** with common columns `title, company, location, skills,
description, experience, source`.

## Setup

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the raw Kaggle files in `data/raw/`:
- `resume_dataset.csv` (from Resume Dataset)
- `naukri_jobs.ldjson` (from Naukri Job Listings)
- `linkedin_postings.csv`, `linkedin_job_skills.csv`, `linkedin_skills_mapping.csv` (from LinkedIn Job Postings)

## Running the pipeline (in order)

```bash
python -m src.data.run_pipeline      # clean data, build job corpus -> data/processed/
python -m src.models.classifier      # train Model A (resume classifier) -> models/
python -m src.models.recommender     # fit job TF-IDF for the recommender -> models/
python -m src.models.clustering      # fit job clusters (K-Means) -> models/
python -m src.models.fit_predictor   # generate pairs + train Model B (optional) -> models/
```

Or run the notebooks in `notebooks/` in numeric order (01 -> 05) — each mirrors one script above
with EDA, extra visualizations, and commentary.

## Running the app

```bash
streamlit run app/streamlit_app.py
```

Upload a resume (PDF/DOCX/TXT) or paste text to see: predicted category, top matching jobs
(with match scores and, if Model B was trained, fit scores), and a skill-gap report.

## Running tests

```bash
pytest tests/ -v
```

## Model A — Resume Category Classifier

Compared Logistic Regression, Linear SVM, and Random Forest on TF-IDF features (8,000 features,
1-2 grams). **Random Forest** was selected as best by macro F1:

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Logistic Regression | 0.690 | 0.671 | 0.651 | 0.647 |
| Linear SVM | 0.718 | 0.702 | 0.680 | 0.675 |
| **Random Forest** | **0.765** | **0.799** | **0.731** | **0.734** |

See `reports/figures/confusion_matrix_classifier.png`.

## Job Recommender

TF-IDF (15,000 features) + cosine similarity over the full 70k-job corpus. Qualitatively strong
(e.g. HR resumes match HR management roles; Python/Django engineers match DevOps/Python roles).
A keyword-overlap Precision@10 proxy is included as a conservative lower-bound metric, since the
job corpus has no ground-truth category labels — see `notebooks/03_recommender.ipynb`.

## Job Clustering

K-Means (k=20, selected by silhouette score) on SVD-reduced (100-dim) job vectors. Produces
interpretable role families (nursing, Java dev, marketing, accounting, BPO, sales/retail, etc.)
— see `reports/figures/clustering_pca_scatter.png` and `notebooks/04_clustering_topics.ipynb`.

## Model B — Fit/Shortlisting Predictor (optional)

No public shortlisting-outcome dataset exists, so training labels are heuristic (category-title
keyword overlap). Logistic Regression beat XGBoost:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.689** | **0.707** | **0.644** | **0.674** | **0.768** |
| XGBoost | 0.678 | 0.708 | 0.606 | 0.653 | 0.750 |

Treat this module as a proof-of-concept, not production-ready — see `reports/final_report.docx`
Section 7 for the full discussion of its limitations (heuristic-label leakage, overconfident scores).

## Repository structure

See project brief Section 7 — this repo follows it exactly (`src/`, `notebooks/`, `models/`,
`app/`, `reports/`, `tests/`).

## Deliverables checklist

1. Cleaned datasets + EDA notebook — `notebooks/01_eda.ipynb`
2. Resume classifier + fit predictor (optional) with comparison tables — `notebooks/02, 05`
3. Recommender + clustering notebooks with visualizations — `notebooks/03, 04`
4. Working Streamlit demo — `app/streamlit_app.py`
5. Written report — `reports/final_report.docx`
6. Full Git repository following Section 7's structure — this repo

## Key decisions & notes

- Naukri postings lack clean structured skills/experience fields — both are regex-extracted from
  the free-text job description.
- LinkedIn skills come from a separate `job_skills.csv` keyed by `job_id`, joined against
  `skills.csv` for human-readable names.
- The job corpus caps each source at 40,000 rows (`MAX_JOBS_PER_SOURCE` in `src/config.py`) for
  speed; raise this later if infra allows.
- The skill-gap and fit-predictor modules share a curated ~100-term skill vocabulary
  (`src/models/skill_gap.py::SKILL_VOCAB`) rather than 24 category-specific lists, since a single
  shared vocabulary with substring matching generalizes better across very different resume domains.
- Model B's heuristic training labels are a known limitation — documented in the final report
  rather than hidden.
