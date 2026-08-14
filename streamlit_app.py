"""
SmartHire Streamlit portal.

Upload a resume -> see: predicted job category, top-N matching jobs (with an
optional fit score per job), and a skill-gap report.

Run with: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import streamlit as st

from src.config import CLASSIFIER_PATH, TFIDF_VECTORIZER_PATH, LABEL_ENCODER_PATH
from src.data.preprocess import clean_text
from src.parsing.resume_parser import extract_text
from src.models.recommender import recommend_jobs, _load_artifacts
from src.models.skill_gap import skill_gap_report
from src.models.fit_predictor import (
    FIT_PREDICTOR_PATH, FIT_SCALER_PATH, FEATURE_COLS,
)
from src.features.match_features import build_feature_row

st.set_page_config(page_title="SmartHire", page_icon="🎯", layout="wide")


@st.cache_resource
def load_classifier():
    model = joblib.load(CLASSIFIER_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, vectorizer, label_encoder


@st.cache_resource
def load_fit_predictor():
    if FIT_PREDICTOR_PATH.exists() and FIT_SCALER_PATH.exists():
        return joblib.load(FIT_PREDICTOR_PATH), joblib.load(FIT_SCALER_PATH)
    return None, None


@st.cache_resource
def load_job_artifacts():
    return _load_artifacts()  # vectorizer, matrix, corpus (cached inside recommender too)


def predict_category(resume_text_clean, model, vectorizer, label_encoder):
    X = vectorizer.transform([resume_text_clean])
    pred = model.predict(X)[0]
    return label_encoder.inverse_transform([pred])[0]


def score_fit(resume_text_clean, job_row, resume_vec_jobspace, job_matrix, fit_model, fit_scaler):
    job_vec = job_matrix[job_row["job_id"]]
    feats = build_feature_row(resume_text_clean, job_row, resume_vec_jobspace, job_vec)
    X = pd.DataFrame([feats])[FEATURE_COLS].fillna(0)
    if hasattr(fit_model, "predict_proba"):
        # LogisticRegression was trained on scaled features; XGBoost on raw.
        X_input = fit_scaler.transform(X) if fit_scaler is not None and _is_logreg(fit_model) else X
        return float(fit_model.predict_proba(X_input)[0][1])
    return None


def _is_logreg(model):
    return model.__class__.__name__ == "LogisticRegression"


def main():
    st.title("🎯 SmartHire")
    st.caption("Upload your resume to get matched jobs, a fit score, and a skill-gap report — "
               "powered by classical ML (TF-IDF, cosine similarity, K-Means, Logistic Regression / "
               "Random Forest). No LLMs involved.")

    uploaded = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

    if uploaded is None:
        st.info("Upload a PDF, DOCX, or TXT resume to get started.")
        return

    with st.spinner("Reading resume..."):
        try:
            raw_text = extract_text(uploaded.read(), uploaded.name)
        except Exception as e:
            st.error(f"Couldn't read this file: {e}")
            return

    resume_text_clean = clean_text(raw_text)
    if len(resume_text_clean) < 30:
        st.warning("Couldn't extract much text from this file — results may be unreliable. "
                   "Try a text-based PDF or DOCX (not a scanned image).")

    with st.expander("Extracted resume text (cleaned)"):
        st.text(resume_text_clean[:3000] + ("..." if len(resume_text_clean) > 3000 else ""))

    # ---- Model A: category classification ----
    clf_model, clf_vectorizer, label_encoder = load_classifier()
    predicted_category = predict_category(resume_text_clean, clf_model, clf_vectorizer, label_encoder)

    st.header("Predicted job category")
    st.success(f"**{predicted_category}**")

    top_n = st.slider("Number of job matches to show", min_value=5, max_value=25, value=10)

    # ---- Unsupervised: job recommender ----
    st.header("Top matching jobs")
    with st.spinner("Ranking jobs..."):
        recs = recommend_jobs(resume_text_clean, top_n=top_n)

    fit_model, fit_scaler = load_fit_predictor()
    job_vectorizer, job_matrix, job_corpus = load_job_artifacts()
    resume_vec_jobspace = job_vectorizer.transform([resume_text_clean])

    if fit_model is not None:
        fit_scores = []
        for _, row in recs.iterrows():
            job_row = job_corpus[job_corpus["job_id"] == row["job_id"]].iloc[0]
            score = score_fit(resume_text_clean, job_row, resume_vec_jobspace, job_matrix,
                               fit_model, fit_scaler)
            fit_scores.append(score)
        recs["fit_score"] = fit_scores

    display_cols = ["title", "company", "location", "match_score"]
    if "fit_score" in recs.columns:
        display_cols.append("fit_score")
    st.dataframe(
        recs[display_cols].style.format({
            "match_score": "{:.3f}",
            **({"fit_score": "{:.3f}"} if "fit_score" in recs.columns else {}),
        }),
        use_container_width=True,
    )
    st.caption("`match_score` = cosine similarity between resume and job text (TF-IDF space). "
               "`fit_score` (if shown) = predicted shortlist probability from Model B — note this "
               "model was trained on heuristic labels, treat it as directional, not definitive.")

    # ---- Unsupervised: skill-gap report ----
    st.header("Skill-gap report")
    with st.spinner("Analyzing skill gaps..."):
        try:
            report = skill_gap_report(resume_text_clean)
        except Exception as e:
            report = None
            st.warning(f"Couldn't generate a skill-gap report: {e}")

    if report:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Your detected skills")
            st.write(", ".join(report["candidate_skills"]) or "_None detected_")
        with col2:
            st.subheader("✅ Skills you have that match")
            st.write(", ".join(report["matched_skills"]) or "_None matched_")
        with col3:
            st.subheader("📈 Skills to develop")
            st.write(", ".join(report["missing_skills"]) or "_No gaps found_")

        st.caption(f"Based on your nearest job cluster (#{report['cluster_id']}) among "
                   f"{job_corpus['job_id'].nunique()} jobs in the corpus.")

    st.divider()
    st.caption("SmartHire — classical ML resume-to-job matching. Built with scikit-learn + Streamlit.")


if __name__ == "__main__":
    main()
