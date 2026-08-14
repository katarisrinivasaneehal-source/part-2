"""
Model B (optional) -- Shortlisting / Fit Predictor.

There's no ground-truth "was this candidate shortlisted for this job" dataset
available, so training labels are heuristic: a (resume, job) pair is labeled
"fit" (1) if the resume's category shares a keyword with the job title (the
same proxy used for the recommender's Precision@K check), else "not fit" (0).
This is a real limitation -- documented here and in the report -- but it lets
us demonstrate the full supervised pipeline (feature engineering, class
imbalance handling, LogReg vs XGBoost comparison, precision/recall/ROC-AUC)
that the brief asks for.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

from src.config import (
    JOB_CORPUS_PARQUET, RESUME_CLEAN_PARQUET, MODELS_DIR, FIGURES_DIR, RANDOM_STATE,
)
from src.models.recommender import _load_artifacts
from src.features.match_features import build_feature_row

FIT_PREDICTOR_PATH = MODELS_DIR / "fit_predictor.pkl"
FIT_SCALER_PATH = MODELS_DIR / "fit_predictor_scaler.pkl"
TRAINING_PAIRS_PATH = MODELS_DIR / "fit_predictor_training_pairs.parquet"

N_RESUMES = 180          # how many resumes to sample for pair generation
POS_PER_RESUME = 4       # candidate jobs whose title overlaps the resume's category
NEG_PER_RESUME = 4       # random jobs (mostly non-matching by construction)


def _heuristic_label(category: str, job_title: str) -> int:
    cat_tokens = set(category.lower().replace("-", " ").split())
    title_tokens = set(str(job_title).lower().split())
    return int(bool(cat_tokens & title_tokens))


def generate_training_pairs(seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    resumes = pd.read_parquet(RESUME_CLEAN_PARQUET)
    # Reuse the cached corpus from _load_artifacts (same object recommender.py
    # already loaded) instead of re-reading the parquet file a second time.
    vectorizer, job_matrix, corpus = _load_artifacts()
    # Parquet loads string columns as pyarrow-backed ArrowDtype by default,
    # whose boolean-mask `.take()` is dramatically slower than plain numpy
    # object arrays when called repeatedly (once per resume, in a loop).
    # Casting to object dtype up front pays off after ~2 resumes.
    str_cols = ["title", "skills_clean", "description_clean", "experience"]
    corpus = corpus.astype({c: "object" for c in str_cols if c in corpus.columns})

    resumes_sample = resumes.sample(n=min(N_RESUMES, len(resumes)), random_state=seed)

    rows = []
    for _, resume in resumes_sample.iterrows():
        cat_tokens = set(resume["category"].lower().replace("-", " ").split())
        title_mask = corpus["title"].fillna("").str.lower().apply(
            lambda t: bool(cat_tokens & set(t.split()))
        )
        candidates_pos = corpus[title_mask]
        candidates_neg_pool = corpus[~title_mask]

        pos = candidates_pos.sample(
            n=min(POS_PER_RESUME, len(candidates_pos)), random_state=seed
        ) if len(candidates_pos) > 0 else candidates_pos
        neg = candidates_neg_pool.sample(n=NEG_PER_RESUME, random_state=seed)

        resume_vec = vectorizer.transform([resume["resume_text_clean"]])

        for _, job in pd.concat([pos, neg]).iterrows():
            job_vec = job_matrix[job["job_id"]]
            feats = build_feature_row(
                resume["resume_text_clean"], job, resume_vec, job_vec
            )
            feats["label"] = _heuristic_label(resume["category"], job["title"])
            feats["resume_id"] = resume["id"]
            feats["job_id"] = job["job_id"]
            rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_parquet(TRAINING_PAIRS_PATH, index=False)
    print(f"Generated {len(df)} training pairs "
          f"({df['label'].sum()} positive, {len(df) - df['label'].sum()} negative)")
    return df


FEATURE_COLS = [
    "skill_overlap_count", "skill_jaccard", "resume_skill_count", "job_skill_count",
    "education_match", "job_experience_years", "text_similarity",
]


def train_and_compare(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_parquet(TRAINING_PAIRS_PATH)

    X = df[FEATURE_COLS].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        if name == "logistic_regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        fitted[name] = model
        r = results[name]
        print(f"{name:20s} | acc={r['accuracy']:.3f} prec={r['precision']:.3f} "
              f"recall={r['recall']:.3f} f1={r['f1']:.3f} roc_auc={r['roc_auc']:.3f}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_model, FIT_PREDICTOR_PATH)
    joblib.dump(scaler, FIT_SCALER_PATH)
    print(f"Saved -> {FIT_PREDICTOR_PATH}")

    _plot_roc_curves(models, fitted, X_test, X_test_scaled, y_test)

    return results, best_name


def _plot_roc_curves(models, fitted, X_test, X_test_scaled, y_test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(7, 6))
    for name in models:
        model = fitted[name]
        X = X_test_scaled if name == "logistic_regression" else X_test
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Fit Predictor -- ROC Curves")
    ax.legend()
    plt.tight_layout()
    out_path = FIGURES_DIR / "fit_predictor_roc.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    df = generate_training_pairs()
    train_and_compare(df)
