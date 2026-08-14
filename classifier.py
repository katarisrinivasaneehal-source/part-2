"""
Model A -- Resume Category Classifier.

Pipeline: clean text (already done in preprocess.py) -> TF-IDF -> multi-class
classifier. Trains Logistic Regression, Linear SVM, and Random Forest, compares
them, and persists the best one.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.config import (
    RESUME_CLEAN_PARQUET,
    CLASSIFIER_PATH,
    TFIDF_VECTORIZER_PATH,
    LABEL_ENCODER_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    FIGURES_DIR,
)
from src.features.text_features import build_tfidf_vectorizer


def load_training_data():
    df = pd.read_parquet(RESUME_CLEAN_PARQUET)
    return df["resume_text_clean"], df["category"]


def train_and_compare():
    X_text, y_raw = load_training_data()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = build_tfidf_vectorizer(max_features=8000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, C=5.0, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "linear_svm": LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = {}
    fitted_models = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )
        results[name] = {"accuracy": acc, "precision_macro": precision,
                          "recall_macro": recall, "f1_macro": f1}
        fitted_models[name] = model
        print(f"{name:20s} | acc={acc:.3f}  precision={precision:.3f}  "
              f"recall={recall:.3f}  f1={f1:.3f}")

    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")

    preds = best_model.predict(X_test)
    print("\nClassification report (best model):")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_, zero_division=0))

    _save_confusion_matrix(y_test, preds, label_encoder.classes_, best_name)

    joblib.dump(best_model, CLASSIFIER_PATH)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"\nSaved: {CLASSIFIER_PATH}, {TFIDF_VECTORIZER_PATH}, {LABEL_ENCODER_PATH}")

    return results, best_name


def _save_confusion_matrix(y_test, preds, class_names, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {model_name}")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path = FIGURES_DIR / "confusion_matrix_classifier.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {out_path}")


def predict_category(resume_text_clean: str):
    model = joblib.load(CLASSIFIER_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    X = vectorizer.transform([resume_text_clean])
    pred = model.predict(X)[0]
    return label_encoder.inverse_transform([pred])[0]


if __name__ == "__main__":
    train_and_compare()
