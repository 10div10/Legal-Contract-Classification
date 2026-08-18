from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

RANDOM_STATE = 42


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=30_000,
    )


def word_char_tfidf() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=25_000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=20_000,
                ),
            ),
        ]
    )


def scorers() -> dict[str, object]:
    return {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(
            precision_score, average="macro", zero_division=0
        ),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "f1_weighted": make_scorer(
            f1_score, average="weighted", zero_division=0
        ),
    }


def evaluate_model(name: str, model: object, x: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> dict:
    pipeline = Pipeline([("tfidf", word_tfidf()), ("model", model)])
    scores = cross_validate(pipeline, x, y, cv=cv, scoring=scorers(), n_jobs=1)

    row = {"model": name, "features": "word TF-IDF"}
    for metric in scorers():
        values = scores[f"test_{metric}"]
        row[metric] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values))
    return row


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "legal_contracts_7class.csv"
    results_dir = project_root / "results"
    models_dir = project_root / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    x = df["text"].fillna("").astype(str).to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"].astype(str))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Multinomial NB": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", C=4.0, random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(class_weight="balanced", C=1.0, random_state=RANDOM_STATE),
        "SGD Classifier": SGDClassifier(
            loss="modified_huber",
            class_weight="balanced",
            max_iter=3000,
            tol=1e-4,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    rows = []
    for name, model in models.items():
        print(f"Evaluating {name}...")
        rows.append(evaluate_model(name, model, x, y, cv))

    enhanced = Pipeline(
        [
            ("features", word_char_tfidf()),
            ("model", LinearSVC(class_weight="balanced", C=1.0, random_state=RANDOM_STATE)),
        ]
    )
    enhanced_scores = cross_validate(
        enhanced, x, y, cv=cv, scoring=scorers(), n_jobs=1
    )
    enhanced_row = {
        "model": "Linear SVM + word/char TF-IDF",
        "features": "word + char TF-IDF",
    }
    for metric in scorers():
        values = enhanced_scores[f"test_{metric}"]
        enhanced_row[metric] = float(np.mean(values))
        enhanced_row[f"{metric}_std"] = float(np.std(values))
    rows.append(enhanced_row)

    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)

    # Out-of-fold report for the best baseline.
    oof_pred = cross_val_predict(enhanced, x, y, cv=cv, n_jobs=1)
    report = classification_report(
        y,
        oof_pred,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        results_dir / "best_model_classification_report.csv"
    )

    matrix = confusion_matrix(y, oof_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(matrix, display_labels=encoder.classes_).plot(
        ax=ax, xticks_rotation=45, values_format="d", colorbar=False
    )
    ax.set_title("5-fold out-of-fold confusion matrix: Linear SVM")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Fit deployable model on the full benchmark dataset using string labels.
    deploy_model = Pipeline(
        [
            ("features", word_char_tfidf()),
            ("model", LinearSVC(class_weight="balanced", C=1.0, random_state=RANDOM_STATE)),
        ]
    )
    deploy_model.fit(x, df["label"].astype(str).to_numpy())
    joblib.dump(deploy_model, models_dir / "legal_contract_classifier.joblib")

    metadata = {
        "n_samples": int(len(df)),
        "classes": sorted(df["label"].unique().tolist()),
        "cv": "Stratified 5-fold, shuffled, random_state=42",
        "best_model": "Linear SVM + word/char TF-IDF",
        "best_cv_accuracy_mean": enhanced_row["accuracy"],
        "best_cv_accuracy_std": enhanced_row["accuracy_std"],
        "best_cv_macro_f1_mean": enhanced_row["f1_macro"],
        "best_cv_macro_f1_std": enhanced_row["f1_macro_std"],
    }
    (results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    display_cols = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]
    print("\nMODEL COMPARISON")
    print(comparison[display_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
