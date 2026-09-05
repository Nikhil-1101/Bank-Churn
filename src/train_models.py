"""Train, evaluate, and select the best churn model.

Trains a baseline (Logistic Regression), tree-based models (Decision Tree,
Random Forest), and boosted models (Gradient Boosting, XGBoost) on a
stratified train/test split, scores them on Accuracy / Precision / Recall /
F1 / ROC-AUC (+ 5-fold CV ROC-AUC), and persists the best pipeline plus all
metrics and evaluation artifacts for the Streamlit app and the research
paper.
"""

import json
import os

from openmp_bootstrap import ensure_openmp

ensure_openmp()

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data_pipeline import build_preprocessor, get_feature_names, load_dataset

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"XGBoost unavailable ({e}); continuing without it (it is optional per spec).")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "European_Bank.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
RANDOM_STATE = 42


def build_models(scale_pos_weight: float) -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return models


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading and engineering features...")
    X, y = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.3f}, Test churn rate: {y_test.mean():.3f}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = build_models(scale_pos_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    fitted_pipelines = {}
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    for name, model in models.items():
        print(f"Training {name}...")
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)

        cv_scores = cross_val_score(
            Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", models_fresh(name, scale_pos_weight))]),
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        metrics["cv_roc_auc_mean"] = cv_scores.mean()
        metrics["cv_roc_auc_std"] = cv_scores.std()
        metrics["model"] = name
        results.append(metrics)
        fitted_pipelines[name] = pipeline

        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=name)
        print(f"  {name}: {metrics}")

    ax.set_title("ROC Curves — Churn Prediction Models")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    results_df = pd.DataFrame(results).set_index("model")
    results_df = results_df[
        ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc_mean", "cv_roc_auc_std"]
    ].sort_values("roc_auc", ascending=False)
    results_df.to_csv(os.path.join(REPORTS_DIR, "model_metrics.csv"))
    print("\n=== Model comparison (sorted by test ROC-AUC) ===")
    print(results_df.round(4))

    best_name = results_df.index[0]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "best_model.joblib"))
    with open(os.path.join(MODELS_DIR, "best_model_name.json"), "w") as f:
        json.dump({"best_model": best_name}, f)

    # Confusion matrix for the best model
    fig, ax_cm = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(
        best_pipeline, X_test, y_test, display_labels=["Retained", "Churned"], ax=ax_cm, cmap="Blues"
    )
    ax_cm.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_best.png"), dpi=150)
    plt.close()

    # Persist test set + predictions for the Streamlit app / research paper
    feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"])
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    X_test_out = X_test.copy()
    X_test_out["y_true"] = y_test.values
    X_test_out["y_proba"] = best_pipeline.predict_proba(X_test)[:, 1]
    X_test_out["y_pred"] = best_pipeline.predict(X_test)
    X_test_out.to_csv(os.path.join(REPORTS_DIR, "test_predictions.csv"), index=False)

    print("\nArtifacts saved:")
    print(f"  - {MODELS_DIR}/best_model.joblib")
    print(f"  - {REPORTS_DIR}/model_metrics.csv")
    print(f"  - {REPORTS_DIR}/test_predictions.csv")
    print(f"  - {FIGURES_DIR}/roc_curves.png")
    print(f"  - {FIGURES_DIR}/confusion_matrix_best.png")


def models_fresh(name: str, scale_pos_weight: float):
    """Return a fresh, unfitted estimator instance for cross-validation."""
    return build_models(scale_pos_weight)[name]


if __name__ == "__main__":
    main()
