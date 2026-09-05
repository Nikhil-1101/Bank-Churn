"""Model explainability: feature importance, SHAP values, partial dependence.

Run after train_models.py. Produces:
  - outputs/figures/feature_importance.png
  - outputs/figures/shap_summary.png
  - outputs/figures/pdp_<feature>.png (top drivers)
  - outputs/reports/feature_importance.csv
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
import shap
from sklearn.inspection import PartialDependenceDisplay

from data_pipeline import build_preprocessor, load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "European_Bank.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(MODELS_DIR, "best_model_name.json")) as f:
        best_name = json.load(f)["best_model"]

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X, y = load_dataset(DATA_PATH)
    # Use a sample for SHAP to keep runtime reasonable
    sample = X.sample(n=min(1500, len(X)), random_state=42)
    X_sample_transformed = preprocessor.transform(sample)
    if hasattr(X_sample_transformed, "toarray"):
        X_sample_transformed = X_sample_transformed.toarray()
    X_sample_df = pd.DataFrame(X_sample_transformed, columns=feature_names)

    # ---- Native feature importance ----
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = None

    if importances is not None:
        imp_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        imp_df.to_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"), index=False)

        plt.figure(figsize=(8, 6))
        top = imp_df.head(15).iloc[::-1]
        plt.barh(top["feature"], top["importance"], color="#2563eb")
        plt.title(f"Feature Importance — {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
        plt.close()
        print("Saved feature_importance.png / .csv")

    # ---- SHAP values ----
    print("Computing SHAP values (this can take a minute)...")
    try:
        if hasattr(model, "predict_proba") and model.__class__.__name__ in (
            "XGBClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "DecisionTreeClassifier",
        ):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            background = shap.sample(X_sample_df, 100, random_state=42)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_sample_df)

        shap.summary_plot(shap_values, X_sample_df, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved shap_summary.png")

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_imp_df = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_imp_df.to_csv(os.path.join(REPORTS_DIR, "shap_importance.csv"), index=False)
    except Exception as e:
        print(f"SHAP computation skipped due to error: {e}")

    # ---- Partial dependence plots for top drivers ----
    # Binary features (e.g. IsActiveMember) trip sklearn's percentile-grid PDP computation
    # when there are only two unique values, so we stick to continuous/near-continuous ones.
    top_features_for_pdp = [
        f for f in ["Age", "NumOfProducts", "Balance", "EngagementProductScore"]
        if f in X.columns
    ][:4]

    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        PartialDependenceDisplay.from_estimator(
            pipeline, X, top_features_for_pdp, ax=ax, n_jobs=-1
        )
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "partial_dependence.png"), dpi=150)
        plt.close()
        print("Saved partial_dependence.png")
    except Exception as e:
        print(f"PDP computation skipped due to error: {e}")


if __name__ == "__main__":
    main()
