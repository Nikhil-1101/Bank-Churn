"""Data loading, cleaning, and feature engineering for the churn pipeline."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_COLUMNS_TO_DROP = ["Year", "CustomerId", "Surname"]

NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "BalanceSalaryRatio",
    "ProductDensity",
    "EngagementProductScore",
    "AgeTenureInteraction",
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
PASSTHROUGH_BINARY_FEATURES = ["HasCrCard", "IsActiveMember", "IsZeroBalance"]
TARGET = "Exited"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and drop non-informative identifier columns."""
    df = df.copy()

    df = df.drop(columns=[c for c in RAW_COLUMNS_TO_DROP if c in df.columns])

    # Balance-to-Salary ratio: how much of a customer's earning power sits idle in the bank
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

    # Product density: products held relative to relationship length
    df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    # Engagement-product interaction: active members who also hold many products are stickier
    df["EngagementProductScore"] = df["IsActiveMember"] * df["NumOfProducts"]

    # Age-tenure interaction: captures customers who joined late in life vs. long-standing ones
    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

    # Zero-balance flag: known strong churn signal independent of the raw balance magnitude
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("bin", "passthrough", PASSTHROUGH_BINARY_FEATURES),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    num_names = NUMERIC_FEATURES
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    bin_names = PASSTHROUGH_BINARY_FEATURES
    return num_names + cat_names + bin_names


def load_dataset(path: str):
    """Convenience loader: raw -> engineered -> (X, y)."""
    df = load_raw(path)
    df = engineer_features(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
