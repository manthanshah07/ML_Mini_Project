"""
preprocess.py
=============
Builds the sklearn preprocessing pipeline and prepares train/test splits.
All feature engineering and encoding happens INSIDE the pipeline so that
no information leaks from the test set into the training set.
"""

import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.base import BaseEstimator, TransformerMixin

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "salary.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Ordered categories for ordinal features
EDUCATION_ORDER = ["High School", "Bachelor's", "Master's", "PhD"]
JOB_LEVEL_ORDER = ["Junior", "Mid", "Senior", "Lead", "Manager"]

# Feature groups
NUMERIC_FEATURES = ["Age", "Years_Experience", "Certifications_Count", "Skills_Count"]
ENGINEERED_FEATURES = [
    "Age_minus_Exp",       # proxy for "years before starting work"
    "Skills_per_Cert",     # skills density per certification
    "Exp_x_JobLevel",      # numeric experience × ordinal job level
]
ORDINAL_FEATURES = ["Education_Level", "Job_Level"]
OHE_FEATURES = [
    "Gender", "Job_Title", "Industry", "City", "Company_Size", "Work_Mode"
]

TARGET = "Salary_INR"


# ── Custom transformer: feature engineering ───────────────────────────────────
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds three engineered columns to the DataFrame:
      - Age_minus_Exp      : Age − Years_Experience (years before first job)
      - Skills_per_Cert    : Skills_Count / (Certifications_Count + 1)
      - Exp_x_JobLevel     : Years_Experience × ordinal(Job_Level)
    Returns a new DataFrame; does not modify the input.
    """

    _job_level_map = {lv: i for i, lv in enumerate(JOB_LEVEL_ORDER)}

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self  # stateless

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["Age_minus_Exp"] = X["Age"] - X["Years_Experience"]
        X["Skills_per_Cert"] = X["Skills_Count"] / (X["Certifications_Count"] + 1)
        X["Exp_x_JobLevel"] = X["Years_Experience"] * X["Job_Level"].map(
            self._job_level_map
        ).fillna(0)
        return X


# ── Pipeline builders ─────────────────────────────────────────────────────────

def _make_column_transformer(scale_numeric: bool = False) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that handles:
      - Ordinal encoding for Education_Level and Job_Level
      - OHE for the remaining categoricals
      - Optional StandardScaler for numeric + engineered columns
        (required for linear models and SVR)

    Parameters
    ----------
    scale_numeric : bool
        If True, applies StandardScaler to numeric/engineered features.
    """
    all_numeric = NUMERIC_FEATURES + ENGINEERED_FEATURES

    numeric_pipe = (
        StandardScaler() if scale_numeric else "passthrough"
    )

    ct = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[EDUCATION_ORDER, JOB_LEVEL_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ORDINAL_FEATURES,
            ),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                OHE_FEATURES,
            ),
            (
                "numeric",
                numeric_pipe,
                all_numeric,
            ),
        ],
        remainder="drop",
    )
    return ct


def build_pipeline(scale_numeric: bool = False) -> Pipeline:
    """
    Returns a full sklearn Pipeline:
      FeatureEngineer → ColumnTransformer

    The pipeline accepts a raw DataFrame (with original column names)
    and returns a numeric feature matrix ready for a model.

    Parameters
    ----------
    scale_numeric : bool
        Pass True for linear / SVR models that need standardised inputs.
    """
    return Pipeline(
        steps=[
            ("engineer", FeatureEngineer()),
            ("preprocessor", _make_column_transformer(scale_numeric)),
        ]
    )


# ── Data loading & splitting ──────────────────────────────────────────────────

def load_data(path: pathlib.Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw salary CSV."""
    df = pd.read_csv(path)
    return df


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Extracts feature names from a fitted pipeline for interpretability.
    """
    ct: ColumnTransformer = pipeline.named_steps["preprocessor"]
    names = []
    for name, transformer, cols in ct.transformers_:
        if name == "ordinal":
            names.extend(cols)
        elif name == "ohe":
            names.extend(transformer.get_feature_names_out(cols))
        elif name == "numeric":
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out())
            else:
                all_numeric = NUMERIC_FEATURES + ENGINEERED_FEATURES
                names.extend(all_numeric)
    return names


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits df into (X_train, X_test, y_train, y_test).
    Returns the *original* INR salary as y; log-transformation is
    handled inside model-specific pipelines where needed.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Duplicates:", df.duplicated().sum())
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    pipe = build_pipeline(scale_numeric=False)
    X_tr = pipe.fit_transform(X_train)
    X_te = pipe.transform(X_test)
    print("Transformed train shape:", X_tr.shape)
    print("Feature names:", get_feature_names(pipe))
