"""
preprocess.py
=============
Builds the sklearn preprocessing pipeline and prepares train/test splits.
All feature engineering and encoding happens INSIDE the pipeline so that
no information leaks from the test set into the training set.
"""

import hashlib
import pathlib
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
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
GENDER_VALUES = ["Male", "Female"]
JOB_TITLE_VALUES = [
    "Software Engineer", "Data Scientist", "Data Analyst", "DevOps Engineer",
    "Product Manager", "Business Analyst", "UI/UX Designer", "HR Executive",
    "Marketing Manager", "Sales Executive",
]
INDUSTRY_VALUES = ["IT", "Finance", "Healthcare", "EdTech", "E-commerce", "Manufacturing"]
CITY_VALUES = ["Bangalore", "Delhi", "Chennai", "Hyderabad", "Pune", "Mumbai", "Remote"]
COMPANY_SIZE_VALUES = ["Startup", "Mid-size", "Enterprise"]
WORK_MODE_VALUES = ["On-site", "Hybrid", "Remote"]

# Raw input schema
INPUT_FEATURES = [
    "Age", "Gender", "Education_Level", "Years_Experience", "Job_Title",
    "Job_Level", "Industry", "City", "Company_Size", "Work_Mode",
    "Certifications_Count", "Skills_Count",
]
CATEGORICAL_FEATURES = [
    "Gender", "Education_Level", "Job_Title", "Job_Level", "Industry",
    "City", "Company_Size", "Work_Mode",
]
INTEGER_FEATURES = ["Age", "Years_Experience", "Certifications_Count", "Skills_Count"]

# Feature groups
# Age is represented by Age_minus_Exp below. Keeping both Age and
# Age_minus_Exp would create an exact linear dependency because
# Age_minus_Exp = Age - Years_Experience.
NUMERIC_FEATURES = ["Years_Experience", "Certifications_Count", "Skills_Count"]
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
CATEGORY_VALUES = {
    "Gender": GENDER_VALUES,
    "Education_Level": EDUCATION_ORDER,
    "Job_Title": JOB_TITLE_VALUES,
    "Job_Level": JOB_LEVEL_ORDER,
    "Industry": INDUSTRY_VALUES,
    "City": CITY_VALUES,
    "Company_Size": COMPANY_SIZE_VALUES,
    "Work_Mode": WORK_MODE_VALUES,
}
INTEGER_RANGES = {
    "Age": (21, 57),
    "Years_Experience": (0, 35),
    "Certifications_Count": (0, 5),
    "Skills_Count": (2, 14),
}


# ── Custom transformer: feature engineering ───────────────────────────────────
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds three engineered columns to the DataFrame:
      - Age_minus_Exp      : Age − Years_Experience (years before first job)
      - Skills_per_Cert    : Skills_Count / (Certifications_Count + 1)
      - Exp_x_JobLevel     : Years_Experience × 1-based ordinal(Job_Level)
    Returns a new DataFrame; does not modify the input.
    """

    # 1-based index ensures Junior (level 1) experience is not nullified to 0
    _job_level_map = {lv: i + 1 for i, lv in enumerate(JOB_LEVEL_ORDER)}

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

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = INPUT_FEATURES
        return list(input_features) + ENGINEERED_FEATURES


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
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
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
    """Load and validate the raw salary CSV."""
    df = pd.read_csv(path)
    expected = set(INPUT_FEATURES + [TARGET])
    missing = sorted(expected - set(df.columns))
    extra = sorted(set(df.columns) - expected)
    if missing or extra:
        raise ValueError(f"Dataset columns do not match schema. Missing={missing}; extra={extra}")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")
    if df.duplicated().any():
        raise ValueError("Dataset contains duplicate rows.")
    for feature, (lower, upper) in INTEGER_RANGES.items():
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise ValueError(f"Dataset feature {feature} must be numeric.")
        if not df[feature].between(lower, upper).all():
            raise ValueError(f"Dataset feature {feature} is outside [{lower}, {upper}].")
    if not pd.api.types.is_numeric_dtype(df[TARGET]) or (df[TARGET] <= 0).any():
        raise ValueError(f"Dataset target {TARGET} must contain positive numeric values.")
    for feature, allowed in CATEGORY_VALUES.items():
        unknown = sorted(set(df[feature].unique()) - set(allowed))
        if unknown:
            raise ValueError(f"Dataset feature {feature} contains unknown values: {unknown}")
    return df


def data_sha256(path: pathlib.Path = DATA_PATH) -> str:
    """Return a stable hash for detecting stale model artifacts."""
    return file_sha256(path)


def file_sha256(path: pathlib.Path) -> str:
    """Return a SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Extracts feature names from a fitted pipeline or preprocessor for interpretability.
    """
    if hasattr(pipeline, "regressor_"):
        pipeline = pipeline.regressor_
    if "preprocess" in getattr(pipeline, "named_steps", {}):
        pipeline = pipeline.named_steps["preprocess"]
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
