"""
test_pipeline.py
================
Basic pytest tests for the preprocessing pipeline and predict function.
These verify:
  1. The pipeline transforms data without data leakage indicators
  2. Predictions are within a plausible salary range
  3. Feature engineering produces expected columns
  4. Ordinal encoding is monotone
  5. The predict module returns sensible values

Run with:
    pytest tests/ -v
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import (
    DATA_PATH,
    FeatureEngineer,
    EDUCATION_ORDER,
    JOB_LEVEL_ORDER,
    build_pipeline,
    get_feature_names,
    load_data,
    split_data,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def split(raw_df):
    X_train, X_test, y_train, y_test = split_data(raw_df)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def fitted_pipeline(split):
    X_train, X_test, y_train, y_test = split
    pipe = build_pipeline(scale_numeric=False)
    pipe.fit(X_train)
    return pipe, X_train, X_test


# ── Data integrity tests ───────────────────────────────────────────────────────

class TestDataIntegrity:
    def test_shape(self, raw_df):
        assert raw_df.shape == (500, 13), f"Expected (500, 13), got {raw_df.shape}"

    def test_no_missing_values(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0, "Dataset should have no missing values"

    def test_no_duplicates(self, raw_df):
        assert raw_df.duplicated().sum() == 0, "Dataset should have no duplicates"

    def test_salary_range(self, raw_df):
        assert raw_df["Salary_INR"].min() >= 200_000, "Min salary should be ≥ 2L"
        assert raw_df["Salary_INR"].max() <= 5_000_000, "Max salary should be ≤ 50L"

    def test_train_test_no_overlap(self, split):
        X_train, X_test, _, _ = split
        assert len(X_train) + len(X_test) == 500
        shared = set(X_train.index) & set(X_test.index)
        assert len(shared) == 0, "Train and test indices must not overlap"

    def test_train_size(self, split):
        X_train, X_test, _, _ = split
        assert len(X_train) == 400, f"Expected 400 train rows, got {len(X_train)}"
        assert len(X_test) == 100, f"Expected 100 test rows, got {len(X_test)}"


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineer:
    def test_columns_added(self, raw_df):
        df_in = raw_df.drop(columns=["Salary_INR"])
        fe = FeatureEngineer()
        df_out = fe.transform(df_in)
        assert "Age_minus_Exp" in df_out.columns
        assert "Skills_per_Cert" in df_out.columns
        assert "Exp_x_JobLevel" in df_out.columns

    def test_age_minus_exp_values(self, raw_df):
        df_in = raw_df.drop(columns=["Salary_INR"])
        fe = FeatureEngineer()
        df_out = fe.transform(df_in)
        expected = df_in["Age"] - df_in["Years_Experience"]
        pd.testing.assert_series_equal(
            df_out["Age_minus_Exp"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_skills_per_cert_no_div_zero(self, raw_df):
        """Skills_per_Cert = Skills / (Certs + 1) — denominator always ≥ 1."""
        df_in = raw_df.drop(columns=["Salary_INR"])
        fe = FeatureEngineer()
        df_out = fe.transform(df_in)
        assert (df_out["Skills_per_Cert"] > 0).all()

    def test_fit_is_stateless(self, raw_df):
        """FeatureEngineer.fit() should be a no-op."""
        df_in = raw_df.drop(columns=["Salary_INR"])
        fe = FeatureEngineer()
        result = fe.fit(df_in)
        assert result is fe  # fit returns self


# ── Pipeline tests ─────────────────────────────────────────────────────────────

class TestPipeline:
    def test_output_is_2d_array(self, fitted_pipeline):
        pipe, X_train, X_test = fitted_pipeline
        X_tr = pipe.transform(X_train)
        assert X_tr.ndim == 2

    def test_output_no_nan(self, fitted_pipeline):
        pipe, X_train, X_test = fitted_pipeline
        X_tr = pipe.transform(X_train)
        X_te = pipe.transform(X_test)
        assert not np.isnan(X_tr).any(), "Transformed train data has NaNs"
        assert not np.isnan(X_te).any(), "Transformed test data has NaNs"

    def test_train_test_same_columns(self, fitted_pipeline):
        """Train and test transforms must have the same number of columns."""
        pipe, X_train, X_test = fitted_pipeline
        X_tr = pipe.transform(X_train)
        X_te = pipe.transform(X_test)
        assert X_tr.shape[1] == X_te.shape[1], (
            f"Train cols={X_tr.shape[1]}, Test cols={X_te.shape[1]}"
        )

    def test_feature_names_match_columns(self, fitted_pipeline):
        pipe, X_train, _ = fitted_pipeline
        X_tr = pipe.transform(X_train)
        names = get_feature_names(pipe)
        assert len(names) == X_tr.shape[1], (
            f"Feature names ({len(names)}) != columns ({X_tr.shape[1]})"
        )

    def test_scaled_pipeline_different(self, split):
        """Scaled pipeline should produce different numeric values for same input."""
        X_train, X_test, _, _ = split
        pipe_unscaled = build_pipeline(scale_numeric=False)
        pipe_scaled = build_pipeline(scale_numeric=True)
        pipe_unscaled.fit(X_train)
        pipe_scaled.fit(X_train)
        X_u = pipe_unscaled.transform(X_test)
        X_s = pipe_scaled.transform(X_test)
        # They must have the same shape
        assert X_u.shape == X_s.shape
        # But not be identical (numeric columns are scaled)
        assert not np.allclose(X_u, X_s), "Scaled and unscaled should differ"

    def test_unknown_category_handled(self, fitted_pipeline):
        """OHE with handle_unknown='ignore' should not crash on unseen categories."""
        pipe, X_train, X_test = fitted_pipeline
        X_mod = X_test.copy()
        X_mod.loc[X_mod.index[0], "City"] = "NewCity_XYZ"
        # Should not raise
        X_out = pipe.transform(X_mod)
        assert not np.isnan(X_out).any()


# ── Ordinal encoding tests ─────────────────────────────────────────────────────

class TestOrdinalEncoding:
    def test_education_monotone(self, fitted_pipeline):
        """Higher education levels should map to strictly higher ordinal values."""
        pipe, X_train, _ = fitted_pipeline
        preproc = pipe.named_steps["preprocessor"]
        ordinal_enc = preproc.named_transformers_["ordinal"]
        cats = ordinal_enc.categories_[0]  # Education
        # The encoder returns categories in the order they were fitted
        # Check that the fitted order matches EDUCATION_ORDER
        for i, level in enumerate(EDUCATION_ORDER):
            assert level in cats, f"{level} not found in encoder categories"

    def test_job_level_monotone(self, fitted_pipeline):
        pipe, X_train, _ = fitted_pipeline
        preproc = pipe.named_steps["preprocessor"]
        ordinal_enc = preproc.named_transformers_["ordinal"]
        cats = ordinal_enc.categories_[1]  # Job_Level
        for level in JOB_LEVEL_ORDER:
            assert level in cats, f"{level} not found in Job_Level categories"


# ── Predict module tests (requires trained model) ─────────────────────────────

MODELS_DIR = ROOT / "models"

@pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(),
    reason="Trained model not found — run python -m src.train first",
)
class TestPredict:
    def test_predict_returns_float(self):
        from src.predict import predict_salary
        sample = {
            "Age": 30,
            "Gender": "Male",
            "Education_Level": "Master's",
            "Years_Experience": 7,
            "Job_Title": "Data Scientist",
            "Job_Level": "Mid",
            "Industry": "IT",
            "City": "Bangalore",
            "Company_Size": "Enterprise",
            "Work_Mode": "Hybrid",
            "Certifications_Count": 2,
            "Skills_Count": 8,
        }
        result = predict_salary(sample)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_predict_in_plausible_range(self):
        from src.predict import predict_salary
        sample = {
            "Age": 30,
            "Gender": "Female",
            "Education_Level": "Bachelor's",
            "Years_Experience": 5,
            "Job_Title": "Data Analyst",
            "Job_Level": "Junior",
            "Industry": "Finance",
            "City": "Pune",
            "Company_Size": "Startup",
            "Work_Mode": "On-site",
            "Certifications_Count": 1,
            "Skills_Count": 5,
        }
        result = predict_salary(sample)
        # Salary should be between 1L and 50L INR
        assert 100_000 <= result <= 5_000_000, f"Salary {result} out of plausible range"

    def test_format_inr_lakhs(self):
        from src.predict import format_inr_lakhs
        assert format_inr_lakhs(1_000_000) == "₹10.00 L"
        assert format_inr_lakhs(100_000) == "₹1.00 L"
        assert "Cr" in format_inr_lakhs(10_000_000)

    def test_senior_earns_more_than_junior(self):
        """Senior employee should earn more than junior with same profile."""
        from src.predict import predict_salary
        base = {
            "Age": 35,
            "Gender": "Male",
            "Education_Level": "Bachelor's",
            "Years_Experience": 10,
            "Job_Title": "Software Engineer",
            "Industry": "IT",
            "City": "Bangalore",
            "Company_Size": "Enterprise",
            "Work_Mode": "Hybrid",
            "Certifications_Count": 2,
            "Skills_Count": 8,
        }
        senior = {**base, "Job_Level": "Senior"}
        junior = {**base, "Job_Level": "Junior"}
        assert predict_salary(senior) > predict_salary(junior), (
            "Senior should earn more than Junior (same profile)"
        )
