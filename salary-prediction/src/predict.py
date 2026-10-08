"""
predict.py
==========
Inference helper for the trained salary prediction model.

Provides:
  - predict_salary(input_dict) → predicted salary in INR
  - format_inr_lakhs(amount)   → Indian lakh-formatted string

Used by the Streamlit app and by tests.

Example usage:
    from src.predict import predict_salary
    salary = predict_salary({
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
    })
    print(f"Predicted: {format_inr_lakhs(salary)}")
"""

import json
import pathlib
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# Cached objects (loaded once on first call)
_estimator = None
_meta = None
_feature_names = None


def _load_model():
    """Lazy-load the saved model and metadata."""
    global _estimator, _meta, _feature_names
    if _estimator is None:
        _estimator = joblib.load(MODELS_DIR / "best_model.joblib")
        with open(MODELS_DIR / "training_meta.json") as f:
            _meta = json.load(f)
        feat_path = MODELS_DIR / "feature_names.json"
        if feat_path.exists():
            with open(feat_path) as f:
                _feature_names = json.load(f)
    return _estimator, _meta, _feature_names


def predict_salary(input_dict: Dict[str, Any]) -> float:
    """
    Predict annual salary in INR for a single employee profile.

    Parameters
    ----------
    input_dict : dict
        Must contain all 12 feature keys:
        Age, Gender, Education_Level, Years_Experience, Job_Title,
        Job_Level, Industry, City, Company_Size, Work_Mode,
        Certifications_Count, Skills_Count.

    Returns
    -------
    float
        Predicted salary in INR.
    """
    estimator, meta, _ = _load_model()
    is_log = meta["log_target"]

    # Convert to single-row DataFrame
    input_df = pd.DataFrame([input_dict])

    raw_pred = estimator.predict(input_df)
    if is_log:
        return float(np.expm1(raw_pred[0]))
    return float(raw_pred[0])


def predict_with_interval(
    input_dict: Dict[str, Any],
    mae: float,
) -> Tuple[float, float, float]:
    """
    Returns (point_estimate, lower_bound, upper_bound) where bounds are
    ± 1 × test MAE (not a formal prediction interval, but a practical range).

    Parameters
    ----------
    input_dict : dict
        Employee profile (12 features).
    mae : float
        Test MAE in INR (loaded from training_meta.json or passed explicitly).

    Returns
    -------
    Tuple[float, float, float]
        (predicted_salary, lower_bound, upper_bound) all in INR.
    """
    pred = predict_salary(input_dict)
    lower = max(0, pred - mae)
    upper = pred + mae
    return pred, lower, upper


def format_inr_lakhs(amount: float) -> str:
    """
    Format an INR amount in Indian lakh style.
    e.g. 1_249_692 → '₹12.50 L'

    Parameters
    ----------
    amount : float
        Amount in INR.

    Returns
    -------
    str
        Formatted string.
    """
    lakhs = amount / 1e5
    if lakhs >= 100:
        crore = lakhs / 100
        return f"₹{crore:.2f} Cr"
    return f"₹{lakhs:.2f} L"


def get_test_mae() -> float:
    """Return the test MAE stored in best_model_metrics.json."""
    metrics_path = MODELS_DIR / "best_model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)["test_mae_inr"]
    # Fallback to CV MAE
    _, meta, _ = _load_model()
    best = meta["best_model"]
    return meta["cv_scores"][best]["cv_mae_inr"]


if __name__ == "__main__":
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
    mae = get_test_mae()
    pred, lo, hi = predict_with_interval(sample, mae)
    print(f"Predicted salary : {format_inr_lakhs(pred)}")
    print(f"Approx. range    : {format_inr_lakhs(lo)} – {format_inr_lakhs(hi)}")
    print(f"(Based on test MAE of {format_inr_lakhs(mae)})")
