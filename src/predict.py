"""
predict.py
==========
Inference helper for the trained salary prediction model.

Provides:
  - predict_salary(input_dict) → predicted salary in INR
  - format_inr_lakhs(amount)   → Indian lakh-formatted string
  - validate_profile(input_dict) → list of warnings / validation messages

Used by the Streamlit app and by tests.
"""

import json
import pathlib
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

REQUIRED_FEATURES = [
    "Age", "Gender", "Education_Level", "Years_Experience", "Job_Title",
    "Job_Level", "Industry", "City", "Company_Size", "Work_Mode",
    "Certifications_Count", "Skills_Count",
]

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


def validate_profile(input_dict: Dict[str, Any]) -> List[str]:
    """
    Validates employee profile values and returns a list of warning messages.
    """
    warnings_list = []
    missing = [k for k in REQUIRED_FEATURES if k not in input_dict]
    if missing:
        warnings_list.append(f"Missing required fields: {', '.join(missing)}")

    age = input_dict.get("Age", 0)
    exp = input_dict.get("Years_Experience", 0)
    if exp > age - 18:
        warnings_list.append(
            f"Unusual profile: Years of experience ({exp}) exceeds career span for age {age} (Age − 18 = {age - 18})."
        )
    return warnings_list


def predict_salary(input_dict: Dict[str, Any]) -> float:
    """
    Predict annual salary in INR for a single employee profile.
    Automatically handles pipeline preprocessing and inverse transformation.
    """
    estimator, _, _ = _load_model()
    input_df = pd.DataFrame([input_dict])
    raw_pred = estimator.predict(input_df)
    return float(raw_pred[0])


def predict_with_interval(
    input_dict: Dict[str, Any],
    mae: float = None,
) -> Tuple[float, float, float]:
    """
    Returns (predicted_salary, lower_bound, upper_bound) based on test MAE.
    """
    if mae is None:
        mae = get_test_mae()
    pred = predict_salary(input_dict)
    lower = max(0, pred - mae)
    upper = pred + mae
    return pred, lower, upper


def format_inr_lakhs(amount: float) -> str:
    """
    Format an INR amount in Indian lakh style.
    e.g. 1_249_692 → '₹12.50 L'
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
            return json.load(f).get("test_mae_inr", 115000.0)
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
