"""
predict.py
==========
Inference helper for the trained salary prediction model.

Provides:
  - predict_salary(input_dict) → predicted salary in INR
  - format_inr_lakhs(amount)   → Indian lakh-formatted string
  - validate_profile(input_dict) → list of validation errors

Used by the Streamlit app and by tests.
"""

import json
import math
import numbers
import pathlib
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd

from src.preprocess import (
    CATEGORY_VALUES,
    INTEGER_RANGES,
    INPUT_FEATURES,
    data_sha256,
    file_sha256,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

REQUIRED_FEATURES = INPUT_FEATURES

VALID_CATEGORIES = {
    field: set(values) for field, values in CATEGORY_VALUES.items()
}

NUMERIC_RANGES = INTEGER_RANGES

# Cached objects (loaded once on first call)
_estimator = None
_meta = None
_feature_names = None


def _load_model():
    """Lazy-load the saved model and metadata."""
    global _estimator, _meta, _feature_names
    if _estimator is None:
        model_path = MODELS_DIR / "best_model.joblib"
        meta_path = MODELS_DIR / "training_meta.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "Model artifacts are missing. Run `python -m src.train` first."
            )
        _estimator = joblib.load(model_path)
        with open(meta_path) as f:
            _meta = json.load(f)
        if _meta.get("data_sha256") and _meta["data_sha256"] != data_sha256():
            raise RuntimeError("The dataset changed after this model was trained.")
        if _meta.get("model_sha256") and _meta["model_sha256"] != file_sha256(model_path):
            raise RuntimeError("The model artifact does not match its metadata.")
        feat_path = MODELS_DIR / "feature_names.json"
        if feat_path.exists():
            with open(feat_path) as f:
                _feature_names = json.load(f)
    return _estimator, _meta, _feature_names


def validate_profile(input_dict: Dict[str, Any]) -> List[str]:
    """
    Validate employee profile values and return human-readable errors.

    The function remains list-based for the Streamlit UI, but callers should
    treat a non-empty result as invalid input rather than merely a warning.
    """
    errors = []
    missing = [k for k in REQUIRED_FEATURES if k not in input_dict]
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")
        return errors

    for field, allowed in VALID_CATEGORIES.items():
        value = input_dict[field]
        if not isinstance(value, str) or value not in allowed:
            errors.append(
                f"Invalid {field}: {value!r}. Expected one of: {', '.join(sorted(allowed))}."
            )

    for field, (lower, upper) in NUMERIC_RANGES.items():
        value = input_dict[field]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            errors.append(f"{field} must be a number between {lower} and {upper}.")
            continue
        if not math.isfinite(float(value)) or int(value) != value:
            errors.append(f"{field} must be a finite whole number between {lower} and {upper}.")
            continue
        if not lower <= value <= upper:
            errors.append(f"{field} must be between {lower} and {upper}; received {value}.")

    age = input_dict.get("Age", 0)
    exp = input_dict.get("Years_Experience", 0)
    if (
        isinstance(age, numbers.Real)
        and isinstance(exp, numbers.Real)
        and exp > age - 18
    ):
        errors.append(
            f"Unusual profile: Years of experience ({exp}) exceeds career span for age {age} (Age − 18 = {age - 18})."
        )
    return errors


def predict_salary(input_dict: Dict[str, Any]) -> float:
    """
    Predict annual salary in INR for a single employee profile.
    Automatically handles pipeline preprocessing and inverse transformation.
    """
    errors = validate_profile(input_dict)
    if errors:
        raise ValueError("Invalid profile: " + " ".join(errors))

    estimator, _, _ = _load_model()
    input_df = pd.DataFrame([input_dict])
    raw_pred = estimator.predict(input_df)
    return float(raw_pred[0])


def predict_with_interval(
    input_dict: Dict[str, Any],
    mae: float = None,
    *,
    error_band: float = None,
) -> Tuple[float, float, float]:
    """Return a prediction and an empirical typical-error band.

    The default band is the out-of-fold 90th percentile of absolute errors
    saved during training. It is empirical and should not be interpreted as a
    guaranteed confidence interval.
    """
    if error_band is None:
        error_band = mae if mae is not None else get_prediction_error_band()
    pred = predict_salary(input_dict)
    lower = max(0, pred - error_band)
    upper = pred + error_band
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
            metrics = json.load(f)
        _, meta, _ = _load_model()
        if (
            metrics.get("best_model") == meta.get("best_model")
            and metrics.get("data_sha256") == meta.get("data_sha256")
            and metrics.get("model_sha256") == meta.get("model_sha256")
        ):
            test_mae = metrics.get("test_mae_inr")
            if test_mae is not None:
                return float(test_mae)
    # Fallback to CV MAE
    _, meta, _ = _load_model()
    best = meta["best_model"]
    return meta["cv_scores"][best]["cv_mae_inr"]


def get_prediction_error_band() -> float:
    """Return the empirical error band generated during training."""
    metrics_path = MODELS_DIR / "best_model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        _, meta, _ = _load_model()
        if (
            metrics.get("best_model") == meta.get("best_model")
            and metrics.get("data_sha256") == meta.get("data_sha256")
            and metrics.get("model_sha256") == meta.get("model_sha256")
        ):
            band = metrics.get("error_band_inr")
            if band is not None:
                return float(band)
    # Compatibility fallback for artifacts generated before error-band
    # calibration was added. This remains an error scale, not an interval.
    return get_test_mae()


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
    error_band = get_prediction_error_band()
    pred, lo, hi = predict_with_interval(sample, error_band)
    print(f"Predicted salary : {format_inr_lakhs(pred)}")
    print(f"Typical error band: {format_inr_lakhs(lo)} – {format_inr_lakhs(hi)}")
    print(f"(Based on an empirical error band of {format_inr_lakhs(error_band)})")
