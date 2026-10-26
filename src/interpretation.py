"""Utilities for readable, model-aware salary explanations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocess import (
    CATEGORICAL_FEATURES,
    INPUT_FEATURES,
    INTEGER_FEATURES,
    JOB_LEVEL_ORDER,
)


FEATURE_LABELS = {
    "Age": "Age",
    "Gender": "Gender",
    "Education_Level": "Education",
    "Years_Experience": "Experience",
    "Job_Title": "Job title",
    "Job_Level": "Job level",
    "Industry": "Industry",
    "City": "Location",
    "Company_Size": "Company size",
    "Work_Mode": "Work mode",
    "Certifications_Count": "Certifications",
    "Skills_Count": "Skills",
    "Age_minus_Exp": "Career-start age",
    "Skills_per_Cert": "Skills per certification",
    "Exp_x_JobLevel": "Experience × job level",
}


def source_feature_name(transformed_name: str) -> str:
    """Map a transformed feature back to its raw input field."""
    if transformed_name in FEATURE_LABELS:
        return transformed_name
    for feature in INPUT_FEATURES:
        if transformed_name == feature or transformed_name.startswith(f"{feature}_"):
            return feature
    return transformed_name


def readable_feature_name(feature: str) -> str:
    """Return a concise label suitable for charts."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def format_feature_value(feature: str, value: Any) -> str:
    """Format a raw feature value for an explanation label."""
    if feature == "Age":
        return f"{value} yrs old"
    if feature == "Years_Experience":
        return f"{value} yrs"
    if feature == "Certifications_Count":
        return f"{value} certs"
    if feature == "Skills_Count":
        return f"{value} skills"
    return str(value)


def input_feature_value(feature: str, input_dict: dict[str, Any]) -> Any:
    """Return a raw or derived value for a profile explanation."""
    if feature in input_dict:
        return input_dict[feature]
    if feature == "Age_minus_Exp":
        return input_dict["Age"] - input_dict["Years_Experience"]
    if feature == "Skills_per_Cert":
        return input_dict["Skills_Count"] / (input_dict["Certifications_Count"] + 1)
    if feature == "Exp_x_JobLevel":
        level = JOB_LEVEL_ORDER.index(input_dict["Job_Level"]) + 1
        return input_dict["Years_Experience"] * level
    return "—"


def aggregate_shap_values(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Aggregate transformed SHAP values into raw feature groups."""
    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP values have {values.shape[1]} features but received "
            f"{len(feature_names)} feature names."
        )

    grouped: dict[str, np.ndarray] = {}
    for index, name in enumerate(feature_names):
        group = source_feature_name(name)
        grouped.setdefault(group, np.zeros(values.shape[0]))
        grouped[group] += values[:, index]

    rows = []
    for group, group_values in grouped.items():
        rows.append(
            {
                "feature": group,
                "label": readable_feature_name(group),
                "mean_abs_shap": float(np.mean(np.abs(group_values))),
                "mean_shap": float(np.mean(group_values)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_shap_factors(
    shap_values: np.ndarray,
    feature_names: list[str],
    input_dict: dict[str, Any],
    top_n: int = 6,
) -> pd.DataFrame:
    """Create a readable grouped explanation for one prediction."""
    values = np.asarray(shap_values, dtype=float).reshape(-1)
    grouped = aggregate_shap_values(values, feature_names)
    grouped["impact"] = grouped["mean_shap"]
    grouped["value"] = grouped["feature"].map(
        lambda feature: input_feature_value(feature, input_dict)
    )
    grouped["display"] = grouped.apply(
        lambda row: f"{row['label']} · {format_feature_value(row['feature'], row['value'])}",
        axis=1,
    )
    return grouped.head(top_n).copy()


def typical_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Build a valid reference profile from the training data."""
    profile: dict[str, Any] = {}
    for feature in INTEGER_FEATURES:
        profile[feature] = int(round(float(df[feature].median())))
    for feature in CATEGORICAL_FEATURES:
        profile[feature] = df[feature].mode(dropna=True).iloc[0]
    return {feature: profile[feature] for feature in INPUT_FEATURES}


def counterfactual_impacts(
    estimator,
    input_dict: dict[str, Any],
    reference_profile: dict[str, Any],
) -> pd.DataFrame:
    """Measure one-feature-at-a-time salary deltas against a reference profile.

    This is a descriptive model comparison, not a causal estimate. All other
    inputs remain fixed while one field is replaced by its reference value.
    """
    input_df = pd.DataFrame([input_dict])
    current_prediction = float(estimator.predict(input_df)[0])
    rows = []
    for feature in INPUT_FEATURES:
        counterfactual = dict(input_dict)
        counterfactual[feature] = reference_profile[feature]
        counterfactual_prediction = float(estimator.predict(pd.DataFrame([counterfactual]))[0])
        rows.append(
            {
                "feature": feature,
                "label": readable_feature_name(feature),
                "value": format_feature_value(feature, input_dict[feature]),
                "reference": format_feature_value(feature, reference_profile[feature]),
                "impact_inr": current_prediction - counterfactual_prediction,
                "prediction_inr": current_prediction,
                "counterfactual_inr": counterfactual_prediction,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("impact_inr", key=lambda values: values.abs(), ascending=False)
        .reset_index(drop=True)
    )
