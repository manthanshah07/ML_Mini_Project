"""
explain.py
==========
Generates explainability artefacts for the best model:
  1. Permutation Feature Importance (model-agnostic)
  2. SHAP Summary Plot (beeswarm) — for tree models
  3. SHAP Bar Plot (mean |SHAP|)
  4. Fairness check: Male vs Female prediction error and predicted salary,
     controlled for Years_Experience band and Job_Level.

All plots saved to reports/figures/ at 150 dpi.

Run with:
    python -m src.explain
"""

import json
import pathlib
import warnings
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.preprocess import (
    DATA_PATH,
    RANDOM_STATE,
    get_feature_names,
    load_data,
    split_data,
)

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inv_log(arr: np.ndarray) -> np.ndarray:
    return np.expm1(arr)


def _load_artifacts():
    """Load best model, metadata, and test split."""
    meta_path = MODELS_DIR / "training_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    estimator = joblib.load(MODELS_DIR / "best_model.joblib")
    return estimator, meta


def _get_preprocessed_data(estimator, X_train, X_test, y_train, y_test, is_log):
    """Transform data through the pipeline and return arrays + feature names."""
    # Extract preprocessor (first step of the pipeline)
    preprocess_pipe = estimator.named_steps["preprocess"]
    feature_names = get_feature_names(preprocess_pipe)

    X_train_t = preprocess_pipe.transform(X_train)
    X_test_t = preprocess_pipe.transform(X_test)

    y_tr_arr = np.log1p(y_train.values) if is_log else y_train.values
    y_te_arr = y_test.values

    return X_train_t, X_test_t, y_tr_arr, y_te_arr, feature_names


# ── Permutation Importance ────────────────────────────────────────────────────

def plot_permutation_importance(
    estimator,
    X_test_t: np.ndarray,
    y_test_arr: np.ndarray,
    feature_names: list,
    best_name: str,
    is_log: bool,
    top_n: int = 20,
) -> pd.DataFrame:
    """Compute and plot permutation feature importance on the test set."""
    model = estimator.named_steps["model"]

    result = permutation_importance(
        model,
        X_test_t,
        y_test_arr,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    imp_df = pd.DataFrame(
        {
            "feature": feature_names[:X_test_t.shape[1]],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    top = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top["feature"][::-1],
        top["importance_mean"][::-1],
        xerr=top["importance_std"][::-1],
        color="#4C72B0",
        edgecolor="white",
        capsize=3,
        error_kw={"elinewidth": 1.2},
    )
    ax.set_xlabel("Mean decrease in MAE (permutation importance)", fontsize=11)
    ax.set_title(
        f"Permutation Feature Importance — {best_name}\n(Test set, 20 repeats)",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIGURES_DIR / "permutation_importance.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return imp_df


# ── SHAP ──────────────────────────────────────────────────────────────────────

def plot_shap(
    estimator,
    X_train_t: np.ndarray,
    X_test_t: np.ndarray,
    feature_names: list,
    best_name: str,
    is_log: bool,
) -> shap.Explanation:
    """Compute SHAP values and produce summary + bar plots."""
    model = estimator.named_steps["model"]
    model_type = type(model).__name__

    # Use TreeExplainer for tree models; KernelExplainer otherwise
    if model_type in ("RandomForestRegressor", "XGBRegressor", "GradientBoostingRegressor"):
        # Use a sample of training data as background for speed
        bg = shap.utils.sample(X_train_t, 100, random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        shap_values = explainer(X_test_t, check_additivity=False)
    else:
        bg = shap.utils.sample(X_train_t, 50, random_state=RANDOM_STATE)
        explainer = shap.LinearExplainer(model, bg)
        shap_values = explainer(X_test_t)

    fn = feature_names[:X_test_t.shape[1]]

    # ── SHAP Beeswarm (summary) ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(
        shap_values.values if hasattr(shap_values, "values") else shap_values,
        X_test_t,
        feature_names=fn,
        show=False,
        max_display=20,
        plot_size=None,
    )
    plt.title(f"SHAP Summary Plot — {best_name}", fontsize=12, pad=12)
    plt.tight_layout()
    path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── SHAP Bar (mean |SHAP|) ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values.values if hasattr(shap_values, "values") else shap_values,
        X_test_t,
        feature_names=fn,
        plot_type="bar",
        show=False,
        max_display=20,
        plot_size=None,
    )
    plt.title(f"SHAP Mean |SHAP| — {best_name}", fontsize=12, pad=12)
    plt.tight_layout()
    path = FIGURES_DIR / "shap_bar.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return shap_values, explainer


# ── Fairness check ────────────────────────────────────────────────────────────

def fairness_check(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    is_log: bool,
) -> pd.DataFrame:
    """
    Compares average prediction error and average predicted salary for
    Male vs Female, controlling (approximately) for Years_Experience band
    and Job_Level.

    NOTE: The dataset has only 500 rows. Any group-level differences here
    should be interpreted very cautiously — they may reflect sampling noise
    rather than systematic bias. No causal claims are made.
    """
    preds_raw = estimator.predict(X_test)
    preds = _inv_log(preds_raw) if is_log else preds_raw
    residuals = y_test.values - preds

    analysis_df = X_test[["Gender", "Years_Experience", "Job_Level"]].copy()
    analysis_df["y_true"] = y_test.values
    analysis_df["y_pred"] = preds
    analysis_df["residual"] = residuals
    analysis_df["abs_error"] = np.abs(residuals)

    # Bin experience into quartiles
    analysis_df["exp_band"] = pd.qcut(
        analysis_df["Years_Experience"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )

    # Overall Male vs Female summary
    overall = (
        analysis_df.groupby("Gender")[["y_pred", "abs_error", "residual"]]
        .agg(["mean", "count"])
        .round(0)
    )

    # Controlled for experience band and job level
    controlled = (
        analysis_df.groupby(["Gender", "exp_band", "Job_Level"])[
            ["y_pred", "abs_error", "residual"]
        ]
        .mean()
        .round(0)
    )

    print("\n  Fairness Check — Overall (Male vs Female):")
    print(overall.to_string())
    print("\n  Fairness Check — Controlled (exp band × job level):")
    print(controlled.to_string())
    print(
        "\n  ⚠ Note: Dataset has 500 rows; group sizes are small. Differences "
        "reflect sampling variance as much as true effects. No causal claims."
    )

    # Save
    overall_path = ROOT / "reports" / "fairness_overall.csv"
    controlled_path = ROOT / "reports" / "fairness_controlled.csv"
    overall.to_csv(overall_path)
    controlled.to_csv(controlled_path)
    print(f"  Saved: {overall_path}, {controlled_path}")

    return analysis_df


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data and artifacts …")
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    estimator, meta = _load_artifacts()
    best_name: str = meta["best_model"]
    is_log: bool = meta["log_target"]
    print(f"Best model: {best_name} | log_target: {is_log}")

    X_train_t, X_test_t, y_tr_arr, y_te_arr, feature_names = _get_preprocessed_data(
        estimator, X_train, X_test, y_train, y_test, is_log
    )

    # Permutation importance
    print("\nComputing permutation importance …")
    imp_df = plot_permutation_importance(
        estimator, X_test_t, y_te_arr, feature_names, best_name, is_log
    )
    print("\n  Top 10 features by permutation importance:")
    print(imp_df.head(10).to_string(index=False))
    imp_df.to_csv(ROOT / "reports" / "permutation_importance.csv", index=False)

    # SHAP
    print("\nComputing SHAP values …")
    shap_values, explainer = plot_shap(
        estimator, X_train_t, X_test_t, feature_names, best_name, is_log
    )

    # Save explainer for the app
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
    print(f"  Saved SHAP explainer to {MODELS_DIR / 'shap_explainer.joblib'}")

    # Save feature names
    import json
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names[:X_test_t.shape[1]], f)

    # Fairness check
    print("\nRunning fairness check …")
    fairness_check(estimator, X_test, y_test, is_log)

    print("\nExplainability analysis complete.")


if __name__ == "__main__":
    main()
