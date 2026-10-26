"""
explain.py
==========
Generates explainability artefacts for the best model:
  1. Permutation Feature Importance (model-agnostic, evaluated in INR MAE)
  2. SHAP Summary Plot (beeswarm)
  3. SHAP Bar Plot (mean |SHAP|)
  4. Fairness check: Male vs Female prediction error and predicted salary,
     controlled for Years_Experience band and Job_Level with sample counts.

All plots saved to reports/figures/ at 150 dpi.

Run with:
    python -m src.explain
"""

import json
import pathlib
from typing import Any, Tuple

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance

from src.preprocess import (
    DATA_PATH,
    RANDOM_STATE,
    data_sha256,
    file_sha256,
    get_feature_names,
    load_data,
    split_data,
)
from src.interpretation import aggregate_shap_values

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_artifacts():
    """Load best model, metadata, and test split."""
    meta_path = MODELS_DIR / "training_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    estimator = joblib.load(MODELS_DIR / "best_model.joblib")
    return estimator, meta


# ── Permutation Importance ────────────────────────────────────────────────────

def plot_permutation_importance(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_name: str,
    top_n: int = 12,
) -> pd.DataFrame:
    """
    Compute and plot permutation feature importance on the held-out test set
    evaluated directly on original INR MAE.
    """
    result = permutation_importance(
        estimator,
        X_test,
        y_test.values,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    imp_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean_inr": result.importances_mean,
            "importance_std_inr": result.importances_std,
        }
    ).sort_values("importance_mean_inr", ascending=False)

    top = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top["feature"][::-1],
        top["importance_mean_inr"][::-1] / 1e5,
        xerr=top["importance_std_inr"][::-1] / 1e5,
        color="#4C72B0",
        edgecolor="white",
        capsize=3,
        error_kw={"elinewidth": 1.2},
    )
    ax.set_xlabel("Mean Increase in MAE when Shuffled (₹ Lakhs)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"₹{x:.1f}L"))
    ax.set_title(
        f"Permutation Feature Importance — {best_name}\n(Evaluated in INR, Test set, 20 repeats)",
        fontsize=11,
    )
    for i, (mean, std) in enumerate(zip(top["importance_mean_inr"][::-1], top["importance_std_inr"][::-1])):
        val_l = mean / 1e5
        std_l = std / 1e5
        offset = max(std_l, 0) + 0.05
        ax.text(val_l + offset, i, f"₹{val_l:.2f}L", va="center", fontsize=8.5)

    plt.tight_layout()
    path = FIGURES_DIR / "permutation_importance.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return imp_df


# ── SHAP ──────────────────────────────────────────────────────────────────────

def plot_shap(
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    best_name: str,
) -> Tuple[Any, Any, list[str], pd.DataFrame]:
    """Compute SHAP values and produce summary + bar plots."""
    preprocess_pipe = estimator.named_steps["preprocess"]
    feature_names = get_feature_names(preprocess_pipe)

    X_train_t = preprocess_pipe.transform(X_train)
    X_test_t = preprocess_pipe.transform(X_test)

    # Unwrap TransformedTargetRegressor if present
    model_step = estimator.named_steps["model"]
    inner_model = getattr(model_step, "regressor_", model_step)
    model_type = type(inner_model).__name__
    target_label = "log-salary" if isinstance(model_step, TransformedTargetRegressor) else "salary"

    if model_type in ("RandomForestRegressor", "XGBRegressor", "GradientBoostingRegressor"):
        bg = shap.utils.sample(X_train_t, 100, random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(inner_model, data=bg, feature_perturbation="interventional")
        shap_values = explainer(X_test_t, check_additivity=False)
    else:
        bg = shap.utils.sample(X_train_t, 50, random_state=RANDOM_STATE)
        explainer = shap.LinearExplainer(inner_model, bg)
        shap_values = explainer(X_test_t)

    fn = feature_names[:X_test_t.shape[1]]

    # ── SHAP Beeswarm (summary) ───────────────────────────────────────────────
    shap.summary_plot(
        shap_values.values if hasattr(shap_values, "values") else shap_values,
        X_test_t,
        feature_names=fn,
        show=False,
        max_display=20,
        plot_size=None,
    )
    plt.title(
        f"Technical SHAP Detail — {best_name} (Impact on {target_label})\n"
        "Encoded model features; grouped view is shown in shap_bar.png",
        fontsize=12,
        pad=12,
    )
    plt.tight_layout()
    path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── Grouped SHAP Bar (mean |SHAP|) ───────────────────────────────────────
    raw_shap_values = shap_values.values if hasattr(shap_values, "values") else shap_values
    grouped = aggregate_shap_values(raw_shap_values, fn).head(12).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(grouped["label"], grouped["mean_abs_shap"], color="#356F68")
    ax.set_xlabel("Mean absolute SHAP value (model output scale)")
    ax.set_title(
        f"Grouped SHAP Importance — {best_name}\n"
        f"(Impact on {target_label}; encoded categories combined)",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = FIGURES_DIR / "shap_bar.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return shap_values, explainer, fn, grouped


# ── Fairness check ────────────────────────────────────────────────────────────

def _bootstrap_mean_interval(values: pd.Series, seed: int = RANDOM_STATE) -> tuple[float, float]:
    """Return a percentile bootstrap interval for a group mean."""
    values = values.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def fairness_check(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Compares average prediction error and average predicted salary for
    Male vs Female, controlling for Years_Experience band and Job_Level.
    Includes sample counts (n) for each bin.
    """
    preds = estimator.predict(X_test)
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

    # Overall Male vs Female summary with uncertainty for mean absolute error.
    overall = (
        analysis_df.groupby("Gender")
        .agg(
            n=("Gender", "size"),
            predicted_salary_mean=("y_pred", "mean"),
            absolute_error_mean=("abs_error", "mean"),
            residual_mean=("residual", "mean"),
        )
        .reset_index()
    )
    intervals = overall["Gender"].map(
        {
            gender: _bootstrap_mean_interval(
                analysis_df.loc[analysis_df["Gender"] == gender, "abs_error"],
                seed=RANDOM_STATE + index,
            )
            for index, gender in enumerate(overall["Gender"])
        }
    )
    overall[["absolute_error_ci_low", "absolute_error_ci_high"]] = pd.DataFrame(
        intervals.tolist(), index=overall.index
    )
    overall = overall.round(0)

    # Controlled for experience band and job level with sample counts
    controlled = (
        analysis_df.groupby(["Gender", "exp_band", "Job_Level"], observed=True)
        .agg(
            n=("Gender", "size"),
            predicted_salary_mean=("y_pred", "mean"),
            absolute_error_mean=("abs_error", "mean"),
            residual_mean=("residual", "mean"),
        )
        .reset_index()
        .round(0)
    )
    controlled["small_group"] = controlled["n"] < 10

    print("\n  Fairness Check — Overall (Male vs Female):")
    print(overall.to_string())
    print("\n  Fairness Check — Controlled (exp band × job level):")
    print(controlled.head(15).to_string(index=False))
    print(
        "\n  ⚠ Note: Dataset has 500 rows; group sizes are small. Differences "
        "reflect sampling variance as much as true effects. No causal claims."
    )

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

    estimator, meta = _load_artifacts()
    expected_hash = meta.get("data_sha256")
    if expected_hash and expected_hash != data_sha256(DATA_PATH):
        raise RuntimeError(
            "The dataset has changed since training. Retrain before explaining."
        )

    split_path = MODELS_DIR / "test_split.joblib"
    if split_path.exists():
        X_test, y_test = joblib.load(split_path)
        X_train = df.drop(columns=["Salary_INR"]).drop(index=X_test.index)
    else:
        X_train, X_test, _, y_test = split_data(df)
    best_name: str = meta["best_model"]
    print(f"Best model: {best_name}")

    # Permutation importance
    print("\nComputing permutation importance …")
    imp_df = plot_permutation_importance(estimator, X_test, y_test, best_name)
    print("\n  Top features by permutation importance:")
    print(imp_df.head(10).to_string(index=False))
    imp_df.to_csv(ROOT / "reports" / "permutation_importance.csv", index=False)

    # SHAP
    print("\nComputing SHAP values …")
    shap_values, explainer, feature_names, grouped = plot_shap(
        estimator, X_train, X_test, best_name
    )

    # Save explainer for the app
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
    print(f"  Saved SHAP explainer to {MODELS_DIR / 'shap_explainer.joblib'}")

    # Save feature names
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    grouped.to_csv(ROOT / "reports" / "shap_grouped_importance.csv", index=False)

    with open(MODELS_DIR / "explainability_meta.json", "w") as f:
        json.dump(
            {
                "data_sha256": data_sha256(DATA_PATH),
                "model_sha256": file_sha256(MODELS_DIR / "best_model.joblib"),
                "feature_count": len(feature_names),
                "best_model": best_name,
            },
            f,
            indent=2,
        )

    # Fairness check
    print("\nRunning fairness check …")
    fairness_check(estimator, X_test, y_test)

    print("\nExplainability analysis complete.")


if __name__ == "__main__":
    main()
