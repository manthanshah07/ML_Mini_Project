"""
evaluate.py
===========
Computes test-set and CV metrics for all trained models and produces
publication-quality figures saved to reports/figures/.

Metrics reported:
  - MAE  (Mean Absolute Error, INR)
  - RMSE (Root Mean Squared Error, INR)
  - R²   (coefficient of determination)
  - CV MAE mean ± std (INR, 5-fold, from training)

Plots produced:
- model_comparison_bar.png   — bar chart of CV MAE per model
  - actual_vs_predicted.png    — scatter of y_true vs y_pred for best model
  - residuals.png              — residual plot for best model
  - learning_curve.png         — train vs CV score vs training size

Run with:
    python -m src.evaluate
"""

import json
import pathlib
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

from src.preprocess import (
    DATA_PATH,
    RANDOM_STATE,
    data_sha256,
    file_sha256,
    load_data,
    split_data,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
CV_FOLDS = 5

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "Ridge": "#4C72B0",
    "Lasso": "#DD8452",
    "RandomForest": "#55A868",
    "XGBoost": "#C44E52",
    "GradientBoosting": "#8172B3",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inr_fmt(x, pos=None) -> str:
    """Format axis ticks in Indian lakh notation (e.g. 10,00,000 → ₹10L)."""
    lakhs = x / 1e5
    return f"₹{lakhs:.0f}L"


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ── Model evaluation ──────────────────────────────────────────────────────────

def evaluate_best_model(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_name: str,
) -> Tuple[Dict, np.ndarray]:
    """
    Full evaluation of the best model on the held-out test set.
    Predictions are directly in INR.
    """
    y_pred = estimator.predict(X_test)
    metrics = _compute_metrics(y_test.values, y_pred)

    print(f"\n  Best model ({best_name}) test-set metrics:")
    print(f"    MAE  = {metrics['MAE']:>12,.0f} INR")
    print(f"    RMSE = {metrics['RMSE']:>12,.0f} INR")
    print(f"    R²   = {metrics['R2']:>12.4f}")

    return metrics, y_pred


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_model_comparison(comparison_df: pd.DataFrame, best_name: str) -> None:
    """Bar chart of CV MAE (INR) for each model."""
    df = comparison_df.copy().sort_values("CV_MAE_mean_INR")
    colors = [PALETTE.get(m, "#888888") for m in df.index]
    highlight = ["#FFD700" if m == best_name else c for m, c in zip(df.index, colors)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        df.index,
        df["CV_MAE_mean_INR"] / 1e5,
        xerr=df["CV_MAE_std_INR"] / 1e5,
        color=highlight,
        edgecolor="white",
        linewidth=0.8,
        capsize=4,
        error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
    )
    ax.set_xlabel("5-Fold CV MAE (₹ Lakhs)", fontsize=12)
    ax.set_title(
        "Model Comparison — 5-Fold CV MAE on Training Set\n"
        "(Gold = selected best model; selection NOT based on test set)",
        fontsize=11,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"₹{x:.1f}L"))
    for bar, val, std in zip(bars, df["CV_MAE_mean_INR"], df["CV_MAE_std_INR"]):
        ax.text(
            bar.get_width() + std / 1e5 + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"₹{val/1e5:.2f}L",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    path = FIGURES_DIR / "model_comparison_bar.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, best_name: str) -> None:
    """Scatter plot of actual vs predicted salary."""
    fig, ax = plt.subplots(figsize=(7, 7))
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]

    ax.scatter(
        y_true / 1e5,
        y_pred / 1e5,
        alpha=0.6,
        s=40,
        color=PALETTE.get(best_name, "#4C72B0"),
        edgecolors="white",
        linewidths=0.4,
    )
    ax.plot(
        [lims[0] / 1e5, lims[1] / 1e5],
        [lims[0] / 1e5, lims[1] / 1e5],
        "r--",
        linewidth=1.5,
        label="Perfect prediction",
    )
    ax.set_xlabel("Actual Salary (₹ Lakhs)", fontsize=12)
    ax.set_ylabel("Predicted Salary (₹ Lakhs)", fontsize=12)
    ax.set_title(f"Actual vs Predicted — {best_name} (Test Set)", fontsize=12)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"₹{x:.0f}L"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"₹{x:.0f}L"))
    plt.tight_layout()
    path = FIGURES_DIR / "actual_vs_predicted.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, best_name: str) -> None:
    """Residual plot (predicted vs residual)."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residual vs Predicted
    axes[0].scatter(
        y_pred / 1e5,
        residuals / 1e5,
        alpha=0.6,
        s=40,
        color=PALETTE.get(best_name, "#4C72B0"),
        edgecolors="white",
        linewidths=0.4,
    )
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Salary (₹ Lakhs)", fontsize=11)
    axes[0].set_ylabel("Residual (₹ Lakhs)", fontsize=11)
    axes[0].set_title("Residuals vs Predicted", fontsize=11)

    # Residual histogram
    axes[1].hist(residuals / 1e5, bins=25, color=PALETTE.get(best_name, "#4C72B0"), edgecolor="white")
    axes[1].set_xlabel("Residual (₹ Lakhs)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Residual Distribution", fontsize=11)

    fig.suptitle(f"Residual Analysis — {best_name} (Test Set)", fontsize=12, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "residuals.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_learning_curve(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_name: str,
) -> None:
    """Learning curve evaluated directly on INR scale."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_train,
        y_train.values,
        train_sizes=np.linspace(0.15, 1.0, 8),
        cv=CV_FOLDS,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    tr_mean = (-train_scores).mean(axis=1)
    tr_std = (-train_scores).std(axis=1)
    val_mean = (-val_scores).mean(axis=1)
    val_std = (-val_scores).std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, tr_mean / 1e5, "o-", color="#4C72B0", label="Training MAE")
    ax.fill_between(
        train_sizes,
        (tr_mean - tr_std) / 1e5,
        (tr_mean + tr_std) / 1e5,
        alpha=0.15,
        color="#4C72B0",
    )
    ax.plot(train_sizes, val_mean / 1e5, "o-", color="#C44E52", label="CV Validation MAE")
    ax.fill_between(
        train_sizes,
        (val_mean - val_std) / 1e5,
        (val_mean + val_std) / 1e5,
        alpha=0.15,
        color="#C44E52",
    )
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("MAE (₹ Lakhs)", fontsize=12)
    ax.set_title(f"Learning Curve — {best_name} (Evaluated on INR)", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x)}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"₹{x:.1f}L"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = FIGURES_DIR / "learning_curve.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data and saved artifacts …")
    df = load_data(DATA_PATH)

    # Load metadata
    meta_path = MODELS_DIR / "training_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    expected_hash = meta.get("data_sha256")
    if expected_hash and expected_hash != data_sha256(DATA_PATH):
        raise RuntimeError(
            "The dataset has changed since training. Retrain before evaluating."
        )

    split_path = MODELS_DIR / "test_split.joblib"
    if split_path.exists():
        X_test, y_test = joblib.load(split_path)
        X_train = df.drop(columns=["Salary_INR"]).drop(index=X_test.index)
        y_train = df.loc[X_train.index, "Salary_INR"]
    else:
        X_train, X_test, y_train, y_test = split_data(df)

    best_name: str = meta["best_model"]
    print(f"Best model: {best_name}")

    # Load best model
    estimator = joblib.load(MODELS_DIR / "best_model.joblib")

    # Evaluate best model on test set
    metrics, y_pred = evaluate_best_model(estimator, X_test, y_test, best_name)

    # Build comparison table with full test and CV metrics for ALL models
    rows = []
    for name, cv_info in meta["cv_scores"].items():
        row = {
            "Model": name,
            "CV_MAE_mean_INR": cv_info["cv_mae_inr"],
            "CV_MAE_std_INR": cv_info["cv_std_inr"],
            "Test_MAE_INR": cv_info.get("test_mae_inr", metrics["MAE"] if name == best_name else np.nan),
            "Test_RMSE_INR": cv_info.get("test_rmse_inr", metrics["RMSE"] if name == best_name else np.nan),
            "Test_R2": cv_info.get("test_r2", metrics["R2"] if name == best_name else np.nan),
        }
        rows.append(row)

    comparison_df = pd.DataFrame(rows).set_index("Model")
    table_path = ROOT / "reports" / "model_comparison.csv"
    comparison_df.to_csv(table_path)
    print(f"\n  Saved comparison table to {table_path}")
    print(comparison_df.to_string())

    # Plots
    print("\nGenerating plots …")
    plot_model_comparison(comparison_df, best_name)
    plot_actual_vs_predicted(y_test.values, y_pred, best_name)
    plot_residuals(y_test.values, y_pred, best_name)
    plot_learning_curve(estimator, X_train, y_train, best_name)

    # Save best model metrics to JSON
    results_path = MODELS_DIR / "best_model_metrics.json"
    best_cv_info = meta["cv_scores"][best_name]
    with open(results_path, "w") as f:
        json.dump(
            {
                "artifact_schema_version": meta.get("artifact_schema_version", 1),
                "data_sha256": meta.get("data_sha256"),
                "model_sha256": meta.get("model_sha256") or file_sha256(MODELS_DIR / "best_model.joblib"),
                "error_band_inr": meta.get("error_band_inr"),
                "error_band_quantile": meta.get("error_band_quantile"),
                "best_model": best_name,
                "test_mae_inr": metrics["MAE"],
                "test_rmse_inr": metrics["RMSE"],
                "test_r2": metrics["R2"],
                "cv_mae_inr": best_cv_info["cv_mae_inr"],
                "cv_std_inr": best_cv_info["cv_std_inr"],
            },
            f,
            indent=2,
        )
    print(f"\n  Saved metrics to {results_path}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
