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
  - model_comparison_bar.png   — bar chart of test MAE per model
  - actual_vs_predicted.png    — scatter of y_true vs y_pred for best model
  - residuals.png              — residual plot for best model
  - learning_curve.png         — train vs CV score vs training size

Run with:
    python -m src.evaluate
"""

import json
import pathlib
import warnings
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocess import (
    DATA_PATH,
    RANDOM_STATE,
    load_data,
    split_data,
)

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
CV_FOLDS = 5
RANDOM_STATE = 42

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


def _inv_log(arr: np.ndarray) -> np.ndarray:
    return np.expm1(arr)


# ── Model evaluation ──────────────────────────────────────────────────────────

def evaluate_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    meta: Dict,
) -> pd.DataFrame:
    """
    Loads every saved model candidate, evaluates on the test set,
    and returns a comparison DataFrame.
    """
    rows = []
    cv_scores = meta.get("cv_scores", {})

    for name, cv_info in cv_scores.items():
        model_candidate_path = MODELS_DIR / f"{name}_model.joblib"
        # We only saved the best model; for others reload from meta
        # and note that test metrics for non-best models are approximated
        # by checking if a per-model file exists
        if not model_candidate_path.exists():
            # Only best model is available on disk; skip individual eval
            rows.append(
                {
                    "Model": name,
                    "Test_MAE": np.nan,
                    "Test_RMSE": np.nan,
                    "Test_R2": np.nan,
                    "CV_MAE_mean": cv_info["cv_mae_inr"],
                    "CV_MAE_std": cv_info["cv_std_inr"],
                }
            )
        else:
            est = joblib.load(model_candidate_path)
            is_log = name in ("Ridge", "Lasso")
            preds = est.predict(X_test)
            if is_log:
                preds = _inv_log(preds)
            m = _compute_metrics(y_test.values, preds)
            rows.append(
                {
                    "Model": name,
                    "Test_MAE": m["MAE"],
                    "Test_RMSE": m["RMSE"],
                    "Test_R2": m["R2"],
                    "CV_MAE_mean": cv_info["cv_mae_inr"],
                    "CV_MAE_std": cv_info["cv_std_inr"],
                }
            )

    return pd.DataFrame(rows).set_index("Model")


def evaluate_best_model(
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_name: str,
    is_log: bool,
) -> Tuple[Dict, np.ndarray]:
    """
    Full evaluation of the best model on the held-out test set.
    Returns (metrics_dict, y_pred).
    """
    y_pred_raw = estimator.predict(X_test)
    y_pred = _inv_log(y_pred_raw) if is_log else y_pred_raw
    metrics = _compute_metrics(y_test.values, y_pred)

    # CV MAE on training set
    y_tr = np.log1p(y_train.values) if is_log else y_train.values
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_maes = []
    for tr_idx, val_idx in kf.split(X_train):
        estimator.fit(X_train.iloc[tr_idx], y_tr[tr_idx])
        preds_fold = estimator.predict(X_train.iloc[val_idx])
        if is_log:
            preds_fold = _inv_log(preds_fold)
        fold_maes.append(mean_absolute_error(y_train.values[val_idx], preds_fold))

    metrics["CV_MAE_mean"] = np.mean(fold_maes)
    metrics["CV_MAE_std"] = np.std(fold_maes)

    # Refit on full training data
    estimator.fit(X_train, y_tr)

    print(f"\n  Best model ({best_name}) test-set metrics:")
    print(f"    MAE  = {metrics['MAE']:>12,.0f} INR")
    print(f"    RMSE = {metrics['RMSE']:>12,.0f} INR")
    print(f"    R²   = {metrics['R2']:>12.4f}")
    print(f"    CV MAE = {metrics['CV_MAE_mean']:>10,.0f} ± {metrics['CV_MAE_std']:>8,.0f} INR")

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
            bar.get_width() + std / 1e5 + 0.1,
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
    y_train_arr: np.ndarray,
    best_name: str,
    is_log: bool,
) -> None:
    """Learning curve to check overfitting."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_train,
        y_train_arr,
        train_sizes=np.linspace(0.15, 1.0, 8),
        cv=CV_FOLDS,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    # Convert to INR if log-target
    def to_inr(scores):
        mae = -scores  # neg_MAE → positive
        if is_log:
            # Approximate: back-transform median log-salary
            med_log = np.log1p(1_083_000)
            mae_inr = (np.expm1(mae + med_log) - 1_083_000)
            return mae_inr  # delta
        return mae

    tr_mean = to_inr(train_scores).mean(axis=1)
    tr_std = to_inr(train_scores).std(axis=1)
    val_mean = to_inr(val_scores).mean(axis=1)
    val_std = to_inr(val_scores).std(axis=1)

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
    ax.set_ylabel("MAE (₹ Lakhs, approx.)", fontsize=12)
    ax.set_title(f"Learning Curve — {best_name}", fontsize=12)
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
    X_train, X_test, y_train, y_test = split_data(df)

    # Load metadata
    meta_path = MODELS_DIR / "training_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    best_name: str = meta["best_model"]
    is_log: bool = meta["log_target"]
    print(f"Best model: {best_name} | log_target: {is_log}")

    # Load best model
    estimator = joblib.load(MODELS_DIR / "best_model.joblib")

    # Evaluate best model
    metrics, y_pred = evaluate_best_model(
        estimator, X_train, X_test, y_train, y_test, best_name, is_log
    )

    # Build comparison table (CV MAE from meta for all models)
    rows = []
    for name, cv_info in meta["cv_scores"].items():
        row = {
            "Model": name,
            "CV_MAE_mean_INR": cv_info["cv_mae_inr"],
            "CV_MAE_std_INR": cv_info["cv_std_inr"],
        }
        if name == best_name:
            row["Test_MAE_INR"] = metrics["MAE"]
            row["Test_RMSE_INR"] = metrics["RMSE"]
            row["Test_R2"] = metrics["R2"]
        else:
            row["Test_MAE_INR"] = np.nan
            row["Test_RMSE_INR"] = np.nan
            row["Test_R2"] = np.nan
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

    y_tr_arr = np.log1p(y_train.values) if is_log else y_train.values
    plot_learning_curve(estimator, X_train, y_tr_arr, best_name, is_log)

    # Save best model metrics to JSON for README
    results_path = MODELS_DIR / "best_model_metrics.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "test_mae_inr": metrics["MAE"],
                "test_rmse_inr": metrics["RMSE"],
                "test_r2": metrics["R2"],
                "cv_mae_inr": metrics["CV_MAE_mean"],
                "cv_std_inr": metrics["CV_MAE_std"],
            },
            f,
            indent=2,
        )
    print(f"\n  Saved metrics to {results_path}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
