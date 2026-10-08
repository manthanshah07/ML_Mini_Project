"""
train.py
========
Trains and tunes multiple regression models via GridSearchCV / RandomizedSearchCV.
Models compared:
  1. Ridge Regression (baseline linear, log-target)
  2. Lasso Regression (linear with L1, log-target)
  3. Random Forest Regressor
  4. XGBoost Regressor
  5. Gradient Boosting Regressor

Best model is selected by 5-fold CV MAE on the training set (NOT by test score).
The best model's full pipeline is saved to models/.

Run with:
    python -m src.train
"""

import json
import pathlib
import warnings
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocess import (
    DATA_PATH,
    RANDOM_STATE,
    build_pipeline,
    get_feature_names,
    load_data,
    split_data,
)

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

CV_FOLDS = 5
SCORING = "neg_mean_absolute_error"  # maximise = minimise MAE


# ── Log-target helpers ────────────────────────────────────────────────────────

def log_target(y: pd.Series) -> np.ndarray:
    """Apply log1p transform to salary (reduces skew for linear models)."""
    return np.log1p(y.values)


def inv_log_target(y_log: np.ndarray) -> np.ndarray:
    """Inverse of log1p."""
    return np.expm1(y_log)


# ── Model definitions ─────────────────────────────────────────────────────────

def _get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict of model configs.  Each entry has:
      - 'model'        : sklearn estimator
      - 'param_grid'   : dict of hyperparameter search space
      - 'search'       : 'grid' or 'random'
      - 'scale'        : whether to StandardScale numeric inputs
      - 'log_target'   : whether to train on log1p(salary)
    """
    return {
        "Ridge": {
            "model": Ridge(random_state=RANDOM_STATE),
            "param_grid": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]},
            "search": "grid",
            "scale": True,
            "log_target": True,
        },
        "Lasso": {
            "model": Lasso(random_state=RANDOM_STATE, max_iter=10_000),
            "param_grid": {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
            "search": "grid",
            "scale": True,
            "log_target": True,
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": [0.5, "sqrt"],
            },
            "search": "random",
            "scale": False,
            "log_target": False,
            "n_iter": 20,
        },
        "XGBoost": {
            "model": XGBRegressor(
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
                tree_method="hist",
            ),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__subsample": [0.7, 1.0],
                "model__colsample_bytree": [0.7, 1.0],
                "model__reg_lambda": [1, 3],
            },
            "search": "random",
            "scale": False,
            "log_target": False,
            "n_iter": 25,
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "param_grid": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            },
            "search": "random",
            "scale": False,
            "log_target": False,
            "n_iter": 15,
        },
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Any]:
    """
    Trains all model configurations via cross-validated hyperparameter search.
    Returns a dict keyed by model name with the fitted SearchCV object and metadata.
    """
    configs = _get_model_configs()
    results: Dict[str, Any] = {}

    for name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        # Build preprocessing pipeline (with/without scaling)
        preproc = build_pipeline(scale_numeric=cfg["scale"])

        # Full pipeline: preprocess → model
        full_pipe = Pipeline(
            steps=[
                ("preprocess", preproc),
                ("model", cfg["model"]),
            ]
        )

        # Choose target (raw or log1p)
        y_tr = log_target(y_train) if cfg["log_target"] else y_train.values

        # Choose search strategy
        if cfg["search"] == "grid":
            searcher = GridSearchCV(
                full_pipe,
                param_grid=cfg["param_grid"],
                cv=CV_FOLDS,
                scoring=SCORING,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
        else:  # random
            searcher = RandomizedSearchCV(
                full_pipe,
                param_distributions=cfg["param_grid"],
                n_iter=cfg.get("n_iter", 20),
                cv=CV_FOLDS,
                scoring=SCORING,
                n_jobs=-1,
                refit=True,
                random_state=RANDOM_STATE,
                verbose=0,
            )

        searcher.fit(X_train, y_tr)

        best_cv_mae = -searcher.best_score_  # convert neg_MAE to positive MAE

        # For log-target models, recompute MAE in original INR scale via CV
        if cfg["log_target"]:
            cv_scores = cross_val_score(
                searcher.best_estimator_,
                X_train,
                y_tr,
                cv=CV_FOLDS,
                scoring=SCORING,
                n_jobs=-1,
            )
            # Approximate original-scale MAE via delta method: |Δy| ≈ |Δlog_y| * median_salary
            # But a more accurate approach: use predictions on hold-out folds
            # We'll compute it properly in evaluate.py
            cv_log_mae = -cv_scores.mean()
            # Rough scale-back: expm1(log1p(median) + delta) ≈ median * (exp(delta)-1)
            # Keep log-scale CV MAE for ranking; evaluate.py computes true MAE
            best_cv_mae = cv_log_mae  # in log-space; will be converted in evaluate.py

        print(f"  Best params : {searcher.best_params_}")
        print(f"  CV MAE      : {best_cv_mae:,.4f} {'(log-space)' if cfg['log_target'] else '(INR)'}")

        results[name] = {
            "searcher": searcher,
            "best_estimator": searcher.best_estimator_,
            "best_params": searcher.best_params_,
            "cv_mae_raw": best_cv_mae,
            "log_target": cfg["log_target"],
            "scale": cfg["scale"],
        }

    return results


def select_best_model(
    results: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[str, Any]:
    """
    Selects the best model by 5-fold CV MAE on the *original INR scale*.
    For log-target models, converts predictions back before computing MAE.

    NOTE: Selection is done purely on CV performance — the held-out test set
    is NOT used here to avoid data leakage in model selection.

    Returns (best_name, best_estimator).
    """
    print("\n" + "="*60)
    print("  Selecting best model by CV MAE (original INR scale)")
    print("="*60)

    cv_maes: Dict[str, Tuple[float, float]] = {}

    for name, res in results.items():
        estimator = res["best_estimator"]
        is_log = res["log_target"]

        y_tr = log_target(y_train) if is_log else y_train.values

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_maes = []

        for train_idx, val_idx in kf.split(X_train):
            X_f_tr = X_train.iloc[train_idx]
            X_f_val = X_train.iloc[val_idx]
            y_f_tr = y_tr[train_idx] if isinstance(y_tr, np.ndarray) else y_tr.iloc[train_idx]
            y_f_val_orig = y_train.values[val_idx]

            estimator.fit(X_f_tr, y_f_tr)
            preds = estimator.predict(X_f_val)
            if is_log:
                preds = inv_log_target(preds)
            mae = np.mean(np.abs(preds - y_f_val_orig))
            fold_maes.append(mae)

        mean_mae = np.mean(fold_maes)
        std_mae = np.std(fold_maes)
        cv_maes[name] = (mean_mae, std_mae)
        print(f"  {name:<20} CV MAE = {mean_mae:>12,.0f} ± {std_mae:>10,.0f} INR")

    # Re-fit best estimator on full training data (it was modified in the fold loop)
    best_name = min(cv_maes, key=lambda k: cv_maes[k][0])
    print(f"\n  ✓ Best model: {best_name} (CV MAE = {cv_maes[best_name][0]:,.0f} INR)")

    # Store final CV scores for reporting
    for name, res in results.items():
        res["cv_mae_inr"], res["cv_std_inr"] = cv_maes[name]

    return best_name, results


def save_artifacts(best_name: str, results: Dict[str, Any]) -> None:
    """Saves the best model pipeline and metadata to models/."""
    best = results[best_name]
    estimator = best["best_estimator"]

    # Refit on all training data is already done in select_best_model loop
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(estimator, model_path)
    print(f"\n  ✓ Saved model to {model_path}")

    # Save metadata (CV scores, best params) as JSON
    meta = {
        "best_model": best_name,
        "best_params": best["best_params"],
        "log_target": best["log_target"],
        "cv_scores": {
            name: {
                "cv_mae_inr": res["cv_mae_inr"],
                "cv_std_inr": res["cv_std_inr"],
            }
            for name, res in results.items()
        },
    }
    meta_path = MODELS_DIR / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Saved metadata to {meta_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    # Save test split for evaluate.py
    test_split_path = MODELS_DIR / "test_split.joblib"
    joblib.dump((X_test, y_test), test_split_path)
    print(f"Test split saved to {test_split_path}")

    print("\nStarting training …")
    results = train_all(X_train, y_train)

    best_name, results = select_best_model(results, X_train, y_train)

    # Refit best model on full training data
    best_res = results[best_name]
    estimator = best_res["best_estimator"]
    y_tr = log_target(y_train) if best_res["log_target"] else y_train.values
    estimator.fit(X_train, y_tr)
    print(f"\n  Best model refitted on full training set.")

    save_artifacts(best_name, results)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
