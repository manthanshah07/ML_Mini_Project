"""
train.py
========
Trains and tunes multiple regression models via GridSearchCV / RandomizedSearchCV.
Models compared:
  1. Ridge Regression (linear, log-target via TransformedTargetRegressor)
  2. Lasso Regression (L1-penalized linear, log-target via TransformedTargetRegressor)
  3. Random Forest Regressor
  4. XGBoost Regressor
  5. Gradient Boosting Regressor

All scoring is evaluated directly on INR scale MAE.
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
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocess import (
    DATA_PATH,
    RANDOM_STATE,
    build_pipeline,
    data_sha256,
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
ARTIFACT_SCHEMA_VERSION = 2


# ── Model definitions ─────────────────────────────────────────────────────────

def _get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict of model configs. Each entry has:
      - 'model'        : sklearn estimator (optionally wrapped in TransformedTargetRegressor)
      - 'param_grid'   : dict of hyperparameter search space
      - 'search'       : 'grid' or 'random'
      - 'scale'        : whether to StandardScale numeric inputs
      - 'log_target'   : whether the model is trained on log1p(salary)
    """
    return {
        "Ridge": {
            "model": TransformedTargetRegressor(
                regressor=Ridge(random_state=RANDOM_STATE),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "param_grid": {"model__regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]},
            "search": "grid",
            "scale": True,
            "log_target": True,
        },
        "Lasso": {
            "model": TransformedTargetRegressor(
                regressor=Lasso(random_state=RANDOM_STATE, max_iter=10_000),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "param_grid": {"model__regressor__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
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
    X_test: pd.DataFrame = None,
    y_test: pd.Series = None,
) -> Dict[str, Any]:
    """
    Trains all model configurations via cross-validated hyperparameter search.
    TransformedTargetRegressor guarantees that CV scores are evaluated in INR MAE.
    """
    configs = _get_model_configs()
    results: Dict[str, Any] = {}

    for name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        # Build preprocessing pipeline
        preproc = build_pipeline(scale_numeric=cfg["scale"])

        # Full pipeline: preprocess → model
        full_pipe = Pipeline(
            steps=[
                ("preprocess", preproc),
                ("model", cfg["model"]),
            ]
        )

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
        else:
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

        # Fit directly on INR target
        searcher.fit(X_train, y_train.values)

        best_cv_mae = -searcher.best_score_
        best_cv_std = searcher.cv_results_["std_test_score"][searcher.best_index_]

        print(f"  Best params : {searcher.best_params_}")
        print(f"  CV MAE      : {best_cv_mae:>12,.0f} ± {best_cv_std:>10,.0f} INR")

        # Evaluate on held-out test split for complete comparison
        test_metrics = {}
        if X_test is not None and y_test is not None:
            preds = searcher.best_estimator_.predict(X_test)
            mae = mean_absolute_error(y_test.values, preds)
            rmse = np.sqrt(mean_squared_error(y_test.values, preds))
            r2 = r2_score(y_test.values, preds)
            test_metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
            print(f"  Test MAE    : {mae:>12,.0f} INR | RMSE: {rmse:>10,.0f} | R²: {r2:.4f}")

        results[name] = {
            "searcher": searcher,
            "best_estimator": searcher.best_estimator_,
            "best_params": searcher.best_params_,
            "cv_mae_inr": float(best_cv_mae),
            "cv_std_inr": float(best_cv_std),
            "test_metrics": test_metrics,
            "log_target": cfg["log_target"],
            "scale": cfg["scale"],
        }

    return results


def select_best_model(results: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Selects the best model by 5-fold CV MAE on the training set.
    """
    best_name = min(results, key=lambda k: results[k]["cv_mae_inr"])
    best_res = results[best_name]
    print(f"\n{'='*60}")
    print(f"  ✓ Best model: {best_name} (CV MAE = {best_res['cv_mae_inr']:,.0f} ± {best_res['cv_std_inr']:,.0f} INR)")
    print(f"{'='*60}")
    return best_name, results


def save_artifacts(
    best_name: str,
    results: Dict[str, Any],
    X_train: pd.DataFrame = None,
    y_train: pd.Series = None,
) -> None:
    """Save a self-consistent set of model artifacts to ``models/``.

    Training can change the selected model and the transformed feature space.
    Keep the metrics and feature names in sync with the newly saved estimator,
    and remove any SHAP explainer generated for a previous model. Explainability
    artifacts are regenerated by ``src.explain``.
    """
    best = results[best_name]
    estimator = best["best_estimator"]

    error_band_inr = None
    if X_train is not None and y_train is not None:
        # Use out-of-fold residuals from the selected fixed-parameter estimator
        # to estimate an empirical 90th-percentile error band without using the
        # held-out test set.
        oof_predictions = cross_val_predict(
            estimator,
            X_train,
            y_train.values,
            cv=CV_FOLDS,
            n_jobs=-1,
        )
        error_band_inr = float(
            np.quantile(np.abs(y_train.values - oof_predictions), 0.90)
        )

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(estimator, model_path)
    print(f"\n  ✓ Saved best model pipeline to {model_path}")

    feature_names_path = MODELS_DIR / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(get_feature_names(estimator), f, indent=2)
    print(f"  ✓ Saved feature names to {feature_names_path}")

    # A SHAP explainer is tied to both the estimator and its transformed
    # feature matrix. Never leave one from an earlier training run in place.
    shap_path = MODELS_DIR / "shap_explainer.joblib"
    if shap_path.exists():
        shap_path.unlink()
        print(f"  ✓ Removed stale SHAP explainer: {shap_path}")

    # Save metadata (CV scores, best params, test metrics)
    meta = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "data_sha256": data_sha256(DATA_PATH),
        "error_band_inr": error_band_inr,
        "error_band_quantile": 0.90,
        "best_model": best_name,
        "best_params": best["best_params"],
        "log_target": best["log_target"],
        "cv_scores": {
            name: {
                "cv_mae_inr": res["cv_mae_inr"],
                "cv_std_inr": res["cv_std_inr"],
                "test_mae_inr": res["test_metrics"].get("MAE"),
                "test_rmse_inr": res["test_metrics"].get("RMSE"),
                "test_r2": res["test_metrics"].get("R2"),
            }
            for name, res in results.items()
        },
    }
    meta_path = MODELS_DIR / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Saved metadata to {meta_path}")

    metrics_path = MODELS_DIR / "best_model_metrics.json"
    test_metrics = best["test_metrics"]
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "data_sha256": data_sha256(DATA_PATH),
                "best_model": best_name,
                "test_mae_inr": test_metrics.get("MAE"),
                "test_rmse_inr": test_metrics.get("RMSE"),
                "test_r2": test_metrics.get("R2"),
                "error_band_inr": error_band_inr,
                "error_band_quantile": 0.90,
                "cv_mae_inr": best["cv_mae_inr"],
                "cv_std_inr": best["cv_std_inr"],
            },
            f,
            indent=2,
        )
    print(f"  ✓ Saved metrics to {metrics_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    # Save test split for evaluate.py and explain.py
    test_split_path = MODELS_DIR / "test_split.joblib"
    joblib.dump((X_test, y_test), test_split_path)
    print(f"Test split saved to {test_split_path}")

    print("\nStarting training …")
    results = train_all(X_train, y_train, X_test, y_test)

    best_name, results = select_best_model(results)
    save_artifacts(best_name, results, X_train, y_train)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
