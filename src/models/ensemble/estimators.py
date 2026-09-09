"""Estimator factory: create base models and voting ensembles."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Lasso, Ridge

PACKAGE_STATUS = "production"

MODEL_KIND_ALIASES = {
    "ensemble": "ensemble",
    "adaptive_ensemble": "ensemble",
    "rf": "rf",
    "randomforest": "rf",
    "random_forest": "rf",
    "gbt": "gbt",
    "gb": "gbt",
    "gradientboosting": "gbt",
    "gradient_boosting": "gbt",
    "ridge": "ridge",
    "mean": "mean",
    "median": "median",
}
SINGLE_KIND_TO_NAME = {
    "rf": "RandomForest",
    "gbt": "GradientBoosting",
    "ridge": "Ridge",
}

DEFAULT_BEST_MODELS = {
    "kidney_left_delta_x": "RandomForest",
    "kidney_left_delta_y": "GradientBoosting",
    "kidney_left_delta_z": "GradientBoosting",
    "kidney_right_delta_x": "Ridge",
    "kidney_right_delta_y": "GradientBoosting",
    "kidney_right_delta_z": "GradientBoosting",
}

DEFAULT_ADAPTIVE_WEIGHTS = {
    "kidney_left_delta_x": {
        "RandomForest": 2.5,
        "Lasso": 1.0,
        "Ridge": 0.8,
        "GradientBoosting": 0.6,
    },
    "kidney_left_delta_y": {
        "RandomForest": 1.2,
        "Lasso": 1.0,
        "Ridge": 0.8,
        "GradientBoosting": 2.2,
    },
    "kidney_left_delta_z": {
        "RandomForest": 1.0,
        "Lasso": 0.6,
        "Ridge": 0.6,
        "GradientBoosting": 2.8,
    },
    "kidney_right_delta_x": {
        "RandomForest": 1.0,
        "Lasso": 0.8,
        "Ridge": 2.5,
        "GradientBoosting": 0.6,
    },
    "kidney_right_delta_y": {
        "RandomForest": 1.2,
        "Lasso": 1.0,
        "Ridge": 0.8,
        "GradientBoosting": 2.2,
    },
    "kidney_right_delta_z": {
        "RandomForest": 1.0,
        "Lasso": 0.6,
        "Ridge": 0.6,
        "GradientBoosting": 2.8,
    },
}


def make_base_models(
    *,
    estimator_profile: str = "production",
    target_name: str | None = None,
) -> dict[str, Any]:
    """Create unfitted RF/Lasso/Ridge/GBT members for one target."""
    if estimator_profile not in {"production", "tiny"}:
        raise ValueError(
            f"estimator_profile must be 'production' or 'tiny', got {estimator_profile!r}"
        )
    axis = target_name.split("_")[-1] if target_name else None

    if estimator_profile == "tiny":
        rf_config = {
            "n_estimators": 4,
            "max_depth": 2,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": 1,
        }
        gbt_config = {
            "n_estimators": 4,
            "learning_rate": 0.2,
            "max_depth": 2,
            "subsample": 1.0,
            "random_state": 42,
        }
        lasso_config = {"alpha": 0.5, "max_iter": 2000, "random_state": 42}
        ridge_config = {"alpha": 1.0, "solver": "auto", "random_state": 42}
    else:
        rf_config = {
            "n_estimators": 600 if axis in ("y", "z") else 500,
            "max_depth": 24 if axis in ("y", "z") else 20,
            "min_samples_split": 6 if axis in ("y", "z") else 10,
            "min_samples_leaf": 2 if axis in ("y", "z") else 4,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
        gbt_config = {
            "n_estimators": 700 if axis == "z" else 550,
            "learning_rate": 0.04 if axis in ("y", "z") else 0.05,
            "max_depth": 7 if axis == "z" else 6,
            "subsample": 0.85,
            "random_state": 42,
        }
        if axis == "z":
            gbt_config["loss"] = "huber"
            gbt_config["alpha"] = 0.9
        lasso_config = {
            "alpha": 0.08 if axis in ("y", "z") else 0.1,
            "max_iter": 5000,
            "random_state": 42,
        }
        ridge_config = {
            "alpha": 0.8 if axis in ("y", "z") else 1.0,
            "solver": "auto",
            "random_state": 42,
        }

    return {
        "RandomForest": RandomForestRegressor(**rf_config),
        "Lasso": Lasso(**lasso_config),
        "Ridge": Ridge(**ridge_config),
        "GradientBoosting": GradientBoostingRegressor(**gbt_config),
    }


def make_single_estimator(kind: str, base_models: Mapping[str, Any]) -> tuple[str, Any]:
    """Return (estimator_name, unfitted estimator) for a non-ensemble kind."""
    if kind in ("mean", "median"):
        return kind, DummyRegressor(strategy=kind)
    name = SINGLE_KIND_TO_NAME[kind]
    return name, clone(base_models[name])


def copy_estimator(model: Any) -> Any:
    try:
        return clone(model)
    except Exception:
        if hasattr(model, "get_params"):
            return type(model)(**model.get_params())
        return type(model)()


def fit_kwargs_for_model(model_name: str, sample_weight: np.ndarray | None) -> dict:
    del model_name
    if sample_weight is None:
        return {}
    return {"sample_weight": sample_weight}


def create_optimized_voting_ensemble(
    models: Mapping[str, Any],
    target_name: str,
    optimized_weights: Mapping[str, float],
    *,
    adaptive_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> VotingRegressor:
    estimators = [(name, clone(models[name])) for name in models.keys()]
    weights = np.array([optimized_weights[name] for name in models.keys()], dtype=float)
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
        target_weights = (adaptive_weights or DEFAULT_ADAPTIVE_WEIGHTS).get(target_name, {})
        weights = np.array(
            [target_weights.get(name, 1.0) for name in models.keys()],
            dtype=float,
        )
        if float(weights.sum()) <= 0:
            weights = np.ones(len(models), dtype=float)
        weights = weights / weights.sum()
    return VotingRegressor(estimators=estimators, weights=weights.tolist(), n_jobs=1)


def create_adaptive_voting_ensemble(
    models: Mapping[str, Any],
    target_name: str,
    *,
    adaptive_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> VotingRegressor:
    target_weights = (adaptive_weights or DEFAULT_ADAPTIVE_WEIGHTS)[target_name]
    estimators = []
    weights = []
    for model_name, weight in target_weights.items():
        if model_name in models:
            estimators.append((model_name, clone(models[model_name])))
            weights.append(weight)
    return VotingRegressor(estimators=estimators, weights=weights, n_jobs=1)


def create_standard_voting_ensemble(models: Mapping[str, Any]) -> VotingRegressor:
    estimators = [(name, clone(models[name])) for name in models.keys()]
    return VotingRegressor(estimators=estimators, weights=None, n_jobs=1)


def fit_voting_ensemble(
    ensemble: VotingRegressor,
    X,
    y,
    sample_weight=None,
) -> None:
    if sample_weight is None:
        ensemble.fit(X, y)
        return
    named = getattr(ensemble, "named_estimators", None)
    if named is None:
        ensemble.fit(X, y)
        return
    fitted = []
    for name, est in named.items():
        est_fitted = copy_estimator(est)
        est_fitted.fit(X, y, **fit_kwargs_for_model(name, sample_weight))
        fitted.append(est_fitted)
    ensemble.estimators_ = fitted
