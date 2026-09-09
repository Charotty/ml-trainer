"""Inner-CV ensemble weight tuner."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.models.ensemble.estimators import copy_estimator, fit_kwargs_for_model

PACKAGE_STATUS = "production"


def sanitize_predictions(pred: np.ndarray, fallback: float) -> np.ndarray:
    out = np.asarray(pred, dtype=float).reshape(-1)
    bad = ~np.isfinite(out)
    if bad.any():
        out = out.copy()
        out[bad] = fallback
    return out


def optimize_ensemble_weights(
    models: Mapping[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    target_name: str,
    sample_weight=None,
    *,
    adaptive_priors: Mapping[str, float] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """MAE-minimizing blend of member predictions (L-BFGS-B, same as legacy trainer)."""
    if verbose:
        print(f"\n[FE] Optimizing ensemble weights for {target_name}...")

    fallback = float(np.nanmedian(y_train)) if len(y_train) else 0.0
    if not np.isfinite(fallback):
        fallback = 0.0

    model_predictions = {}
    for model_name, model in models.items():
        model_copy = copy_estimator(model)
        fit_kwargs = fit_kwargs_for_model(model_name, sample_weight)
        try:
            model_copy.fit(X_train, y_train, **fit_kwargs)
            pred = sanitize_predictions(model_copy.predict(X_val), fallback)
        except Exception as exc:
            if verbose:
                print(f"  [WARN] {model_name} fit failed during weight search: {exc}")
            pred = np.full(len(y_val), fallback, dtype=float)
        model_predictions[model_name] = pred

    def objective_function(weights):
        weights = np.abs(weights) / np.sum(np.abs(weights))
        ensemble_pred = np.zeros(len(y_val))
        for i, pred in enumerate(model_predictions.values()):
            ensemble_pred += weights[i] * pred
        ensemble_pred = sanitize_predictions(ensemble_pred, fallback)
        return mean_absolute_error(y_val, ensemble_pred)

    initial_weights = np.ones(len(models)) / len(models)
    bounds = [(0, 1) for _ in range(len(models))]
    try:
        result = minimize(
            objective_function,
            initial_weights,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100},
        )
        raw_w = np.abs(np.asarray(result.x, dtype=float))
        denom = float(np.sum(raw_w))
        if not np.isfinite(denom) or denom <= 0 or not np.all(np.isfinite(raw_w)):
            raise ValueError("non-finite optimized weights")
        optimal_weights = raw_w / denom
    except Exception as exc:
        if verbose:
            print(f"  [WARN] Weight optimization failed, using adaptive priors: {exc}")
        priors = adaptive_priors or {}
        total = sum(priors.get(n, 1.0) for n in models.keys()) or len(models)
        optimal_weights = np.array(
            [priors.get(n, 1.0) / total for n in models.keys()],
            dtype=float,
        )

    optimized_weights = {
        model_name: float(optimal_weights[i])
        for i, model_name in enumerate(models.keys())
    }
    if verbose:
        print(f"  [OK] Optimized weights: {optimized_weights}")

    equal_weights = {name: 1.0 / len(models) for name in models.keys()}
    equal_pred = np.zeros(len(y_val))
    optimized_pred = np.zeros(len(y_val))
    for model_name, pred in model_predictions.items():
        equal_pred += equal_weights[model_name] * pred
        optimized_pred += optimized_weights[model_name] * pred
    equal_pred = sanitize_predictions(equal_pred, fallback)
    optimized_pred = sanitize_predictions(optimized_pred, fallback)
    equal_mae = mean_absolute_error(y_val, equal_pred)
    optimized_mae = mean_absolute_error(y_val, optimized_pred)
    if verbose and equal_mae > 0:
        improvement = ((equal_mae - optimized_mae) / equal_mae) * 100
        print(f"  Improvement: {improvement:.1f}% (MAE: {equal_mae:.3f} -> {optimized_mae:.3f})")
    return optimized_weights


def average_groupkfold_weights(
    models: Mapping[str, Any],
    X_train,
    y_train,
    groups,
    target_name: str,
    sample_weight=None,
    *,
    n_splits: int = 3,
    adaptive_priors: Mapping[str, float] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, float]]:
    """Average per-fold optimized ensemble weights (patient-level GroupKFold)."""
    groups = np.asarray(groups)
    n_unique = len(np.unique(groups))
    n_splits = min(int(n_splits), n_unique)
    if n_splits < 2:
        priors = adaptive_priors or {}
        total = sum(priors.get(n, 1.0) for n in models.keys()) or len(models)
        averaged = {name: priors.get(name, 1.0) / total for name in models.keys()}
        return averaged, [dict(averaged)], {name: 0.0 for name in models.keys()}

    splitter = GroupKFold(n_splits=n_splits)
    fold_weights: list[dict[str, float]] = []
    for train_idx, val_idx in splitter.split(X_train, y_train, groups=groups):
        w_fold = sample_weight[train_idx] if sample_weight is not None else None
        fold_weights.append(
            optimize_ensemble_weights(
                models,
                X_train[train_idx],
                y_train[train_idx],
                X_train[val_idx],
                y_train[val_idx],
                target_name,
                sample_weight=w_fold,
                adaptive_priors=adaptive_priors,
                verbose=verbose,
            )
        )
    traces = [{str(k): float(v) for k, v in fw.items()} for fw in fold_weights]
    variance = {name: float(np.var([fw[name] for fw in fold_weights])) for name in models.keys()}
    averaged = {name: float(np.mean([fw[name] for fw in fold_weights])) for name in models.keys()}
    total = sum(averaged.values()) or 1.0
    averaged = {name: w / total for name, w in averaged.items()}
    return averaged, traces, variance
