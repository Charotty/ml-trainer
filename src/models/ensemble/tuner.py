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


def stacking_weights_groupkfold(
    models: Mapping[str, Any],
    X_train,
    y_train,
    groups,
    sample_weight=None,
    *,
    n_splits: int = 3,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Non-negative ridge stacker on inner GroupKFold OOF member predictions.

    Returns normalized non-negative weights (sum 1). Falls back to equal
    weights when the stacker collapses to all-zero coefficients.
    """
    from sklearn.linear_model import Ridge

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    groups = np.asarray(groups)
    names = list(models.keys())
    n_splits = min(int(n_splits), len(np.unique(groups)))
    equal = {n: 1.0 / len(names) for n in names}
    if n_splits < 2:
        return equal
    fallback = float(np.nanmedian(y_train)) if len(y_train) else 0.0
    oof = np.full((len(y_train), len(names)), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X_train, y_train, groups=groups):
        w_tr = sample_weight[tr] if sample_weight is not None else None
        for j, name in enumerate(names):
            est = copy_estimator(models[name])
            try:
                est.fit(X_train[tr], y_train[tr], **fit_kwargs_for_model(name, w_tr))
                oof[va, j] = sanitize_predictions(est.predict(X_train[va]), fallback)
            except Exception:
                oof[va, j] = fallback
    mask = np.all(np.isfinite(oof), axis=1) & np.isfinite(y_train)
    if int(mask.sum()) < 3:
        return equal
    stacker = Ridge(alpha=alpha, positive=True, fit_intercept=False)
    stacker.fit(oof[mask], y_train[mask])
    coef = np.clip(np.asarray(stacker.coef_, dtype=float), 0.0, None)
    if not np.isfinite(coef).all() or coef.sum() <= 0:
        return equal
    coef = coef / coef.sum()
    return {name: float(coef[j]) for j, name in enumerate(names)}


SMALL_N_GBT_GRID: tuple[dict[str, Any], ...] = tuple(
    {"max_depth": d, "n_estimators": n, "learning_rate": lr, "min_samples_leaf": leaf, "subsample": 0.7}
    for d in (2, 3)
    for (n, lr) in ((100, 0.05), (300, 0.03))
    for leaf in (8, 15)
)
SMALL_N_RF_GRID: tuple[dict[str, Any], ...] = tuple(
    {"max_depth": d, "min_samples_leaf": leaf, "max_features": mf}
    for d in (4, 8)
    for leaf in (8, 15)
    for mf in (0.3, 0.5)
)


def select_tree_params_groupkfold(
    base_models: Mapping[str, Any],
    X_train,
    y_train,
    groups,
    sample_weight=None,
    *,
    n_splits: int = 3,
    gbt_grid: tuple[dict[str, Any], ...] = SMALL_N_GBT_GRID,
    rf_grid: tuple[dict[str, Any], ...] = SMALL_N_RF_GRID,
) -> dict[str, dict[str, Any]]:
    """Pick RF/GBT params by inner GroupKFold MAE (outer fold never seen)."""
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    groups = np.asarray(groups)
    n_splits = min(int(n_splits), len(np.unique(groups)))
    if n_splits < 2:
        return {}
    splits = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups=groups))
    chosen: dict[str, dict[str, Any]] = {}
    for name, grid in (("GradientBoosting", gbt_grid), ("RandomForest", rf_grid)):
        if name not in base_models:
            continue
        best_params, best_mae = None, float("inf")
        for params in grid:
            errs = []
            for tr, va in splits:
                est = copy_estimator(base_models[name])
                est.set_params(**params)
                w_tr = sample_weight[tr] if sample_weight is not None else None
                try:
                    est.fit(X_train[tr], y_train[tr], **fit_kwargs_for_model(name, w_tr))
                    pred = est.predict(X_train[va])
                    errs.append(float(np.mean(np.abs(pred - y_train[va]))))
                except Exception:
                    errs.append(float("inf"))
            mae = float(np.mean(errs))
            if mae < best_mae:
                best_mae, best_params = mae, dict(params)
        if best_params is not None:
            chosen[name] = best_params
    return chosen
