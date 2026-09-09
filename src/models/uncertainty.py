"""Honest prediction intervals from OOF residuals (conformal).

This module must not invent a scalar 'confidence' from prediction magnitude.
Intervals are split-conformal on absolute residuals collected on outer folds.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_COVERAGE = 0.90
UNAVAILABLE_NOTE = (
    "No calibrated uncertainty is available for this artifact. "
    "Magnitude-based fake confidence (1 - |pred|/50) was removed and is not used."
)
METHOD_CONFORMAL = "oof_residual_conformal"
METHOD_UNAVAILABLE = "unavailable"


def _finite_abs_resid(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    return np.abs(yp[mask] - yt[mask])


def residual_quantile(abs_resid: np.ndarray, coverage: float) -> float:
    vals = np.sort(np.asarray(abs_resid, dtype=float))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    # Finite-sample conformal correction: ceil((n+1)*coverage) order statistic.
    n = vals.size
    k = int(np.ceil((n + 1) * float(coverage))) - 1
    k = min(n - 1, max(0, k))
    return float(vals[k])


def empirical_coverage(abs_resid: np.ndarray, q_hat: float) -> float:
    vals = np.asarray(abs_resid, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or not np.isfinite(q_hat):
        return float("nan")
    return float(np.mean(vals <= q_hat))


def fit_conformal_from_oof(
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    targets: Sequence[str],
    coverage: float = DEFAULT_COVERAGE,
    fold_masks: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Fit absolute-residual conformal intervals; check coverage on outer folds.

    ``fold_masks[i]`` is True for rows in outer-validation fold i. For each
    fold, ``q`` is estimated on the complementary rows and scored on the fold
    (honest coverage). The stored ``q_hat`` used at inference is computed on
    all OOF residuals.
    """
    per_target: dict[str, Any] = {}
    fold_coverages: list[dict[str, Any]] = []
    for target in targets:
        if target not in truth.columns or target not in pred.columns:
            continue
        abs_all = _finite_abs_resid(truth[target].to_numpy(), pred[target].to_numpy())
        q_hat = residual_quantile(abs_all, coverage)
        per_target[target] = {
            "q_hat_mm": q_hat,
            "mean_interval_width_mm": float(2.0 * q_hat) if np.isfinite(q_hat) else float("nan"),
            "n_oof": int(abs_all.size),
            "in_sample_coverage": empirical_coverage(abs_all, q_hat),
        }

        if fold_masks:
            covs: list[float] = []
            widths: list[float] = []
            for mask in fold_masks:
                val_idx = np.asarray(mask, dtype=bool)
                train_idx = ~val_idx
                q_fold = residual_quantile(
                    _finite_abs_resid(
                        truth.loc[train_idx, target].to_numpy(),
                        pred.loc[train_idx, target].to_numpy(),
                    ),
                    coverage,
                )
                covs.append(
                    empirical_coverage(
                        _finite_abs_resid(
                            truth.loc[val_idx, target].to_numpy(),
                            pred.loc[val_idx, target].to_numpy(),
                        ),
                        q_fold,
                    )
                )
                widths.append(float(2.0 * q_fold) if np.isfinite(q_fold) else float("nan"))
            per_target[target]["fold_coverage_mean"] = float(np.nanmean(covs)) if covs else float("nan")
            per_target[target]["fold_width_mean_mm"] = float(np.nanmean(widths)) if widths else float("nan")
            fold_coverages.append(
                {"target": target, "per_fold_coverage": covs, "per_fold_width_mm": widths}
            )

    return {
        "method": METHOD_CONFORMAL,
        "nominal_coverage": float(coverage),
        "note": (
            "Symmetric intervals pred ± q_hat from OOF |residual| quantiles. "
            "This is not a probability that the point prediction is 'correct'."
        ),
        "per_target": per_target,
        "fold_checks": fold_coverages,
    }


def intervals_for_point_predictions(
    predictions: Mapping[str, float],
    conformal: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build API uncertainty payload. Never returns a fake 0.5–0.95 confidence."""
    if not conformal or conformal.get("method") != METHOD_CONFORMAL:
        return {
            "method": METHOD_UNAVAILABLE,
            "nominal_coverage": None,
            "intervals": None,
            "prediction_confidence": None,
            "note": UNAVAILABLE_NOTE,
        }
    per_target = conformal.get("per_target") or {}
    intervals: dict[str, Any] = {}
    for name, pred in predictions.items():
        q_hat = (per_target.get(name) or {}).get("q_hat_mm")
        try:
            p = float(pred)
        except (TypeError, ValueError):
            continue
        if q_hat is None or not np.isfinite(q_hat) or not np.isfinite(p):
            intervals[name] = {
                "point_mm": p if np.isfinite(p) else None,
                "lower_mm": None,
                "upper_mm": None,
                "q_hat_mm": None,
            }
            continue
        q = float(q_hat)
        intervals[name] = {
            "point_mm": p,
            "lower_mm": p - q,
            "upper_mm": p + q,
            "q_hat_mm": q,
        }
    return {
        "method": METHOD_CONFORMAL,
        "nominal_coverage": conformal.get("nominal_coverage", DEFAULT_COVERAGE),
        "intervals": intervals,
        "prediction_confidence": None,
        "note": conformal.get("note"),
    }


def extract_conformal_from_model_data(model_data: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(model_data, Mapping):
        return None
    for key in ("conformal", "conformal_intervals"):
        payload = model_data.get(key)
        if isinstance(payload, Mapping) and payload.get("method") == METHOD_CONFORMAL:
            return dict(payload)
    meta = model_data.get("training_meta")
    if isinstance(meta, Mapping):
        payload = meta.get("conformal")
        if isinstance(payload, Mapping) and payload.get("method") == METHOD_CONFORMAL:
            return dict(payload)
    return None
