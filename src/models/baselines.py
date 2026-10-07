"""Required nested-CV baselines and RF-dominance / simplicity rule (Block 3).

All comparisons go through ``src.models.nested_cv.evaluate_nested_groupkfold_oof``.
This module does not fork a third evaluation path.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.features.na_trend_features import NaTrendStore
from src.features.phase1_schema import TARGET_NAMES
from src.models.clinical_metrics import bootstrap_mean_ci
from src.models.nested_cv import (
    NestedOOFResult,
    TrainerFactory,
    evaluate_nested_groupkfold_oof,
)

BASELINE_KINDS: tuple[str, ...] = (
    "median",
    "mean",
    "group_median",
    "ridge",
    "rf",
    "gbt",
    "ensemble",
)
KIND_ALIASES = {
    "median": "median",
    "mean": "mean",
    "group_median": "group_median",
    "grouped_median": "group_median",
    "ridge": "ridge",
    "rf": "rf",
    "randomforest": "rf",
    "random_forest": "rf",
    "gbt": "gbt",
    "gb": "gbt",
    "gradientboosting": "gbt",
    "gradient_boosting": "gbt",
    "ensemble": "ensemble",
    "adaptive_ensemble": "ensemble",
}


def normalize_model_kind(kind: str) -> str:
    key = str(kind).strip().lower()
    if key in KIND_ALIASES:
        return KIND_ALIASES[key]
    from src.models.ensemble import model_kinds as _mk

    resolved = _mk.resolve(key)
    if resolved is not None:
        return resolved
    raise ValueError(
        f"Unknown model kind {kind!r}; expected one of "
        f"{BASELINE_KINDS + tuple(sorted(set(_mk.all_aliases().values())))}"
    )


def make_trainer_factory(
    kind: str = "ensemble",
    **trainer_kwargs: Any,
) -> TrainerFactory:
    """Factory compatible with ``evaluate_nested_groupkfold_oof``."""
    from src.models.ensemble import AdaptiveEnsembleTrainer

    model_kind = normalize_model_kind(kind)
    params = dict(trainer_kwargs)
    params.setdefault("enrichment_mode", "none")
    params.setdefault("estimator_profile", "tiny")
    params["model_kind"] = model_kind
    if model_kind == "ridge":
        params.setdefault("encode_categoricals", True)

    def _factory(*, na_trend_store: NaTrendStore | None = None, **extra: Any):
        merged = dict(params)
        merged.update(extra)
        if na_trend_store is not None:
            merged["na_trend_store"] = na_trend_store
        return AdaptiveEnsembleTrainer(**merged)

    return _factory


def _fold_mae(result: NestedOOFResult, truth: pd.DataFrame, targets: list[str]) -> list[float]:
    maes: list[float] = []
    groups = np.asarray(result.group_ids, dtype=str)
    pred = result.oof_predictions
    for fold in result.folds:
        val = set(str(g) for g in fold.get("val_groups") or [])
        mask = np.array([g in val for g in groups], dtype=bool)
        if not mask.any():
            maes.append(float("nan"))
            continue
        errs = []
        for t in targets:
            if t not in truth.columns or t not in pred.columns:
                continue
            yt = truth.loc[mask, t].to_numpy(dtype=float)
            yp = pred.loc[mask, t].to_numpy(dtype=float)
            finite = np.isfinite(yt) & np.isfinite(yp)
            if finite.any():
                errs.append(float(np.mean(np.abs(yt[finite] - yp[finite]))))
        maes.append(float(np.mean(errs)) if errs else float("nan"))
    return maes


def paired_outer_fold_comparison(
    ensemble: NestedOOFResult,
    reference: NestedOOFResult,
    truth: pd.DataFrame,
    *,
    targets: list[str] | None = None,
    reference_name: str = "rf",
) -> dict[str, Any]:
    """Paired outer-fold MAE: ensemble vs a single-model reference (usually RF)."""
    cols = list(targets or TARGET_NAMES)
    ens_mae = np.asarray(_fold_mae(ensemble, truth, cols), dtype=float)
    ref_mae = np.asarray(_fold_mae(reference, truth, cols), dtype=float)
    delta = ref_mae - ens_mae  # >0 means ensemble has lower MAE
    finite = np.isfinite(delta)
    mean_delta = float(np.mean(delta[finite])) if finite.any() else float("nan")
    lo, hi = bootstrap_mean_ci(delta[finite]) if finite.any() else (float("nan"), float("nan"))
    return {
        "reference": reference_name,
        "ensemble_fold_mae_mm": [None if not np.isfinite(v) else float(v) for v in ens_mae],
        "reference_fold_mae_mm": [None if not np.isfinite(v) else float(v) for v in ref_mae],
        "delta_reference_minus_ensemble_mm": [
            None if not np.isfinite(v) else float(v) for v in delta
        ],
        "mean_delta_mm": mean_delta,
        "delta_ci95": [lo, hi],
        "n_folds": int(finite.sum()),
    }


def simplicity_rule(
    comparison: Mapping[str, Any],
    *,
    min_absolute_gain_mm: float = 0.25,
) -> dict[str, Any]:
    """Prefer RF when ensemble gain is smaller than uncertainty (or negative)."""
    mean_delta = comparison.get("mean_delta_mm")
    ci = list(comparison.get("delta_ci95") or [float("nan"), float("nan")])
    lo = float(ci[0]) if ci else float("nan")
    hi = float(ci[1]) if len(ci) > 1 else float("nan")
    gain = float(mean_delta) if mean_delta is not None else float("nan")
    if not np.isfinite(gain):
        selected = "rf"
        reason = "ensemble vs RF delta is undefined; default to the simpler model"
    elif gain < min_absolute_gain_mm:
        selected = "rf"
        reason = (
            f"ensemble mean gain {gain:.3f} mm is below the {min_absolute_gain_mm} mm "
            "simplicity threshold"
        )
    elif np.isfinite(lo) and lo <= 0:
        selected = "rf"
        reason = (
            f"ensemble gain CI [{lo:.3f}, {hi:.3f}] includes 0; treat as within uncertainty"
        )
    else:
        selected = "ensemble"
        reason = (
            f"ensemble mean gain {gain:.3f} mm with CI [{lo:.3f}, {hi:.3f}] "
            "exceeds uncertainty"
        )
    return {
        "selected": selected,
        "reason": reason,
        "min_absolute_gain_mm": min_absolute_gain_mm,
        "comparison": dict(comparison),
        "production_note": (
            "This rule is for model-class selection on nested OOF. "
            "It does not promote an artifact."
        ),
    }


def inner_weight_summary(result: NestedOOFResult) -> dict[str, Any]:
    """Persist inner-fold ensemble weights and their variance across outer folds."""
    traces: dict[str, list[dict[str, float]]] = {}
    variances: dict[str, dict[str, list[float]]] = {}
    for fold in result.folds:
        per_target = fold.get("inner_ensemble_weights") or {}
        fold_traces = fold.get("inner_fold_weight_traces") or {}
        fold_var = fold.get("inner_weight_variance") or {}
        for target, weights in per_target.items():
            traces.setdefault(target, []).append({str(k): float(v) for k, v in dict(weights).items()})
        for target, var_map in fold_var.items():
            bucket = variances.setdefault(str(target), {})
            for name, val in dict(var_map).items():
                bucket.setdefault(str(name), []).append(float(val))
        del fold_traces
    mean_weights: dict[str, dict[str, float]] = {}
    weight_var: dict[str, dict[str, float]] = {}
    for target, rows in traces.items():
        names = sorted({k for row in rows for k in row})
        mean_weights[target] = {
            name: float(np.mean([row.get(name, 0.0) for row in rows])) for name in names
        }
        weight_var[target] = {
            name: float(np.var([row.get(name, 0.0) for row in rows])) for name in names
        }
    return {
        "per_outer_fold_weights": traces,
        "mean_weights": mean_weights,
        "variance_across_outer_folds": weight_var,
        "mean_inner_fold_variance": {
            t: {k: float(np.mean(vs)) for k, vs in names.items()}
            for t, names in variances.items()
        },
    }


def evaluate_baselines_nested(
    df: pd.DataFrame,
    *,
    kinds: tuple[str, ...] | list[str] = BASELINE_KINDS,
    n_splits: int = 2,
    trainer_kwargs: Mapping[str, Any] | None = None,
    na_trend_store: NaTrendStore | None = None,
    targets: list[str] | None = None,
    yz_target_boost: bool = False,
) -> dict[str, Any]:
    """Run required baselines on the same nested GroupKFold helper."""
    cols = list(targets or [c for c in TARGET_NAMES if c in df.columns])
    kwargs = dict(trainer_kwargs or {})
    kwargs.setdefault("enrichment_mode", "none")
    kwargs.setdefault("estimator_profile", "tiny")
    kwargs.setdefault("inner_n_splits", 2)
    kwargs["yz_target_boost"] = bool(yz_target_boost)

    results: dict[str, NestedOOFResult] = {}
    reports: dict[str, Any] = {}
    for kind in kinds:
        factory = make_trainer_factory(kind, **kwargs)
        result = evaluate_nested_groupkfold_oof(
            df,
            trainer_factory=factory,
            n_splits=n_splits,
            na_trend_store=na_trend_store,
            targets=cols,
        )
        results[normalize_model_kind(kind)] = result
        reports[normalize_model_kind(kind)] = result.to_report_dict(include_predictions=True)

    comparison = None
    decision = None
    if "ensemble" in results and "rf" in results:
        comparison = paired_outer_fold_comparison(
            results["ensemble"], results["rf"], df.reset_index(drop=True), targets=cols
        )
        decision = simplicity_rule(comparison)

    return {
        "protocol": "nested_groupkfold_baselines",
        "kinds": [normalize_model_kind(k) for k in kinds],
        "n_splits": n_splits,
        "yz_target_boost": bool(yz_target_boost),
        "results": reports,
        "inner_weight_summary": inner_weight_summary(results["ensemble"]) if "ensemble" in results else {},
        "ensemble_vs_rf": comparison,
        "simplicity_rule": decision,
        "note": (
            "Baselines share evaluate_nested_groupkfold_oof. "
            "Do not treat a tiny-n run as a production model decision."
        ),
    }
