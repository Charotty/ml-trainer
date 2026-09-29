"""Production-equivalent nested GroupKFold evaluation (Block 2).

Outer folds are evaluation-only. Inner folds (inside ``evaluate_fold``) tune
ensemble weights. Imputer, scaler, feature drop, and optional NaTrendStore
statistics are refit on each outer-train split.

This module must not import ``scripts.*``. Callers inject a trainer factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.features.na_trend_features import NaTrendStore
from src.features.phase1_schema import TARGET_NAMES
from src.models.clinical_metrics import compute_clinical_report
from src.models.uncertainty import DEFAULT_COVERAGE, fit_conformal_from_oof

TrainerFactory = Callable[..., Any]
NaTrendFactory = Callable[[pd.DataFrame, pd.DataFrame], NaTrendStore | None]
ProxyTrainBuilder = Callable[[pd.DataFrame, Any], pd.DataFrame]

SEED = 42
N_BOOTSTRAP = 2000
Z_TARGETS = ("kidney_left_delta_z", "kidney_right_delta_z")
PROTOCOL_NESTED = "nested_groupkfold"
PROTOCOL_FIXED_PRIOR = "fixed_prior_benchmark"
PROTOCOL_NESTED_PROXY = "nested_proxy_vs_honest"
WEIGHT_INNER = "inner_groupkfold"
WEIGHT_FIXED_PRIOR = "fixed_prior"
DEFAULT_REPEAT_SEEDS = (0, 1, 2)


def patient_group_column(df: pd.DataFrame) -> str:
    if "full_name" in df.columns:
        return "full_name"
    if "case_id" in df.columns:
        return "case_id"
    raise ValueError("DataFrame needs full_name or case_id for GroupKFold")


def patient_groups(df: pd.DataFrame) -> np.ndarray:
    return df[patient_group_column(df)].astype(str).values


def remap_groups_for_seed(groups: np.ndarray, seed: int | None) -> np.ndarray:
    """Prefix group labels with a seeded permutation so GroupKFold splits differ.

    sklearn ``GroupKFold`` sorts unique group labels; remapping with a random
    zero-padded prefix changes fold membership across seeds while keeping
    patients atomically in one fold.
    """
    groups = np.asarray(groups, dtype=str)
    if seed is None:
        return groups
    rng = np.random.default_rng(int(seed))
    unique = np.unique(groups)
    order = rng.permutation(len(unique))
    mapping = {g: f"{order[i]:08d}__{g}" for i, g in enumerate(unique)}
    return np.array([mapping[g] for g in groups], dtype=str)


def _bootstrap_ci(per_patient_avg: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(per_patient_avg)
    if n == 0:
        return float("nan"), float("nan")
    means = [per_patient_avg[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _axis_summary(per_target_mae: Mapping[str, float]) -> dict[str, float]:
    axes: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    for target, mae in per_target_mae.items():
        axis = str(target).split("_")[-1]
        if axis in axes:
            axes[axis].append(float(mae))
    return {axis: float(np.mean(vals)) for axis, vals in axes.items() if vals}


def summarize_oof(
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    cols = list(targets or TARGET_NAMES)
    per_target = {}
    per_rmse = {}
    per_r2 = {}
    for t in cols:
        yt = truth[t].to_numpy(dtype=float)
        yp = pred[t].to_numpy(dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if int(mask.sum()) == 0:
            per_target[t] = float("nan")
            per_rmse[t] = float("nan")
            per_r2[t] = float("nan")
            continue
        per_target[t] = float(mean_absolute_error(yt[mask], yp[mask]))
        per_rmse[t] = float(np.sqrt(mean_squared_error(yt[mask], yp[mask])))
        per_r2[t] = float(r2_score(yt[mask], yp[mask])) if int(mask.sum()) >= 2 else float("nan")
    err_parts = []
    for idx in range(len(truth)):
        vals = []
        for t in cols:
            yt = truth[t].iloc[idx]
            yp = pred[t].iloc[idx]
            if np.isfinite(yt) and np.isfinite(yp):
                vals.append(abs(float(yp) - float(yt)))
        if vals:
            err_parts.append(float(np.mean(vals)))
    abs_err = np.asarray(err_parts, dtype=float) if err_parts else np.asarray([np.nan])
    lo, hi = _bootstrap_ci(abs_err[np.isfinite(abs_err)]) if np.isfinite(abs_err).any() else (float("nan"), float("nan"))
    z_vals = [per_target[t] for t in cols if t in Z_TARGETS and np.isfinite(per_target[t])]
    finite_maes = [v for v in per_target.values() if np.isfinite(v)]
    finite_rmses = [v for v in per_rmse.values() if np.isfinite(v)]
    return {
        "per_target_mae_mm": per_target,
        "per_target_rmse_mm": per_rmse,
        "per_target_r2": per_r2,
        "axis_mae_mm": _axis_summary(per_target),
        "axis_rmse_mm": _axis_summary(per_rmse),
        "axis_r2": _axis_summary(per_r2),
        "avg_mae_mm": float(np.mean(finite_maes)) if finite_maes else float("nan"),
        "avg_rmse_mm": float(np.mean(finite_rmses)) if finite_rmses else float("nan"),
        "avg_r2": float(np.nanmean(list(per_r2.values()))),
        "avg_mae_ci95": [lo, hi],
        "z_avg_mae_mm": float(np.mean(z_vals)) if z_vals else float("nan"),
    }


def _clone_na_trend_store(store: NaTrendStore | None) -> NaTrendStore | None:
    if store is None:
        return None
    return NaTrendStore.from_dict(store.to_dict())


def _drop_groups(df: pd.DataFrame | None, banned: set[str]) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    try:
        col = patient_group_column(df)
    except ValueError:
        return df
    return df[~df[col].astype(str).isin(banned)].copy()


def _read_csv_if_exists(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    return pd.read_csv(file_path)


def na_trend_factory_from_fitted_store(store: NaTrendStore | None) -> NaTrendFactory:
    """Refit aux-cohort stats each outer fold, dropping validation patient ids."""
    if store is None:
        return default_na_trend_store_factory(None)
    return default_na_trend_store_factory(
        store,
        aux_spine_df=_read_csv_if_exists(store.spine_path),
        aux_boku_df=_read_csv_if_exists(store.boku_path),
        aux_kits_df=_read_csv_if_exists(store.kits_path) if store.include_kits else None,
        include_kits=bool(store.include_kits),
    )


def default_na_trend_store_factory(
    base_store: NaTrendStore | None = None,
    *,
    aux_spine_df: pd.DataFrame | None = None,
    aux_boku_df: pd.DataFrame | None = None,
    aux_kits_df: pd.DataFrame | None = None,
    include_kits: bool = False,
) -> NaTrendFactory:
    """Build a per-outer-fold store. Aux frames are refit with val patients dropped."""

    def _factory(train_df: pd.DataFrame, val_df: pd.DataFrame) -> NaTrendStore | None:
        del train_df  # enrichment stats come from aux cohorts, not clinical y
        if aux_spine_df is not None or aux_boku_df is not None or aux_kits_df is not None:
            banned = set(patient_groups(val_df))
            return NaTrendStore.fit_from_frames(
                spine_df=_drop_groups(aux_spine_df, banned),
                boku_df=_drop_groups(aux_boku_df, banned),
                kits_df=_drop_groups(aux_kits_df, banned) if include_kits else None,
                include_kits=include_kits,
                spine_path=getattr(base_store, "spine_path", "") if base_store else "",
                boku_path=getattr(base_store, "boku_path", "") if base_store else "",
                kits_path=getattr(base_store, "kits_path", "") if base_store else "",
            )
        return _clone_na_trend_store(base_store)

    return _factory


def _predict_scaled(trainer: Any, X_scaled: np.ndarray) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for target in trainer.target_names:
        model = trainer.trained_models[target]
        data[str(target)] = np.asarray(model.predict(X_scaled), dtype=float).reshape(-1)
    return pd.DataFrame(data)


def _preprocessor_snapshot(trainer: Any) -> dict[str, Any]:
    names = [str(n) for n in list(trainer.feature_names or [])]
    imputer_stats = getattr(trainer.imputer, "statistics_", None)
    scaler_mean = getattr(trainer.scaler, "mean_", None)
    scaler_scale = getattr(trainer.scaler, "scale_", None)
    out: dict[str, Any] = {
        "feature_names": names,
        "n_features": len(names),
        "imputer_statistics": None,
        "scaler_mean": None,
        "scaler_scale": None,
    }
    if imputer_stats is not None and len(imputer_stats) == len(names):
        out["imputer_statistics"] = {
            name: (None if not np.isfinite(val) else float(val))
            for name, val in zip(names, imputer_stats)
        }
    if scaler_mean is not None and len(scaler_mean) == len(names):
        out["scaler_mean"] = {
            name: (None if not np.isfinite(val) else float(val))
            for name, val in zip(names, scaler_mean)
        }
    if scaler_scale is not None and len(scaler_scale) == len(names):
        out["scaler_scale"] = {
            name: (None if not np.isfinite(val) else float(val))
            for name, val in zip(names, scaler_scale)
        }
    return out


def _assert_no_val_in_proxy_train(proxy_train: pd.DataFrame, val_df: pd.DataFrame) -> None:
    val_ids = set(patient_groups(val_df))
    try:
        col = patient_group_column(proxy_train)
    except ValueError:
        return
    overlap = set(proxy_train[col].astype(str)) & val_ids
    if overlap:
        raise ValueError(
            "Proxy train includes outer-validation patients "
            f"{sorted(overlap)}; teacher/proxy fit must exclude them."
        )


@dataclass
class NestedOOFResult:
    protocol: str
    weight_mode: str
    n_splits: int
    n_patients: int
    metrics: dict[str, Any]
    oof_predictions: pd.DataFrame
    folds: list[dict[str, Any]] = field(default_factory=list)
    group_ids: list[str] = field(default_factory=list)
    conformal: dict[str, Any] | None = None
    clinical_metrics: dict[str, Any] | None = None
    subgroup_metrics: dict[str, Any] | None = None
    worst_cases: list[dict[str, Any]] = field(default_factory=list)

    def to_manifest_fields(self, *, include_predictions: bool = True) -> dict[str, Any]:
        pred_payload: dict[str, Any] | None = None
        if include_predictions:
            pred_payload = {
                "group_ids": list(self.group_ids),
                "targets": {
                    col: [None if not np.isfinite(v) else float(v) for v in self.oof_predictions[col].tolist()]
                    for col in self.oof_predictions.columns
                },
            }
        clinical = dict(self.clinical_metrics or {})
        return {
            "folds": self.folds,
            "oof_predictions": pred_payload,
            "aggregated_metrics": dict(self.metrics),
            "clinical_metrics": clinical or None,
            "tail_metrics": (clinical.get("clinical_3d") if clinical else None),
            "subgroup_metrics": self.subgroup_metrics,
            "worst_cases": list(self.worst_cases),
            "conformal": self.conformal,
            "oof_protocol": self.protocol,
            "oof_status": "computed",
            "oof_weight_mode": self.weight_mode,
        }

    def to_report_dict(self, *, include_predictions: bool = True) -> dict[str, Any]:
        payload = dict(self.metrics)
        payload.update(self.to_manifest_fields(include_predictions=include_predictions))
        payload["n_splits"] = self.n_splits
        payload["n_patients"] = self.n_patients
        payload["protocol"] = self.protocol
        payload["weight_mode"] = self.weight_mode
        return payload


def clinical_3d_value(clinical_3d: dict[str, Any] | None, key: str) -> float:
    """Read a 3D endpoint metric from ``clinical_3d`` (patient-mean block preferred)."""
    c = clinical_3d or {}
    block = c.get("mean") if isinstance(c.get("mean"), dict) else c
    try:
        return float(block.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def evaluate_nested_groupkfold_oof(
    df: pd.DataFrame,
    *,
    trainer_factory: TrainerFactory,
    n_splits: int = 5,
    na_trend_store: NaTrendStore | None = None,
    na_trend_store_factory: NaTrendFactory | None = None,
    weight_mode: str = WEIGHT_INNER,
    targets: list[str] | None = None,
    group_seed: int | None = None,
) -> NestedOOFResult:
    """Evidentiary nested OOF. ``weight_mode='fixed_prior'`` is a benchmark only.

    ``group_seed`` remaps patient groups so repeated GroupKFold runs (seeds
    0/1/2) produce different outer folds for paired comparisons.
    """
    if weight_mode not in {WEIGHT_INNER, WEIGHT_FIXED_PRIOR}:
        raise ValueError(f"Unknown weight_mode={weight_mode!r}")
    protocol = PROTOCOL_NESTED if weight_mode == WEIGHT_INNER else PROTOCOL_FIXED_PRIOR
    target_cols = list(targets or TARGET_NAMES)
    frame = df.reset_index(drop=True)
    name_col = patient_group_column(frame)
    groups_raw = frame[name_col].astype(str).values
    groups = remap_groups_for_seed(groups_raw, group_seed)
    n_unique = len(np.unique(groups_raw))
    splits = min(int(n_splits), n_unique)
    if splits < 2:
        raise ValueError(f"Need at least 2 groups for nested GroupKFold, got {n_unique}")

    gkf = GroupKFold(n_splits=splits)
    oof = {t: np.full(len(frame), np.nan) for t in target_cols}
    folds: list[dict[str, Any]] = []
    store_factory = na_trend_store_factory or na_trend_factory_from_fitted_store(na_trend_store)

    for fold_i, (train_idx, val_idx) in enumerate(
        gkf.split(frame, frame[target_cols[0]], groups=groups)
    ):
        tr = frame.iloc[train_idx].reset_index(drop=True)
        te = frame.iloc[val_idx].reset_index(drop=True)
        fold_store = store_factory(tr, te)
        fold_trainer = trainer_factory(na_trend_store=fold_store)
        prepared = fold_trainer.prepare_training_data_split(tr, te)
        if prepared[0] is None:
            raise RuntimeError(f"Outer fold {fold_i}: prepare_training_data_split failed")
        X_tr, X_te, y_tr, y_te = prepared
        g_tr = tr[name_col].astype(str).values
        fold_trainer.evaluate_fold(
            X_tr,
            X_te,
            y_tr,
            y_te,
            groups=g_tr,
            weight_mode=weight_mode,
        )
        pred = _predict_scaled(fold_trainer, X_te)
        for t in target_cols:
            oof[t][val_idx] = pred[t].to_numpy()
        folds.append(
            {
                "fold": fold_i,
                "n_train": int(len(tr)),
                "n_val": int(len(te)),
                "val_groups": sorted(set(te[name_col].astype(str))),
                "group_seed": group_seed,
                "preprocessor": _preprocessor_snapshot(fold_trainer),
                "weight_mode": weight_mode,
                "inner_ensemble_weights": dict(getattr(fold_trainer, "_optimized_weights", {}) or {}),
                "inner_fold_weight_traces": dict(
                    getattr(fold_trainer, "_inner_fold_weight_traces", {}) or {}
                ),
                "inner_weight_variance": dict(
                    getattr(fold_trainer, "_inner_weight_variance", {}) or {}
                ),
                "model_kind": getattr(fold_trainer, "model_kind", "ensemble"),
                "yz_target_boost": bool(getattr(fold_trainer, "yz_target_boost", True)),
            }
        )

    pred_df = pd.DataFrame(oof)
    metrics = summarize_oof(frame[target_cols], pred_df, target_cols)
    metrics["protocol"] = protocol
    metrics["weight_mode"] = weight_mode
    metrics["n_splits"] = splits
    metrics["n_patients"] = int(len(frame))
    metrics["group_seed"] = group_seed
    clinical = compute_clinical_report(
        frame[target_cols],
        pred_df,
        frame=frame,
        target_columns=target_cols,
    )
    fold_masks = []
    for fold in folds:
        val = set(str(g) for g in fold.get("val_groups") or [])
        fold_masks.append(np.array([str(g) in val for g in groups_raw], dtype=bool))
    conformal = fit_conformal_from_oof(
        frame[target_cols],
        pred_df,
        targets=target_cols,
        coverage=DEFAULT_COVERAGE,
        fold_masks=fold_masks,
    )
    metrics["clinical_3d"] = clinical.get("clinical_3d")
    metrics["conformal_coverage"] = {
        t: (conformal.get("per_target") or {}).get(t, {}).get("fold_coverage_mean")
        for t in target_cols
    }
    return NestedOOFResult(
        protocol=protocol,
        weight_mode=weight_mode,
        n_splits=splits,
        n_patients=int(len(frame)),
        metrics=metrics,
        oof_predictions=pred_df,
        folds=folds,
        group_ids=[str(g) for g in groups_raw],
        conformal=conformal,
        clinical_metrics=clinical,
        subgroup_metrics=clinical.get("subgroup_metrics"),
        worst_cases=list(clinical.get("worst_cases") or []),
    )


def evaluate_repeated_nested_cv(
    df: pd.DataFrame,
    *,
    trainer_factory: TrainerFactory,
    seeds: Sequence[int] = DEFAULT_REPEAT_SEEDS,
    n_splits: int = 5,
    na_trend_store: NaTrendStore | None = None,
    na_trend_store_factory: NaTrendFactory | None = None,
    weight_mode: str = WEIGHT_INNER,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """Run nested GroupKFold for each seed and aggregate mean/std metrics."""
    seed_list = [int(s) for s in seeds]
    per_seed: list[dict[str, Any]] = []
    results: list[NestedOOFResult] = []
    for seed in seed_list:
        result = evaluate_nested_groupkfold_oof(
            df,
            trainer_factory=trainer_factory,
            n_splits=n_splits,
            na_trend_store=na_trend_store,
            na_trend_store_factory=na_trend_store_factory,
            weight_mode=weight_mode,
            targets=targets,
            group_seed=seed,
        )
        results.append(result)
        per_seed.append(
            {
                "group_seed": seed,
                "avg_mae_mm": result.metrics.get("avg_mae_mm"),
                "z_avg_mae_mm": result.metrics.get("z_avg_mae_mm"),
                "axis_mae_mm": dict(result.metrics.get("axis_mae_mm") or {}),
                "avg_r2": result.metrics.get("avg_r2"),
                "avg_mae_ci95": list(result.metrics.get("avg_mae_ci95") or []),
                "clinical_3d": (result.metrics.get("clinical_3d") or {}),
            }
        )

    def _mean_std(values: list[float]) -> dict[str, float]:
        arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    aggregated = {
        "avg_mae_mm": _mean_std([r["avg_mae_mm"] for r in per_seed]),
        "z_avg_mae_mm": _mean_std([r["z_avg_mae_mm"] for r in per_seed]),
        "avg_r2": _mean_std([r["avg_r2"] for r in per_seed]),
        "axis_mae_mm": {
            axis: _mean_std(
                [
                    float((r.get("axis_mae_mm") or {}).get(axis, float("nan")))
                    for r in per_seed
                ]
            )
            for axis in ("x", "y", "z")
        },
        "within_10mm_ratio": _mean_std(
            [clinical_3d_value(r.get("clinical_3d"), "within_10mm_ratio") for r in per_seed]
        ),
        "mean_3d_error_mm": _mean_std(
            [clinical_3d_value(r.get("clinical_3d"), "mean_mm") for r in per_seed]
        ),
    }
    return {
        "seeds": seed_list,
        "n_splits": n_splits,
        "weight_mode": weight_mode,
        "per_seed": per_seed,
        "aggregated": aggregated,
        "results": results,
    }


def paired_compare_repeated(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Pairwise delta of repeated-CV aggregates (candidate − reference)."""
    cand = candidate.get("aggregated") or {}
    ref = reference.get("aggregated") or {}

    def _delta(key: str) -> float:
        c = (cand.get(key) or {}).get("mean", float("nan"))
        r = (ref.get(key) or {}).get("mean", float("nan"))
        if not np.isfinite(c) or not np.isfinite(r):
            return float("nan")
        return float(c - r)

    return {
        "delta_avg_mae_mm": _delta("avg_mae_mm"),
        "delta_z_avg_mae_mm": _delta("z_avg_mae_mm"),
        "delta_within_10mm_ratio": _delta("within_10mm_ratio"),
    }


def evaluate_nested_proxy_vs_honest(
    clinical_df: pd.DataFrame,
    *,
    trainer_factory: TrainerFactory,
    proxy_train_builder: ProxyTrainBuilder,
    n_splits: int = 5,
    na_trend_store: NaTrendStore | None = None,
    na_trend_store_factory: NaTrendFactory | None = None,
    weight_mode: str = WEIGHT_INNER,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """Same outer folds: honest (clinical train) vs proxy (builder after teacher).

    ``proxy_train_builder(train_clinical, teacher_trainer)`` must not include
    outer-validation patients. This function never calls honest-only OOF and
    labels it as a proxy metric.
    """
    if weight_mode not in {WEIGHT_INNER, WEIGHT_FIXED_PRIOR}:
        raise ValueError(f"Unknown weight_mode={weight_mode!r}")
    target_cols = list(targets or TARGET_NAMES)
    frame = clinical_df.reset_index(drop=True)
    name_col = patient_group_column(frame)
    groups = frame[name_col].astype(str).values
    splits = min(int(n_splits), len(np.unique(groups)))
    if splits < 2:
        raise ValueError("Need at least 2 groups for nested proxy vs honest")

    gkf = GroupKFold(n_splits=splits)
    honest_oof = {t: np.full(len(frame), np.nan) for t in target_cols}
    proxy_oof = {t: np.full(len(frame), np.nan) for t in target_cols}
    folds: list[dict[str, Any]] = []
    store_factory = na_trend_store_factory or na_trend_factory_from_fitted_store(na_trend_store)

    for fold_i, (train_idx, val_idx) in enumerate(
        gkf.split(frame, frame[target_cols[0]], groups=groups)
    ):
        tr = frame.iloc[train_idx].reset_index(drop=True)
        te = frame.iloc[val_idx].reset_index(drop=True)
        fold_store = store_factory(tr, te)

        honest = trainer_factory(na_trend_store=fold_store)
        h_prep = honest.prepare_training_data_split(tr, te)
        if h_prep[0] is None:
            raise RuntimeError(f"Outer fold {fold_i}: honest prepare failed")
        Xh_tr, Xh_te, yh_tr, yh_te = h_prep
        g_tr = tr[name_col].astype(str).values
        honest.evaluate_fold(Xh_tr, Xh_te, yh_tr, yh_te, groups=g_tr, weight_mode=weight_mode)
        h_pred = _predict_scaled(honest, Xh_te)

        proxy_train = proxy_train_builder(tr, honest)
        if proxy_train is None or len(proxy_train) == 0:
            raise ValueError(f"Outer fold {fold_i}: proxy_train_builder returned no rows")
        _assert_no_val_in_proxy_train(proxy_train, te)

        proxy = trainer_factory(na_trend_store=_clone_na_trend_store(fold_store))
        p_prep = proxy.prepare_training_data_split(proxy_train, te)
        if p_prep[0] is None:
            raise RuntimeError(f"Outer fold {fold_i}: proxy prepare failed")
        Xp_tr, Xp_te, yp_tr, yp_te = p_prep
        try:
            g_proxy = proxy_train[patient_group_column(proxy_train)].astype(str).values
        except ValueError:
            g_proxy = np.arange(len(proxy_train)).astype(str)
        proxy.evaluate_fold(Xp_tr, Xp_te, yp_tr, yp_te, groups=g_proxy, weight_mode=weight_mode)
        p_pred = _predict_scaled(proxy, Xp_te)

        for t in target_cols:
            honest_oof[t][val_idx] = h_pred[t].to_numpy()
            proxy_oof[t][val_idx] = p_pred[t].to_numpy()
        folds.append(
            {
                "fold": fold_i,
                "n_train_clinical": int(len(tr)),
                "n_val": int(len(te)),
                "n_proxy_train": int(len(proxy_train)),
                "val_groups": sorted(set(te[name_col].astype(str))),
                "honest_preprocessor": _preprocessor_snapshot(honest),
                "proxy_preprocessor": _preprocessor_snapshot(proxy),
            }
        )

    honest_pred = pd.DataFrame(honest_oof)
    proxy_pred = pd.DataFrame(proxy_oof)
    honest_metrics = summarize_oof(frame[target_cols], honest_pred, target_cols)
    proxy_metrics = summarize_oof(frame[target_cols], proxy_pred, target_cols)
    delta = {
        "avg_mae_mm": proxy_metrics["avg_mae_mm"] - honest_metrics["avg_mae_mm"],
        "z_avg_mae_mm": proxy_metrics["z_avg_mae_mm"] - honest_metrics["z_avg_mae_mm"],
        "per_target_mae_mm": {
            t: proxy_metrics["per_target_mae_mm"][t] - honest_metrics["per_target_mae_mm"][t]
            for t in target_cols
        },
    }
    return {
        "protocol": PROTOCOL_NESTED_PROXY,
        "weight_mode": weight_mode,
        "n_splits": splits,
        "n_patients": int(len(frame)),
        "folds": folds,
        "clinical_honest": honest_metrics,
        "clinical_proxy": proxy_metrics,
        "delta_proxy_minus_honest": delta,
        "honest_oof_predictions": {
            t: [None if not np.isfinite(v) else float(v) for v in honest_pred[t].tolist()]
            for t in target_cols
        },
        "proxy_oof_predictions": {
            t: [None if not np.isfinite(v) else float(v) for v in proxy_pred[t].tolist()]
            for t in target_cols
        },
        "group_ids": [str(g) for g in groups],
    }
