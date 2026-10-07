"""Clinical displacement metrics: 3D endpoint error, tails, subgroups.

Primary metric is Euclidean 3D endpoint error per kidney (left/right), not
axis-wise MAE and not |‖Δ_true‖ − ‖Δ_pred‖|. Secondary metrics are per-axis
MAE / RMSE / R². Tail summaries and within-threshold fractions are always
computed from the 3D errors.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.phase1_schema import TARGET_NAMES

WITHIN_THRESHOLDS_MM: tuple[float, ...] = (5.0, 10.0, 15.0)
TAIL_PERCENTILES: tuple[float, ...] = (50.0, 90.0, 95.0)
LEFT_XYZ: tuple[str, ...] = (
    "kidney_left_delta_x",
    "kidney_left_delta_y",
    "kidney_left_delta_z",
)
RIGHT_XYZ: tuple[str, ...] = (
    "kidney_right_delta_x",
    "kidney_right_delta_y",
    "kidney_right_delta_z",
)
DEMOGRAPHIC_SUBGROUP_COLS: tuple[str, ...] = (
    "sex",
    "age",
    "bmi",
    "body_type",
    "has_previous_surgery",
)
MISSINGNESS_COLS: tuple[str, ...] = ("sex", "age", "bmi", "body_type", "has_previous_surgery")
SMALL_N_WARN: int = 8
SEED = 42
N_BOOTSTRAP = 2000


def _as_target_frame(data: Any, cols: Sequence[str]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.loc[:, [c for c in cols if c in data.columns]].copy()
    if isinstance(data, Mapping):
        return pd.DataFrame({c: data[c] for c in cols if c in data})
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(cols):
        raise ValueError(f"Expected array shape (n, {len(cols)}), got {arr.shape}")
    return pd.DataFrame(arr, columns=list(cols))


def has_full_xyz(cols: Sequence[str]) -> bool:
    present = set(cols)
    return all(c in present for c in LEFT_XYZ) and all(c in present for c in RIGHT_XYZ)


def endpoint_errors_3d(
    y_true: Any,
    y_pred: Any,
    target_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-row Euclidean endpoint error (mm) for left, right, and mean."""
    cols = list(target_columns or TARGET_NAMES)
    true_df = _as_target_frame(y_true, cols)
    pred_df = _as_target_frame(y_pred, cols)
    n = len(true_df)
    out = pd.DataFrame(index=true_df.index)
    if all(c in true_df.columns and c in pred_df.columns for c in LEFT_XYZ):
        diff = pred_df[list(LEFT_XYZ)].to_numpy(dtype=float) - true_df[list(LEFT_XYZ)].to_numpy(
            dtype=float
        )
        out["endpoint_error_left_mm"] = np.sqrt(np.sum(diff * diff, axis=1))
    else:
        out["endpoint_error_left_mm"] = np.full(n, np.nan)
    if all(c in true_df.columns and c in pred_df.columns for c in RIGHT_XYZ):
        diff = pred_df[list(RIGHT_XYZ)].to_numpy(dtype=float) - true_df[list(RIGHT_XYZ)].to_numpy(
            dtype=float
        )
        out["endpoint_error_right_mm"] = np.sqrt(np.sum(diff * diff, axis=1))
    else:
        out["endpoint_error_right_mm"] = np.full(n, np.nan)
    left = out["endpoint_error_left_mm"].to_numpy(dtype=float)
    right = out["endpoint_error_right_mm"].to_numpy(dtype=float)
    if not (np.isfinite(left).any() or np.isfinite(right).any()):
        out["endpoint_error_mean_mm"] = np.full(n, np.nan)
        out["endpoint_error_max_mm"] = np.full(n, np.nan)
        return out
    stacked = np.vstack([left, right])
    with np.errstate(all="ignore"):
        out["endpoint_error_mean_mm"] = np.nanmean(stacked, axis=0)
        out["endpoint_error_max_mm"] = np.nanmax(stacked, axis=0)
    return out


def _tail_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "mean_mm": float("nan"),
            "median_mm": float("nan"),
            "p90_mm": float("nan"),
            "p95_mm": float("nan"),
            "max_mm": float("nan"),
            "n": 0,
        }
    return {
        "mean_mm": float(np.mean(finite)),
        "median_mm": float(np.median(finite)),
        "p90_mm": float(np.percentile(finite, 90)),
        "p95_mm": float(np.percentile(finite, 95)),
        "max_mm": float(np.max(finite)),
        "n": int(finite.size),
    }


def _within_fractions(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {f"within_{int(t)}mm_ratio": float("nan") for t in WITHIN_THRESHOLDS_MM}
    return {
        f"within_{int(t)}mm_ratio": float(np.mean(finite <= t)) for t in WITHIN_THRESHOLDS_MM
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    n = len(finite)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        v = float(finite[0])
        return v, v
    means = [finite[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def axis_regression_metrics(
    y_true: Any,
    y_pred: Any,
    target_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    cols = list(target_columns or TARGET_NAMES)
    true_df = _as_target_frame(y_true, cols)
    pred_df = _as_target_frame(y_pred, cols)
    per_mae: dict[str, float] = {}
    per_rmse: dict[str, float] = {}
    per_r2: dict[str, float] = {}
    for col in cols:
        if col not in true_df.columns or col not in pred_df.columns:
            continue
        yt = true_df[col].to_numpy(dtype=float)
        yp = pred_df[col].to_numpy(dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() < 1:
            per_mae[col] = float("nan")
            per_rmse[col] = float("nan")
            per_r2[col] = float("nan")
            continue
        per_mae[col] = float(mean_absolute_error(yt[mask], yp[mask]))
        per_rmse[col] = float(np.sqrt(mean_squared_error(yt[mask], yp[mask])))
        if mask.sum() >= 2:
            per_r2[col] = float(r2_score(yt[mask], yp[mask]))
        else:
            per_r2[col] = float("nan")
    axes: dict[str, dict[str, list[float]]] = {
        "x": {"mae": [], "rmse": [], "r2": []},
        "y": {"mae": [], "rmse": [], "r2": []},
        "z": {"mae": [], "rmse": [], "r2": []},
    }
    for col, mae in per_mae.items():
        axis = str(col).split("_")[-1]
        if axis in axes:
            axes[axis]["mae"].append(mae)
            axes[axis]["rmse"].append(per_rmse[col])
            axes[axis]["r2"].append(per_r2[col])
    axis_mae = {k: float(np.mean(v["mae"])) for k, v in axes.items() if v["mae"]}
    axis_rmse = {k: float(np.mean(v["rmse"])) for k, v in axes.items() if v["rmse"]}
    axis_r2 = {
        k: float(np.nanmean(v["r2"])) for k, v in axes.items() if v["r2"]
    }
    return {
        "per_target_mae_mm": per_mae,
        "per_target_rmse_mm": per_rmse,
        "per_target_r2": per_r2,
        "axis_mae_mm": axis_mae,
        "axis_rmse_mm": axis_rmse,
        "axis_r2": axis_r2,
        "avg_mae_mm": float(np.mean(list(per_mae.values()))) if per_mae else float("nan"),
        "avg_rmse_mm": float(np.mean(list(per_rmse.values()))) if per_rmse else float("nan"),
        "avg_r2": float(np.nanmean(list(per_r2.values()))) if per_r2 else float("nan"),
    }


def summarize_3d_and_tails(errors: pd.DataFrame) -> dict[str, Any]:
    """Primary 3D + tail + within-threshold block."""
    payload: dict[str, Any] = {}
    for key, label in (
        ("endpoint_error_left_mm", "left"),
        ("endpoint_error_right_mm", "right"),
        ("endpoint_error_mean_mm", "mean"),
    ):
        if key not in errors.columns:
            continue
        vals = errors[key].to_numpy(dtype=float)
        block = _tail_stats(vals)
        block.update(_within_fractions(vals))
        lo, hi = bootstrap_mean_ci(vals)
        block["mean_ci95"] = [lo, hi]
        payload[label] = block
    payload["primary"] = "endpoint_error_3d_mm"
    payload["thresholds_mm"] = list(WITHIN_THRESHOLDS_MM)
    return payload


def missingness_flags(frame: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=frame.index)
    for col in MISSINGNESS_COLS:
        if col in frame.columns:
            flags[f"{col}_missing"] = pd.to_numeric(frame[col], errors="coerce").isna().astype(int)
        else:
            flags[f"{col}_missing"] = 0
    flag_cols = [c for c in flags.columns if c.endswith("_missing")]
    flags["any_demographics_missing"] = flags[flag_cols].max(axis=1) if flag_cols else 0
    return flags


def ood_flags(frame: pd.DataFrame) -> pd.Series:
    if "geometry_ood_x" not in frame.columns:
        return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index, name="geometry_ood_x")
    raw = frame["geometry_ood_x"]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int).rename("geometry_ood_x")
    return raw.astype(bool).astype(int).rename("geometry_ood_x")


def worst_case_rows(
    frame: pd.DataFrame,
    errors: pd.DataFrame,
    *,
    top_n: int = 10,
    extra_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Highest 3D errors with missingness / OOD flags (no feature dump of targets)."""
    joined = pd.concat(
        [frame.reset_index(drop=True), errors.reset_index(drop=True), missingness_flags(frame).reset_index(drop=True)],
        axis=1,
    )
    joined["geometry_ood_x"] = ood_flags(frame).to_numpy()
    sort_col = "endpoint_error_mean_mm"
    if sort_col not in joined.columns:
        return joined.head(0)
    keep = [
        c
        for c in (
            "full_name",
            "case_id",
            "endpoint_error_left_mm",
            "endpoint_error_right_mm",
            "endpoint_error_mean_mm",
            "endpoint_error_max_mm",
            "sex_missing",
            "age_missing",
            "bmi_missing",
            "body_type_missing",
            "has_previous_surgery_missing",
            "any_demographics_missing",
            "geometry_ood_x",
        )
        if c in joined.columns
    ]
    for col in extra_cols or ():
        if col in joined.columns and col not in keep:
            keep.append(col)
    ranked = joined.sort_values(sort_col, ascending=False, na_position="last")
    return ranked[keep].head(int(top_n)).reset_index(drop=True)


def _bin_age(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 40:
        return "<40"
    if value < 60:
        return "40-59"
    return ">=60"


def _bin_bmi(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 25:
        return "<25"
    if value < 30:
        return "25-29.9"
    return ">=30"


def _bin_binary(value: float, *, positive: str, negative: str) -> str:
    if not np.isfinite(value):
        return "missing"
    return positive if value >= 0.5 else negative


def _bin_sex(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if abs(value - 1.0) < 1e-9:
        return "male"
    if abs(value - 2.0) < 1e-9:
        return "female"
    return f"code_{value:g}"


def _bin_body_type(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    labels = {0.0: "normosthenic", 1.0: "asthenic", 2.0: "hypersthenic"}
    for code, name in labels.items():
        if abs(value - code) < 1e-9:
            return name
    return f"code_{value:g}"


def subgroup_error_table(
    frame: pd.DataFrame,
    errors: pd.DataFrame,
    *,
    error_col: str = "endpoint_error_mean_mm",
) -> dict[str, Any]:
    """OOF errors + CI by demographic / missingness slices.

    Small groups publish ``n`` and a wide CI. The payload never claims
    'no difference'; underpowered slices are flagged instead.
    """
    if error_col not in errors.columns:
        return {"note": f"missing {error_col}", "groups": {}}
    err = errors[error_col].to_numpy(dtype=float)
    assignments: dict[str, np.ndarray] = {}
    if "sex" in frame.columns:
        assignments["sex"] = pd.to_numeric(frame["sex"], errors="coerce").map(_bin_sex).to_numpy()
    if "age" in frame.columns:
        assignments["age"] = pd.to_numeric(frame["age"], errors="coerce").map(_bin_age).to_numpy()
    if "bmi" in frame.columns:
        assignments["bmi"] = pd.to_numeric(frame["bmi"], errors="coerce").map(_bin_bmi).to_numpy()
    if "body_type" in frame.columns:
        assignments["body_type"] = (
            pd.to_numeric(frame["body_type"], errors="coerce").map(_bin_body_type).to_numpy()
        )
    if "has_previous_surgery" in frame.columns:
        assignments["prior_surgery"] = (
            pd.to_numeric(frame["has_previous_surgery"], errors="coerce")
            .map(lambda v: _bin_binary(v, positive="yes", negative="no"))
            .to_numpy()
        )
    miss = missingness_flags(frame)
    assignments["demographics_missing"] = np.where(
        miss["any_demographics_missing"].to_numpy() > 0, "missing", "observed"
    )

    groups: dict[str, list[dict[str, Any]]] = {}
    for factor, labels in assignments.items():
        rows: list[dict[str, Any]] = []
        for label in sorted(set(str(x) for x in labels)):
            mask = np.asarray(labels) == label
            vals = err[mask]
            n = int(np.isfinite(vals).sum())
            lo, hi = bootstrap_mean_ci(vals)
            underpowered = n < SMALL_N_WARN
            rows.append(
                {
                    "level": label,
                    "n": n,
                    "mean_error_mm": float(np.nanmean(vals)) if n else float("nan"),
                    "median_error_mm": float(np.nanmedian(vals)) if n else float("nan"),
                    "mean_ci95": [lo, hi],
                    "underpowered": underpowered,
                    "inference_note": (
                        "n is small; CI is wide; do not claim 'no difference'."
                        if underpowered
                        else "descriptive only; not a causal contrast."
                    ),
                }
            )
        groups[factor] = rows
    return {
        "error_col": error_col,
        "small_n_threshold": SMALL_N_WARN,
        "no_difference_claims": False,
        "groups": groups,
    }


def compute_clinical_report(
    y_true: Any,
    y_pred: Any,
    *,
    frame: pd.DataFrame | None = None,
    target_columns: Sequence[str] | None = None,
    top_n_worst: int = 10,
) -> dict[str, Any]:
    """Full clinical metric block for nested OOF / offline evaluation."""
    cols = list(target_columns or TARGET_NAMES)
    axis = axis_regression_metrics(y_true, y_pred, cols)
    errors = endpoint_errors_3d(y_true, y_pred, cols)
    clinical = summarize_3d_and_tails(errors)
    payload: dict[str, Any] = {
        **axis,
        "clinical_3d": clinical,
        "n_rows": int(len(errors)),
        "has_full_xyz": has_full_xyz(cols),
    }
    if frame is not None:
        payload["worst_cases"] = worst_case_rows(frame, errors, top_n=top_n_worst).to_dict(
            orient="records"
        )
        payload["subgroup_metrics"] = subgroup_error_table(frame, errors)
        payload["missingness_rate"] = {
            col: float(missingness_flags(frame)[col].mean())
            for col in missingness_flags(frame).columns
        }
    else:
        payload["worst_cases"] = []
        payload["subgroup_metrics"] = {"groups": {}, "no_difference_claims": False}
    return payload


def compute_clinical_within_ratios(
    y_true: Any,
    y_pred: Any,
    target_columns: Sequence[str] | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Backward-compatible wrapper used by ``evaluate_metrics.py``.

    ``within_*mm_ratio`` is the fraction of patients whose *mean* left/right
    3D endpoint error is within the threshold. Pointwise axis rates stay
    under distinct names.
    """
    cols = list(target_columns or TARGET_NAMES)
    true_df = _as_target_frame(y_true, cols)
    pred_df = _as_target_frame(y_pred, cols)
    errors = endpoint_errors_3d(true_df, pred_df, cols)
    left = errors["endpoint_error_left_mm"].to_numpy(dtype=float)
    right = errors["endpoint_error_right_mm"].to_numpy(dtype=float)
    mean_err = errors["endpoint_error_mean_mm"].to_numpy(dtype=float)
    present = [c for c in cols if c in true_df.columns and c in pred_df.columns]
    pointwise_abs = np.abs(true_df[present].to_numpy() - pred_df[present].to_numpy())
    summary = {
        "vector_error_left_mae_mm": float(np.nanmean(left)),
        "vector_error_right_mae_mm": float(np.nanmean(right)),
        "vector_error_mean_mae_mm": float(np.nanmean(mean_err)),
        "endpoint_error_left_mae_mm": float(np.nanmean(left)),
        "endpoint_error_right_mae_mm": float(np.nanmean(right)),
        "endpoint_error_mean_mae_mm": float(np.nanmean(mean_err)),
        "within_5mm_ratio": float(np.nanmean(mean_err <= 5.0)),
        "within_10mm_ratio": float(np.nanmean(mean_err <= 10.0)),
        "within_15mm_ratio": float(np.nanmean(mean_err <= 15.0)),
        "within_5mm_pointwise_ratio": float((pointwise_abs <= 5.0).mean()) if pointwise_abs.size else float("nan"),
        "within_10mm_pointwise_ratio": float((pointwise_abs <= 10.0).mean()) if pointwise_abs.size else float("nan"),
        "sample_count": float(len(true_df)),
        "median_endpoint_error_mm": float(np.nanmedian(mean_err)),
        "p90_endpoint_error_mm": float(np.nanpercentile(mean_err, 90)) if len(mean_err) else float("nan"),
        "p95_endpoint_error_mm": float(np.nanpercentile(mean_err, 95)) if len(mean_err) else float("nan"),
        "max_endpoint_error_mm": float(np.nanmax(mean_err)) if len(mean_err) else float("nan"),
    }
    per_patient = {
        "vector_error_left_mm": left,
        "vector_error_right_mm": right,
        "vector_error_mean_mm": mean_err,
        "endpoint_error_left_mm": left,
        "endpoint_error_right_mm": right,
        "endpoint_error_mean_mm": mean_err,
    }
    return summary, per_patient
