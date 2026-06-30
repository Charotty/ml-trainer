"""Attach teacher-predicted displacement labels to unlabeled feature rows."""

from __future__ import annotations

from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe

DEFAULT_MAX_AXIS_MM = 80.0
DEFAULT_MAX_DELTA_NORM_MM = 120.0


def vybor_reference_bounds(reference_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    ref = normalize_dataframe(reference_df)
    medians: dict[str, float] = {}
    low: dict[str, float] = {}
    high: dict[str, float] = {}
    for col in TARGET_NAMES:
        if col not in ref.columns:
            continue
        s = pd.to_numeric(ref[col], errors="coerce").dropna()
        medians[col] = float(s.median())
        low[col] = float(s.quantile(0.05))
        high[col] = float(s.quantile(0.95))
    return medians, low, high


def _delta_norm_frame(df: pd.DataFrame) -> pd.Series:
    left = df[["kidney_left_delta_x", "kidney_left_delta_y", "kidney_left_delta_z"]].astype(float)
    right = df[["kidney_right_delta_x", "kidney_right_delta_y", "kidney_right_delta_z"]].astype(float)
    return np.sqrt((left ** 2).sum(axis=1)) + np.sqrt((right ** 2).sum(axis=1))


def _sanitize_predictions(
    pred: pd.DataFrame,
    *,
    medians: Mapping[str, float],
    low: Mapping[str, float],
    high: Mapping[str, float],
    max_axis_mm: float,
) -> pd.DataFrame:
    out = pred.copy()
    for col in TARGET_NAMES:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce").astype(float)
        bad = values.abs() > max_axis_mm
        if bad.any():
            values = values.where(~bad, medians.get(col, 0.0))
        if col in low and col in high:
            values = values.clip(lower=low[col], upper=high[col])
        out[col] = values
    return out


def attach_pseudo_displacement_labels(
    df: pd.DataFrame,
    teacher_bundle,
    *,
    predict_fn: Callable,
    reference_df: pd.DataFrame,
    max_axis_mm: float = DEFAULT_MAX_AXIS_MM,
    max_delta_norm_mm: float = DEFAULT_MAX_DELTA_NORM_MM,
    label_quality: str = "pseudo_dicom",
) -> pd.DataFrame:
    """Predict δ with a Vybor teacher; clip/fallback to Vybor clinical priors."""
    out = normalize_dataframe(df.copy())
    if len(out) == 0:
        return out

    medians, low, high = vybor_reference_bounds(reference_df)
    raw_pred = predict_fn(teacher_bundle, out)
    pred = _sanitize_predictions(
        raw_pred,
        medians=medians,
        low=low,
        high=high,
        max_axis_mm=max_axis_mm,
    )

    for col in TARGET_NAMES:
        out[col] = pred[col].values

    # Final fallback: population median vector for any still-insane rows.
    norms = _delta_norm_frame(out)
    insane = norms > max_delta_norm_mm
    if insane.any():
        for col in TARGET_NAMES:
            out.loc[insane, col] = medians.get(col, np.nan)
        out.loc[insane, "label_quality"] = "population_prior_dicom"
    out.loc[~insane, "label_quality"] = label_quality
    out["pseudo_delta_norm_mm"] = _delta_norm_frame(out)
    return out
