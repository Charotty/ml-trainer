"""OOF-gated span/lordosis calibration for kidney Z (no lateral leakage)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeVar

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.features.phase1_schema import normalize_dataframe

T = TypeVar("T")


@dataclass(frozen=True)
class SpanLordosisParams:
    span_threshold_mm: float = 80.0
    k_span: float = 0.15
    k_lordosis: float = 0.08
    k_z_rel: float = 0.35
    blend: float = 0.20
    clip_mm: tuple[float, float] = (-55.0, 25.0)


def _clinical_z_arrays(
    df: pd.DataFrame,
    *,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = normalize_dataframe(df.copy())
    span = frame.get(
        f"kidney_{side}_z_span_supine_mm",
        pd.Series(np.nan, index=frame.index),
    )
    lordosis = frame.get(
        "lumbar_lordosis_deg",
        pd.Series(np.nan, index=frame.index),
    )
    z_rel = frame.get(
        f"kidney_{side}_center_z_rel",
        pd.Series(np.nan, index=frame.index),
    )
    return (
        span.astype(float).fillna(0.0).values,
        lordosis.astype(float).fillna(0.0).values,
        z_rel.astype(float).fillna(0.0).values,
    )


def _apply_span_lordosis(
    raw_pred: np.ndarray,
    span: np.ndarray,
    lordosis: np.ndarray,
    z_rel: np.ndarray,
    params: SpanLordosisParams,
) -> np.ndarray:
    raw = np.asarray(raw_pred, dtype=float).reshape(-1)
    anchor = (
        span * params.k_span
        - lordosis * params.k_lordosis
        - z_rel * params.k_z_rel
    )
    need = span >= params.span_threshold_mm
    out = raw.copy()
    out[need] = (1.0 - params.blend) * raw[need] + params.blend * anchor[need]
    return np.clip(out, params.clip_mm[0], params.clip_mm[1])


def _grid_search(
    raw_pred: np.ndarray,
    span: np.ndarray,
    lordosis: np.ndarray,
    z_rel: np.ndarray,
    y_true: np.ndarray,
    clip_mm: tuple[float, float],
) -> SpanLordosisParams:
    best_mae = float("inf")
    best = SpanLordosisParams(clip_mm=clip_mm)
    for thr, k_sp, k_lor, k_zr, blend in itertools.product(
        (60.0, 80.0, 100.0),
        (0.10, 0.15, 0.20),
        (0.05, 0.08, 0.12),
        (0.25, 0.35, 0.45),
        (0.15, 0.20, 0.30),
    ):
        params = SpanLordosisParams(
            span_threshold_mm=thr,
            k_span=k_sp,
            k_lordosis=k_lor,
            k_z_rel=k_zr,
            blend=blend,
            clip_mm=clip_mm,
        )
        pred = _apply_span_lordosis(raw_pred, span, lordosis, z_rel, params)
        mae = mean_absolute_error(y_true, pred)
        if mae < best_mae:
            best_mae = mae
            best = params
    return best


class SideZCalibrator:
    """Supine-only Z calibration (span + lordosis anchor, no lateral features)."""

    def __init__(self, *, side: str = "left", clip_mm: tuple[float, float] = (-55.0, 25.0)):
        self.side = side
        self.clip_mm = clip_mm
        self.params = SpanLordosisParams(clip_mm=clip_mm)
        self.fitted_ = False
        self.train_mae_before_: float | None = None
        self.train_mae_after_: float | None = None
        self.oof_mae_before_: float | None = None
        self.oof_mae_after_: float | None = None

    @property
    def target(self) -> str:
        return f"kidney_{self.side}_delta_z"

    def fit(
        self,
        df: pd.DataFrame,
        raw_pred: Sequence[float],
        y_true: Sequence[float],
    ) -> "SideZCalibrator":
        raw = np.asarray(raw_pred, dtype=float).reshape(-1)
        y = np.asarray(y_true, dtype=float).reshape(-1)
        span, lordosis, z_rel = _clinical_z_arrays(df, side=self.side)
        self.train_mae_before_ = float(mean_absolute_error(y, raw))
        self.params = _grid_search(raw, span, lordosis, z_rel, y, self.clip_mm)
        calibrated = _apply_span_lordosis(raw, span, lordosis, z_rel, self.params)
        self.train_mae_after_ = float(mean_absolute_error(y, calibrated))
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, raw_pred: Sequence[float]) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("SideZCalibrator is not fitted")
        raw = np.asarray(raw_pred, dtype=float).reshape(-1)
        span, lordosis, z_rel = _clinical_z_arrays(df, side=self.side)
        return _apply_span_lordosis(raw, span, lordosis, z_rel, self.params)

    def apply_scalar(self, patient_data: Mapping[str, Any], raw_pred: float) -> float:
        df = pd.DataFrame([dict(patient_data)])
        return float(self.transform(df, [raw_pred])[0])

    def describe(self) -> dict[str, Any]:
        return {
            "method": "span_lordosis_blend_supine_only",
            "side": self.side,
            "params": self.params.__dict__,
            "train_mae_before_mm": self.train_mae_before_,
            "train_mae_after_mm": self.train_mae_after_,
            "oof_mae_before_mm": self.oof_mae_before_,
            "oof_mae_after_mm": self.oof_mae_after_,
        }


def fit_calibrator_oof_gated(
    calibrator: SideZCalibrator,
    df: pd.DataFrame,
    raw_pred: Sequence[float],
    y_true: Sequence[float],
    groups: Sequence[str],
    *,
    n_splits: int = 5,
    min_improvement_mm: float = 0.0,
) -> SideZCalibrator | None:
    """Fit calibrator only when GroupKFold OOF MAE improves."""
    raw = np.asarray(raw_pred, dtype=float).reshape(-1)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    groups_arr = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups_arr)))
    if n_splits < 2:
        calibrator.fit(df, raw, y)
        if (calibrator.train_mae_after_ or float("inf")) <= (
            calibrator.train_mae_before_ or float("inf")
        ) - min_improvement_mm:
            return calibrator
        return None

    splitter = GroupKFold(n_splits=n_splits)
    oof_raw = np.full(len(y), np.nan)
    oof_cal = np.full(len(y), np.nan)
    for train_idx, val_idx in splitter.split(df, y, groups=groups_arr):
        fold_cal = SideZCalibrator(side=calibrator.side, clip_mm=calibrator.clip_mm)
        fold_cal.fit(df.iloc[train_idx], raw[train_idx], y[train_idx])
        oof_raw[val_idx] = raw[val_idx]
        oof_cal[val_idx] = fold_cal.transform(df.iloc[val_idx], raw[val_idx])

    valid = np.isfinite(oof_raw) & np.isfinite(oof_cal)
    if not np.any(valid):
        return None
    mae_before = float(mean_absolute_error(y[valid], oof_raw[valid]))
    mae_after = float(mean_absolute_error(y[valid], oof_cal[valid]))
    calibrator.oof_mae_before_ = mae_before
    calibrator.oof_mae_after_ = mae_after
    if mae_after > mae_before - min_improvement_mm:
        return None
    calibrator.fit(df, raw, y)
    return calibrator
