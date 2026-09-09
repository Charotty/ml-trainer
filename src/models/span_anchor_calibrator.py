"""Side-aware span-anchor Z calibrator (legacy algorithm).

Production training uses ``src.models.z_calibrator_oof.SideZCalibrator``.
This module unifies the older left/right span-delta calibrators.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.features.displacement_axis_features import add_displacement_axis_features
from src.features.phase1_schema import normalize_dataframe

PACKAGE_STATUS = "legacy"


@dataclass(frozen=True)
class SpanAnchorParams:
    ds_abs_threshold_mm: float = 5.0
    k_delta_span: float = 0.35
    k_z_rel: float = 0.45
    blend: float = 0.25
    clip_mm: tuple[float, float] = (-55.0, 25.0)


class SpanAnchorCalibrator:
    """Blend raw Z with a span/lordosis-style anchor. Parameterized by ``side``."""

    def __init__(self, *, side: str = "left", clip_mm: tuple[float, float] = (-55.0, 25.0)):
        if side not in {"left", "right"}:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side
        self.clip_mm = clip_mm
        self.params = SpanAnchorParams(clip_mm=clip_mm)
        self.fitted_ = False
        self.train_mae_before_: float | None = None
        self.train_mae_after_: float | None = None

    @property
    def target(self) -> str:
        return f"kidney_{getattr(self, 'side', 'left')}_delta_z"

    def _clinical_arrays(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        side = getattr(self, "side", "left")
        frame = add_displacement_axis_features(normalize_dataframe(df.copy()))
        delta_span = frame.get(
            f"kidney_{side}_z_delta_span_mm",
            pd.Series(np.nan, index=frame.index),
        )
        z_rel = frame.get(
            f"kidney_{side}_center_z_rel",
            pd.Series(np.nan, index=frame.index),
        )
        return (
            delta_span.astype(float).fillna(0.0).values,
            z_rel.astype(float).fillna(0.0).values,
        )

    def _apply(
        self,
        raw_pred: np.ndarray,
        delta_span: np.ndarray,
        z_rel: np.ndarray,
        params: SpanAnchorParams,
    ) -> np.ndarray:
        raw = np.asarray(raw_pred, dtype=float).reshape(-1)
        anchor = delta_span * params.k_delta_span - z_rel * params.k_z_rel
        need = np.abs(delta_span) >= params.ds_abs_threshold_mm
        out = raw.copy()
        out[need] = (1.0 - params.blend) * raw[need] + params.blend * anchor[need]
        return np.clip(out, params.clip_mm[0], params.clip_mm[1])

    def _grid_search(
        self,
        raw_pred: np.ndarray,
        delta_span: np.ndarray,
        z_rel: np.ndarray,
        y_true: np.ndarray,
    ) -> SpanAnchorParams:
        best_mae = float("inf")
        best = SpanAnchorParams(clip_mm=self.clip_mm)
        for ds_thr, k_ds, k_zr, blend in itertools.product(
            (3.0, 4.0, 5.0, 6.0),
            (0.30, 0.35, 0.45, 0.55),
            (0.20, 0.30, 0.45),
            (0.20, 0.25, 0.35, 0.45),
        ):
            params = SpanAnchorParams(
                ds_abs_threshold_mm=ds_thr,
                k_delta_span=k_ds,
                k_z_rel=k_zr,
                blend=blend,
                clip_mm=self.clip_mm,
            )
            pred = self._apply(raw_pred, delta_span, z_rel, params)
            mae = mean_absolute_error(y_true, pred)
            if mae < best_mae:
                best_mae = mae
                best = params
        return best

    def fit(
        self,
        df: pd.DataFrame,
        raw_pred: Sequence[float],
        y_true: Sequence[float],
    ) -> "SpanAnchorCalibrator":
        raw = np.asarray(raw_pred, dtype=float).reshape(-1)
        y = np.asarray(y_true, dtype=float).reshape(-1)
        delta_span, z_rel = self._clinical_arrays(df)
        self.train_mae_before_ = float(mean_absolute_error(y, raw))
        self.params = self._grid_search(raw, delta_span, z_rel, y)
        calibrated = self._apply(raw, delta_span, z_rel, self.params)
        self.train_mae_after_ = float(mean_absolute_error(y, calibrated))
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, raw_pred: Sequence[float]) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        raw = np.asarray(raw_pred, dtype=float).reshape(-1)
        delta_span, z_rel = self._clinical_arrays(df)
        return self._apply(raw, delta_span, z_rel, self.params)

    def apply_scalar(self, patient_data: Mapping[str, Any], raw_pred: float) -> float:
        df = pd.DataFrame([dict(patient_data)])
        return float(self.transform(df, [raw_pred])[0])

    def describe(self) -> dict[str, Any]:
        return {
            "method": "span_anchor_blend",
            "side": getattr(self, "side", None),
            "params": self.params.__dict__,
            "train_mae_before_mm": self.train_mae_before_,
            "train_mae_after_mm": self.train_mae_after_,
        }
