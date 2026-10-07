"""Post-hoc calibration for kidney_left_delta_z (legacy pickle-compatible wrapper)."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.models.span_anchor_calibrator import SpanAnchorCalibrator, SpanAnchorParams

TARGET = "kidney_left_delta_z"
PACKAGE_STATUS = "legacy"


class LeftZCalibrator(SpanAnchorCalibrator):
    """Pickle-stable left-side wrapper around ``SpanAnchorCalibrator``."""

    def __init__(self, *, clip_mm: tuple[float, float] = (-55.0, 25.0)):
        super().__init__(side="left", clip_mm=clip_mm)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if not getattr(self, "side", None):
            self.side = "left"


def apply_left_z_calibration(
    model_data: Mapping[str, Any],
    df: pd.DataFrame,
    raw_predictions: pd.DataFrame,
) -> pd.DataFrame:
    calibrator = model_data.get("left_z_calibrator")
    if calibrator is None:
        return raw_predictions
    out = raw_predictions.copy()
    if TARGET not in out.columns:
        return out
    out[TARGET] = calibrator.transform(df, out[TARGET].values)
    return out


__all__ = ["LeftZCalibrator", "SpanAnchorParams", "TARGET", "apply_left_z_calibration"]
