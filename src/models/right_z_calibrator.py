"""Post-hoc calibration for kidney_right_delta_z (legacy pickle-compatible wrapper)."""

from __future__ import annotations

from typing import Any

from src.models.span_anchor_calibrator import SpanAnchorCalibrator, SpanAnchorParams

TARGET = "kidney_right_delta_z"
PACKAGE_STATUS = "legacy"


class RightZCalibrator(SpanAnchorCalibrator):
    """Pickle-stable right-side wrapper around ``SpanAnchorCalibrator``."""

    def __init__(self, *, clip_mm: tuple[float, float] = (-55.0, 25.0)):
        super().__init__(side="right", clip_mm=clip_mm)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if not getattr(self, "side", None):
            self.side = "right"


__all__ = ["RightZCalibrator", "SpanAnchorParams", "TARGET"]
