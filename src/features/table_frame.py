"""Map CT vertebral-frame kidney points onto the Excel sheet frame.

Training rows store the supine middle point the way the sheet does: unsigned
X from the midline, and Y/Z as signed offsets from the same vertebral reading.
``*_vert`` is that reading from TotalSegmentator, but the paired cohort sits
at a different origin (see ``docs/CLINICAL_TRAIN_XLSX14_20261013.md`` §3):

* X is about +11 mm too large
* Y is about −39 mm (near 0 instead of about +41)
* Z is about +63 mm (near 0 instead of about −62)

Those three constants are the mean CT-minus-Excel gap on the lateral middle
point (left and right averaged). They are not fit on the holdout. Y is
negated first when it is clearly the opposite sign of the sheet (the sheet
median is positive; a large negative vertebral Y is mirrored).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features.ct_geometry import (
    FEATURE_FRAME_CLINICAL_SIGNED_X,
    flag_geometry_ood_x,
)

# Mean (CT vertebral middle − Excel middle), mm. Lateral paired cohort.
VERT_TO_TABLE_BIAS_MM = {"x": 11.25, "y": -38.75, "z": 62.6}
SHEET_Y_SIGN = 1.0
Y_SIGN_FLIP_MIN_MM = 15.0

# Sheet body size. The fast skin mask is ~100 mm wide and is not this quantity.
TABLE_WIDTH_RANGE_MM = (200.0, 500.0)
TABLE_DEPTH_RANGE_MM = (140.0, 400.0)

GEOMETRY_RANGE_KEYS: Tuple[str, ...] = (
    "kidney_left_supine_middle_z",
    "kidney_right_supine_middle_z",
    "kidney_left_to_spine_distance",
    "kidney_right_to_spine_distance",
)

Z_TARGET_NAMES: Tuple[str, ...] = (
    "kidney_left_delta_z",
    "kidney_right_delta_z",
)


def _finite(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def correct_vertebral_point(
    x_vert: float,
    y_vert: float,
    z_vert: float,
) -> Tuple[float, float, float, bool]:
    """Return (x, y, z, y_flipped) in the sheet frame."""
    x_out = abs(float(x_vert)) - VERT_TO_TABLE_BIAS_MM["x"]
    y_val = float(y_vert)
    y_flipped = False
    if y_val * SHEET_Y_SIGN < 0.0 and abs(y_val) >= Y_SIGN_FLIP_MIN_MM:
        y_val = -y_val
        y_flipped = True
    y_out = y_val - VERT_TO_TABLE_BIAS_MM["y"]
    z_out = float(z_vert) - VERT_TO_TABLE_BIAS_MM["z"]
    return x_out, y_out, z_out, y_flipped


def _clear_non_table_body_size(out: Dict[str, object]) -> None:
    width = _finite(out.get("body_width_mm"))
    depth = _finite(out.get("body_depth_mm"))
    if width is None or not (TABLE_WIDTH_RANGE_MM[0] <= width <= TABLE_WIDTH_RANGE_MM[1]):
        out["body_width_mm"] = None
        out["body_area_mm2"] = None
    if depth is None or not (TABLE_DEPTH_RANGE_MM[0] <= depth <= TABLE_DEPTH_RANGE_MM[1]):
        out["body_depth_mm"] = None
        out["body_area_mm2"] = None
    width = _finite(out.get("body_width_mm"))
    depth = _finite(out.get("body_depth_mm"))
    if width is not None and depth is not None:
        out["body_area_mm2"] = float(width * depth)


def align_ct_to_table_frame(features: Mapping[str, object]) -> Dict[str, object]:
    """Rewrite a CT row so base features match ``excel_displacement_adapter``.

    Needs both kidneys' middle ``*_vert`` points. Without them the row is
    returned unchanged and the patient-mm harmonizer still runs.
    """
    out: Dict[str, object] = dict(features)
    points: Dict[str, Tuple[float, float, float]] = {}
    flipped: List[str] = []
    for side in ("left", "right"):
        coords = [
            _finite(out.get(f"kidney_{side}_middle_{axis}_vert"))
            for axis in "xyz"
        ]
        if any(value is None for value in coords):
            return out
        x_out, y_out, z_out, y_flipped = correct_vertebral_point(
            coords[0], coords[1], coords[2]  # type: ignore[arg-type]
        )
        points[side] = (x_out, y_out, z_out)
        if y_flipped:
            flipped.append(side)

    left = np.array(points["left"], dtype=float)
    right = np.array(points["right"], dtype=float)
    spine = 0.5 * (left + right)
    left_rel = left - spine
    right_rel = right - spine

    width = _finite(out.get("body_width_mm"))
    depth = _finite(out.get("body_depth_mm"))
    if width is None or not (TABLE_WIDTH_RANGE_MM[0] <= width <= TABLE_WIDTH_RANGE_MM[1]):
        width = None
    if depth is None or not (TABLE_DEPTH_RANGE_MM[0] <= depth <= TABLE_DEPTH_RANGE_MM[1]):
        depth = None
    lordosis = _finite(out.get("lumbar_lordosis_deg"))
    tilt = _finite(out.get("s1_plate_tilt_deg"))
    lordosis_rad = np.deg2rad(lordosis) if lordosis is not None else 0.0
    tilt_rad = np.deg2rad(tilt) if tilt is not None else 0.0
    com_y = depth * 0.06 * np.cos(lordosis_rad) if depth is not None else 0.0
    com_z = depth * 0.04 * np.sin(tilt_rad) if depth is not None else 0.0
    com_x = width * 0.02 if width is not None else 0.0
    body_com = spine + np.array([com_x, com_y, com_z], dtype=float)

    for side, point, rel in (
        ("left", left, left_rel),
        ("right", right, right_rel),
    ):
        for axis_i, axis in enumerate("xyz"):
            out[f"kidney_{side}_center_{axis}"] = float(point[axis_i])
            out[f"kidney_{side}_center_{axis}_rel"] = float(rel[axis_i])
            out[f"kidney_{side}_supine_middle_{axis}"] = float(point[axis_i])
        out[f"kidney_{side}_to_spine_distance"] = float(np.linalg.norm(rel))
        out[f"kidney_{side}_to_body_center_distance"] = float(np.linalg.norm(point - body_com))

    for axis_i, axis in enumerate("xyz"):
        out[f"spine_center_{axis}"] = float(spine[axis_i])
        out[f"body_com_{axis}"] = float(body_com[axis_i])
        out[f"kidney_lr_sep_{axis}"] = float(right[axis_i] - left[axis_i])

    _clear_non_table_body_size(out)
    out["feature_frame"] = FEATURE_FRAME_CLINICAL_SIGNED_X
    out["table_frame_source"] = "vertebral_middle"
    out["table_frame_y_flipped"] = flipped
    is_ood, reasons = flag_geometry_ood_x(out)
    out["geometry_ood_x"] = is_ood
    out["geometry_ood_x_reasons"] = reasons
    return out


def geometry_ranges_from_frame(
    frame: pd.DataFrame,
    *,
    q_low: float = 0.01,
    q_high: float = 0.99,
    keys: Sequence[str] = GEOMETRY_RANGE_KEYS,
) -> Dict[str, List[float]]:
    """Training percentile window used by the serve-time Z gate."""
    ranges: Dict[str, List[float]] = {}
    for key in keys:
        if key not in frame.columns:
            continue
        series = pd.to_numeric(frame[key], errors="coerce").dropna()
        if len(series) < 20:
            continue
        ranges[key] = [float(series.quantile(q_low)), float(series.quantile(q_high))]
    return ranges


def constant_target_baseline(
    frame: pd.DataFrame,
    targets: Sequence[str] = Z_TARGET_NAMES,
) -> Dict[str, Dict[str, float]]:
    """MAE of predicting each target by its training mean."""
    out: Dict[str, Dict[str, float]] = {}
    for target in targets:
        if target not in frame.columns:
            continue
        values = pd.to_numeric(frame[target], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        out[target] = {
            "mean_mm": mean,
            "mae_mm": float((values - mean).abs().mean()),
            "n": float(len(values)),
        }
    return out


def flag_table_geometry_ood(
    features: Mapping[str, object],
    ranges: Mapping[str, Sequence[float]] | None,
) -> Tuple[bool, List[str]]:
    """True when converted supine Z or kidney-to-spine distance leaves p1–p99."""
    if not ranges:
        return False, []
    reasons: List[str] = []
    for key, bounds in ranges.items():
        if bounds is None or len(bounds) < 2:
            continue
        value = _finite(features.get(key))
        if value is None:
            continue
        lo, hi = float(bounds[0]), float(bounds[1])
        if value < lo or value > hi:
            reasons.append(f"{key}={value:.1f} outside [{lo:.1f}, {hi:.1f}]")
    return bool(reasons), reasons
