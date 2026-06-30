"""Extra engineered features aimed at kidney displacement Y and Z axes."""

from __future__ import annotations

import numpy as np
import pandas as pd

DISPLACEMENT_AXIS_FEATURES: list[str] = [
    "kidney_z_asymmetry_rel",
    "kidney_y_asymmetry_rel",
    "kidney_left_z_over_depth",
    "kidney_right_z_over_depth",
    "kidney_left_y_over_depth",
    "kidney_right_y_over_depth",
    "body_sagittal_index",
    "lordosis_x_depth",
    "abd_wall_over_depth",
    "kidney_left_z_span_supine_mm",
    "kidney_right_z_span_supine_mm",
    "kidney_left_y_span_supine_mm",
    "kidney_right_y_span_supine_mm",
    # delta_span_* excluded: lateral-derived leakage (see leakage_safe.py)
]

# Supine-only anatomical inputs (non-degenerate vs kidney-midpoint frame).
ANATOMICAL_FEATURES: list[str] = [
    "kidney_lr_sep_x",
    "kidney_lr_sep_y",
    "kidney_lr_sep_z",
    "kidney_left_supine_middle_x",
    "kidney_left_supine_middle_y",
    "kidney_left_supine_middle_z",
    "kidney_right_supine_middle_x",
    "kidney_right_supine_middle_y",
    "kidney_right_supine_middle_z",
    "lumbar_lordosis_deg",
    "s1_plate_tilt_deg",
    "abd_wall_thickness_mm",
]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.astype(float).replace(0, np.nan)
    return num.astype(float) / den


def add_displacement_axis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Y/Z-oriented features; missing inputs stay NaN for imputer."""
    out = df.copy()

    if all(c in out.columns for c in ("kidney_left_center_z_rel", "kidney_right_center_z_rel")):
        out["kidney_z_asymmetry_rel"] = (
            out["kidney_left_center_z_rel"].astype(float)
            - out["kidney_right_center_z_rel"].astype(float)
        )

    if all(c in out.columns for c in ("kidney_left_center_y_rel", "kidney_right_center_y_rel")):
        out["kidney_y_asymmetry_rel"] = (
            out["kidney_left_center_y_rel"].astype(float)
            - out["kidney_right_center_y_rel"].astype(float)
        )

    if "body_depth_mm" in out.columns:
        depth = out["body_depth_mm"]
        for side in ("left", "right"):
            z_col = f"kidney_{side}_center_z_rel"
            y_col = f"kidney_{side}_center_y_rel"
            if z_col in out.columns:
                out[f"kidney_{side}_z_over_depth"] = _safe_div(out[z_col], depth)
            if y_col in out.columns:
                out[f"kidney_{side}_y_over_depth"] = _safe_div(out[y_col], depth)

    if all(c in out.columns for c in ("body_depth_mm", "body_width_mm")):
        out["body_sagittal_index"] = _safe_div(out["body_depth_mm"], out["body_width_mm"])

    if all(c in out.columns for c in ("lumbar_lordosis_deg", "body_depth_mm")):
        out["lordosis_x_depth"] = _safe_div(out["lumbar_lordosis_deg"], out["body_depth_mm"])

    if "abd_wall_thickness_mm" in out.columns and "body_depth_mm" in out.columns:
        out["abd_wall_over_depth"] = _safe_div(out["abd_wall_thickness_mm"], out["body_depth_mm"])

    return out


CLINICAL_EXTRA_COLUMNS: list[str] = [
    "abd_wall_thickness_mm",
    "lumbar_lordosis_deg",
    "s1_plate_tilt_deg",
    "kidney_left_z_span_supine_mm",
    "kidney_right_z_span_supine_mm",
    "kidney_left_y_span_supine_mm",
    "kidney_right_y_span_supine_mm",
    "kidney_left_z_delta_span_mm",
    "kidney_right_z_delta_span_mm",
    "kidney_left_y_delta_span_mm",
    "kidney_right_y_delta_span_mm",
]
