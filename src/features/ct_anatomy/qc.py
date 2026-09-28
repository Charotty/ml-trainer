"""Plausibility limits and the Excel body-habitus code derived from BMI.

QC values written next to a measurement:
  ok            computed inside the volume
  approx        proxy landmark or a definition that still needs a clinician
  out_of_fov    the source mask touches the scan border
  missing_mask  the structure was not segmented
  out_of_range  the number is outside the limits below and is cleared
  manual        not measurable on CT (diagnosis, prior surgery)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

QC_OK = "ok"
QC_APPROX = "approx"
QC_OUT_OF_FOV = "out_of_fov"
QC_MISSING = "missing_mask"
QC_OUT_OF_RANGE = "out_of_range"
QC_MANUAL = "manual"

# Excel "Телосложение": 0 норма, 1 астеник, 2 гиперстеник.
# Cutoffs are the WHO BMI bands used as a stand-in for that habitus code.
ASTHENIC_BMI = 18.5
HYPERSTHENIC_BMI = 25.0

SHARED_RANGES: Dict[str, Tuple[float, float]] = {
    "abd_width_l3l4_mm": (200.0, 500.0),
    "abd_depth_l3l4_mm": (140.0, 400.0),
    "abd_wall_thickness_mm": (2.0, 80.0),
    "lumbar_lordosis_deg": (5.0, 90.0),
    "s1_plate_tilt_deg": (-30.0, 80.0),
    "disc_l3l4_anterior_height_mm": (1.0, 25.0),
    "disc_l3l4_posterior_height_mm": (1.0, 25.0),
    "disc_l4l5_anterior_height_mm": (1.0, 25.0),
    "disc_l4l5_posterior_height_mm": (1.0, 25.0),
    "body_type": (0.0, 2.0),
}

PER_SIDE_RANGES: Dict[str, Tuple[float, float]] = {
    "upper_pole_to_rib11_mm": (0.0, 250.0),
    "lower_pole_to_iliac_crest_mm": (0.0, 250.0),
    "upper_pole_to_diaphragm_mm": (0.0, 250.0),
    "medial_to_spine_mm": (0.0, 80.0),
    "perirenal_dorsal_mm": (0.0, 60.0),
    "perirenal_ventral_mm": (0.0, 60.0),
    "perirenal_lateral_mm": (0.0, 60.0),
    "perirenal_hu": (-160.0, 30.0),
    "perirenal_stranding": (0.0, 1.0),
    "map_score": (0.0, 5.0),
    "psoas_area_cm2": (0.3, 40.0),
    "psoas_thickness_mm": (3.0, 60.0),
    "rotation_a_deg": (-90.0, 90.0),
    "rotation_b_deg": (-90.0, 90.0),
    "rotation_c_deg": (-180.0, 180.0),
    "pedicle_length_mm": (5.0, 150.0),
    "pedicle_origin_angle_deg": (-180.0, 180.0),
    "pedicle_drop_angle_deg": (-180.0, 180.0),
}

# Columns the validation script compares with «Смещение - конечное -13».
COMPARISON_FEATURES = (
    list(SHARED_RANGES)
    + [f"kidney_{side}_{suffix}" for side in ("left", "right") for suffix in PER_SIDE_RANGES]
)


def body_type_from_bmi(bmi: object) -> Optional[float]:
    """Map BMI onto the Excel habitus code. Unknown BMI stays missing."""
    try:
        value = float(bmi)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    if value < ASTHENIC_BMI:
        return 1.0
    if value < HYPERSTHENIC_BMI:
        return 0.0
    return 2.0


def _ranges_in(features: Dict[str, object]) -> Dict[str, Tuple[float, float]]:
    found: Dict[str, Tuple[float, float]] = {}
    for key, bounds in SHARED_RANGES.items():
        if key in features:
            found[key] = bounds
    for side in ("left", "right"):
        for suffix, bounds in PER_SIDE_RANGES.items():
            key = f"kidney_{side}_{suffix}"
            if key in features:
                found[key] = bounds
    return found


def finalize_qc(features: Dict[str, object]) -> Dict[str, object]:
    """Clear out-of-range numbers. Do not overwrite a QC flag already set."""
    out = dict(features)
    for key, (low, high) in _ranges_in(out).items():
        qc_key = f"{key}_qc"
        value = out.get(key)
        if value is None:
            out.setdefault(qc_key, QC_MISSING)
            continue
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            out[key] = None
            out[qc_key] = QC_MISSING
            continue
        if not np.isfinite(number) or not (low <= number <= high):
            out[key] = None
            out[qc_key] = QC_OUT_OF_RANGE
            continue
        out.setdefault(qc_key, QC_OK)
    return out
