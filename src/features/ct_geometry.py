"""DICOM / NIfTI geometry helpers for Phase 1 feature extraction (patient LPS mm).

Signed clinical-supine X contract
=================================

Extractor / DICOM space
-----------------------
- **LPS** (DICOM): +X toward patient left, +Y toward posterior, +Z toward superior.
- **RAS** (3D Slicer): +X toward patient right, +Y toward anterior, +Z toward superior.
  RAS is LPS with X and Y negated; the map is an involution (round-trip identity).

Clinical Excel / Vybor supine frame (training)
----------------------------------------------
- Absolute clinical X for each kidney is **unsigned laterality**: distance from the
  true mid-sagittal plane (vertebral ``spine_center_x`` when present, otherwise
  patient/scanner origin X=0). Both kidneys sit on the same half-axis (~+60..+90 mm).
- Relative features use the L/R kidney midpoint in *this* frame as origin::

      kidney_{side}_center_x_rel = clinical_x[side] - midpoint_x

  so left and right relative X are opposite signs of equal magnitude.
- ``kidney_lr_sep_x = clinical_x[right] - clinical_x[left]`` (signed, not abs).
- Y and Z are signed offsets from the same L/R midpoint (never abs).

Forbidden: ``abs(kidney_x - kidney_midpoint_x)``
------------------------------------------------
Because ``mid = (L+R)/2``, ``|L-mid| == |R-mid|`` for every pair. Encoding those
unsigned distances as the clinical X values forces::

      left_rel_x == right_rel_x == 0
      kidney_lr_sep_x == 0

The degenerate helper is kept as ``unsigned_x_from_kidney_midpoint`` for tests.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None  # type: ignore[assignment]


def _as_float3(values: Sequence[float], default: Tuple[float, float, float]) -> np.ndarray:
    if values is None or len(values) < 3:
        return np.array(default, dtype=float)
    return np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)


def dicom_pixel_to_patient_mm(ds, row: float, col: float) -> np.ndarray:
    """Convert pixel (row, col) to patient LPS coordinates in mm."""
    ipp = _as_float3(getattr(ds, "ImagePositionPatient", None), (0.0, 0.0, 0.0))
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is not None and len(iop) >= 6:
        row_cos = np.array([float(iop[0]), float(iop[1]), float(iop[2])], dtype=float)
        col_cos = np.array([float(iop[3]), float(iop[4]), float(iop[5])], dtype=float)
    else:
        row_cos = np.array([1.0, 0.0, 0.0], dtype=float)
        col_cos = np.array([0.0, 1.0, 0.0], dtype=float)
    ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
    row_spacing = float(ps[0])
    col_spacing = float(ps[1]) if len(ps) > 1 else row_spacing
    return ipp + row * row_spacing * row_cos + col * col_spacing * col_cos


def dicom_centroid_to_patient_mm(ds, centroid_row: float, centroid_col: float) -> np.ndarray:
    return dicom_pixel_to_patient_mm(ds, centroid_row, centroid_col)


# Feature-frame id written by ``harmonize_ct_to_clinical_frame``.
FEATURE_FRAME_CLINICAL_SIGNED_X = "clinical_midpoint_signed_x"
# Legacy id from the abs-from-kidney-midpoint encoder (always-zero X rel/sep).
FEATURE_FRAME_CLINICAL_UNSIGNED_X = "clinical_midpoint_unsigned_x"

# Clinical train |x_rel| max is ~12 mm; 20 mm is the historical sanity bound.
CLINICAL_X_REL_ABS_MAX_MM = 20.0
CLINICAL_LR_SEP_X_ABS_MAX_MM = 40.0
CLINICAL_TO_SPINE_MAX_MM = 40.0

X_DISTRIBUTION_FEATURES: Tuple[str, ...] = (
    "kidney_left_center_x_rel",
    "kidney_right_center_x_rel",
    "kidney_lr_sep_x",
)


def patient_kidney_side(kidney_x: float, spine_x: float) -> str:
    """LPS: patient-left kidney has larger X than spine."""
    return "left" if kidney_x >= spine_x else "right"


def flip_lps_ras(coords: np.ndarray) -> np.ndarray:
    """LPS ↔ RAS: negate X and Y. Involution (applying twice is identity)."""
    out = np.asarray(coords, dtype=float).copy()
    if out.ndim == 1:
        out[0] *= -1.0
        out[1] *= -1.0
        return out
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


def unsigned_x_from_kidney_midpoint(left_x: float, right_x: float) -> Tuple[float, float]:
    """Broken helper: unsigned distance of each kidney from the L/R X midpoint.

    Algebraic identity: ``mid = (L+R)/2`` implies ``|L-mid| == |R-mid|`` always,
    so the two returned values are equal and any later midpoint-relative encoding
    collapses ``*_x_rel`` and ``kidney_lr_sep_x`` to zero.

    Kept only to reproduce the defect in tests. Do not call from production.
    """
    mid_x = 0.5 * (float(left_x) + float(right_x))
    return abs(float(left_x) - mid_x), abs(float(right_x) - mid_x)


def resolve_mid_sagittal_x(
    features: Mapping[str, object],
    left_x: float,
    right_x: float,
) -> Tuple[float, str]:
    """True mid-sagittal X in patient mm; never a freshly computed kidney midpoint.

    Prefers vertebral ``spine_center_x`` (even when it happens to lie near the
    kidney midpoint — that is anatomy, and keeps symmetric-about-spine cases at
    rel X ≈ 0). Falls back to ``body_com_x``, then patient/scanner origin (0).
    """
    del left_x, right_x  # origin must not be derived from the L/R kidney pair
    for key in ("spine_center_x", "body_com_x"):
        value = _finite(features.get(key))
        if value is not None:
            return value, key
    return 0.0, "patient_origin_x"


def clinical_x_from_patient_x(
    left_x: float,
    right_x: float,
    midline_x: float,
) -> Tuple[float, float]:
    """Map signed patient X (LPS or RAS) to clinical unsigned laterality.

    ``clinical_x = |patient_x - midline_x|``. The midline must be the true
    mid-sagittal plane, not the L/R kidney midpoint. Relative X is then the
    *signed* offset of these values from their midpoint (see
    ``harmonize_ct_to_clinical_frame``).
    """
    return abs(float(left_x) - float(midline_x)), abs(float(right_x) - float(midline_x))


def signed_rel_from_clinical_pair(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spine = L/R midpoint; relative coords = kidney − spine (signed, no abs)."""
    spine = 0.5 * (left_xyz + right_xyz)
    return left_xyz - spine, right_xyz - spine, spine


def flag_geometry_ood_x(
    features: Mapping[str, object],
    *,
    x_rel_abs_max: float = CLINICAL_X_REL_ABS_MAX_MM,
    lr_sep_abs_max: float = CLINICAL_LR_SEP_X_ABS_MAX_MM,
    to_spine_max: float = CLINICAL_TO_SPINE_MAX_MM,
) -> Tuple[bool, List[str]]:
    """Return (is_ood, reasons) when X features leave the clinical train envelope."""
    reasons: List[str] = []
    left_rel = _finite(features.get("kidney_left_center_x_rel"))
    right_rel = _finite(features.get("kidney_right_center_x_rel"))
    sep = _finite(features.get("kidney_lr_sep_x"))
    left_dist = _finite(features.get("kidney_left_to_spine_distance"))
    right_dist = _finite(features.get("kidney_right_to_spine_distance"))
    if left_rel is not None and abs(left_rel) > x_rel_abs_max:
        reasons.append(f"kidney_left_center_x_rel={left_rel:.3f} exceeds ±{x_rel_abs_max}")
    if right_rel is not None and abs(right_rel) > x_rel_abs_max:
        reasons.append(f"kidney_right_center_x_rel={right_rel:.3f} exceeds ±{x_rel_abs_max}")
    if sep is not None and abs(sep) > lr_sep_abs_max:
        reasons.append(f"kidney_lr_sep_x={sep:.3f} exceeds ±{lr_sep_abs_max}")
    if left_dist is not None and left_dist > to_spine_max:
        reasons.append(f"kidney_left_to_spine_distance={left_dist:.3f} exceeds {to_spine_max}")
    if right_dist is not None and right_dist > to_spine_max:
        reasons.append(f"kidney_right_to_spine_distance={right_dist:.3f} exceeds {to_spine_max}")
    return bool(reasons), reasons


def voxels_to_patient_mm(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """Map Nx3 voxel indices to Nx3 patient mm."""
    coords = np.asarray(ijk, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1, 3)
    hom = np.concatenate([coords, np.ones((coords.shape[0], 1), dtype=float)], axis=1)
    return (hom @ affine.T)[:, :3]


def mask_centroid_patient_mm(mask: np.ndarray, affine: np.ndarray) -> Optional[np.ndarray]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    mm = voxels_to_patient_mm(affine, coords.astype(float))
    return mm.mean(axis=0)


def mask_extent_patient_mm(
    mask: np.ndarray,
    affine: np.ndarray,
    zooms: Tuple[float, float, float],
) -> Dict[str, object]:
    """Volume, cranio-caudal length, and upper/middle/lower points in patient mm."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return {}
    mm = voxels_to_patient_mm(affine, coords.astype(float))
    voxel_volume_mm3 = float(np.prod(zooms))
    volume_cm3 = len(coords) * voxel_volume_mm3 / 1000.0

    z_vals = mm[:, 2]
    z_min, z_max = float(z_vals.min()), float(z_vals.max())
    length_mm = z_max - z_min

    def _point_at_z(target_z: float) -> np.ndarray:
        band = mm[np.abs(mm[:, 2] - target_z) <= max(zooms[2], 1.0)]
        if len(band) == 0:
            idx = int(np.argmin(np.abs(z_vals - target_z)))
            return mm[idx]
        return band.mean(axis=0)

    z_mid = (z_min + z_max) / 2.0
    upper = _point_at_z(z_max)
    middle = _point_at_z(z_mid)
    lower = _point_at_z(z_min)

    return {
        "center": mm.mean(axis=0),
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "volume_cm3": volume_cm3,
        "length_mm": length_mm,
        "z_min": z_min,
        "z_max": z_max,
    }


def kidney_features_from_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    zooms: Tuple[float, float, float],
    prefix: str,
) -> Dict[str, float]:
    """Build extractor-style kidney keys from a 3D boolean mask."""
    stats = mask_extent_patient_mm(mask, affine, zooms)
    if not stats:
        return {}

    out: Dict[str, float] = {}
    for point_name, suffix in (("upper", "upper"), ("middle", "middle"), ("lower", "lower")):
        pt = stats[point_name]
        for axis, i in zip("xyz", range(3)):
            out[f"{prefix}_{suffix}_{axis}"] = float(pt[i])

    center = stats["center"]
    for axis, i in zip("xyz", range(3)):
        out[f"{prefix}_center_{axis}"] = float(center[i])

    out[f"{prefix}_volume_cm3"] = float(stats["volume_cm3"])
    out[f"{prefix}_length_mm"] = float(stats["length_mm"])
    return out


def merge_spine_relative(
    features: Dict[str, Optional[float]],
    spine_x: Optional[float],
    spine_y: Optional[float],
    spine_z: Optional[float],
) -> Dict[str, Optional[float]]:
    """Fill vs_spine and rel center columns from absolute patient-mm coords."""
    out = dict(features)
    if spine_x is None or spine_y is None or spine_z is None:
        return out

    out["spine_center_x"] = spine_x
    out["spine_center_y"] = spine_y
    out["spine_center_z"] = spine_z

    for side in ("left", "right"):
        prefix = f"kidney_{side}"
        cx = out.get(f"{prefix}_center_x")
        cy = out.get(f"{prefix}_center_y")
        cz = out.get(f"{prefix}_center_z")
        if cx is None or cy is None or cz is None:
            continue
        out[f"{prefix}_vs_spine_x"] = float(cx - spine_x)
        out[f"{prefix}_vs_spine_y"] = float(cy - spine_y)
        out[f"{prefix}_vs_spine_z"] = float(cz - spine_z)
        out[f"{prefix}_center_x_rel"] = out[f"{prefix}_vs_spine_x"]
        out[f"{prefix}_center_y_rel"] = out[f"{prefix}_vs_spine_y"]
        out[f"{prefix}_center_z_rel"] = out[f"{prefix}_vs_spine_z"]

        dist = float(np.sqrt(
            (cx - spine_x) ** 2 + (cy - spine_y) ** 2 + (cz - spine_z) ** 2
        ))
        out[f"{prefix}_to_spine_distance"] = dist

        bcx = out.get("body_com_x")
        bcy = out.get("body_com_y")
        bcz = out.get("body_com_z")
        if bcx is not None and bcy is not None and bcz is not None:
            out[f"{prefix}_to_body_center_distance"] = float(np.sqrt(
                (cx - bcx) ** 2 + (cy - bcy) ** 2 + (cz - bcz) ** 2
            ))
    return out


def _finite(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def harmonize_ct_to_clinical_frame(features: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Map CT patient-mm kidney centers to the Vybor/Excel clinical feature frame.

    X uses unsigned laterality from the true mid-sagittal plane, then a *signed*
    midpoint encoding (see module docstring). Y/Z stay signed throughout.
    ``abs(x - kidney_midpoint)`` is not used: that identity zeros relative X.

    Expects raw patient-mm kidney centers (LPS or RAS). Rows already tagged
    ``feature_frame=clinical_midpoint_signed_x`` are returned unchanged so a
    second pass cannot treat the rewritten midpoint spine as a mid-sagittal origin.
    """
    out = dict(features)
    if out.get("feature_frame") in (
        FEATURE_FRAME_CLINICAL_SIGNED_X,
        FEATURE_FRAME_CLINICAL_UNSIGNED_X,
    ):
        return out
    lx = _finite(out.get("kidney_left_center_x"))
    ly = _finite(out.get("kidney_left_center_y"))
    lz = _finite(out.get("kidney_left_center_z"))
    rx = _finite(out.get("kidney_right_center_x"))
    ry = _finite(out.get("kidney_right_center_y"))
    rz = _finite(out.get("kidney_right_center_z"))
    if None in (lx, ly, lz, rx, ry, rz):
        return out

    midline_x, midline_source = resolve_mid_sagittal_x(out, lx, rx)
    left_clin_x, right_clin_x = clinical_x_from_patient_x(lx, rx, midline_x)

    mid_y = 0.5 * (ly + ry)
    mid_z = 0.5 * (lz + rz)
    left_sup = np.array([left_clin_x, ly - mid_y, lz - mid_z], dtype=float)
    right_sup = np.array([right_clin_x, ry - mid_y, rz - mid_z], dtype=float)
    left_rel, right_rel, spine = signed_rel_from_clinical_pair(left_sup, right_sup)

    out["mid_sagittal_x"] = float(midline_x)
    out["mid_sagittal_x_source"] = midline_source
    out["spine_center_x"] = float(spine[0])
    out["spine_center_y"] = float(spine[1])
    out["spine_center_z"] = float(spine[2])
    out["kidney_left_center_x_rel"] = float(left_rel[0])
    out["kidney_left_center_y_rel"] = float(left_rel[1])
    out["kidney_left_center_z_rel"] = float(left_rel[2])
    out["kidney_right_center_x_rel"] = float(right_rel[0])
    out["kidney_right_center_y_rel"] = float(right_rel[1])
    out["kidney_right_center_z_rel"] = float(right_rel[2])
    out["kidney_left_to_spine_distance"] = float(np.linalg.norm(left_rel))
    out["kidney_right_to_spine_distance"] = float(np.linalg.norm(right_rel))

    # Body COM offsets mirror excel_displacement_adapter defaults when depth/width exist.
    width = _finite(out.get("body_width_mm"))
    depth = _finite(out.get("body_depth_mm"))
    com_x_off = (width * 0.02) if width is not None else 0.0
    com_y_off = (depth * 0.06) if depth is not None else 0.0
    out["body_com_x"] = float(spine[0] + com_x_off)
    out["body_com_y"] = float(spine[1] + com_y_off)
    out["body_com_z"] = float(spine[2])
    out["kidney_left_to_body_center_distance"] = float(
        np.linalg.norm(left_sup - np.array([out["body_com_x"], out["body_com_y"], out["body_com_z"]]))
    )
    out["kidney_right_to_body_center_distance"] = float(
        np.linalg.norm(right_sup - np.array([out["body_com_x"], out["body_com_y"], out["body_com_z"]]))
    )
    # Signed L/R separation in the clinical frame (right − left), not abs.
    out["kidney_lr_sep_x"] = float(right_sup[0] - left_sup[0])
    out["kidney_lr_sep_y"] = float(right_sup[1] - left_sup[1])
    out["kidney_lr_sep_z"] = float(right_sup[2] - left_sup[2])
    out["feature_frame"] = FEATURE_FRAME_CLINICAL_SIGNED_X

    is_ood, reasons = flag_geometry_ood_x(out)
    out["geometry_ood_x"] = is_ood
    out["geometry_ood_x_reasons"] = reasons
    return out


def sanitize_body_size_for_clinical_model(
    features: Dict[str, Optional[float]],
    *,
    width_range: Tuple[float, float] = (200.0, 500.0),
    depth_range: Tuple[float, float] = (140.0, 400.0),
) -> Dict[str, Optional[float]]:
    """Drop FOV-cropped CT body sizes that are outside the clinical train range."""
    out = dict(features)
    width = _finite(out.get("body_width_mm"))
    depth = _finite(out.get("body_depth_mm"))
    if width is not None and not (width_range[0] <= width <= width_range[1]):
        out["body_width_mm"] = None
        out["body_area_mm2"] = None
    if depth is not None and not (depth_range[0] <= depth <= depth_range[1]):
        out["body_depth_mm"] = None
        out["body_area_mm2"] = None
    width = _finite(out.get("body_width_mm"))
    depth = _finite(out.get("body_depth_mm"))
    if width is not None and depth is not None and _finite(out.get("body_area_mm2")) is None:
        out["body_area_mm2"] = float(width * depth)
    return out


def aggregate_body_at_z_band(
    slice_metrics: List[Dict[str, float]],
    z_min: float,
    z_max: float,
) -> Dict[str, Optional[float]]:
    """Median body geometry on slices overlapping kidney cranio-caudal extent."""
    band = [
        m for m in slice_metrics
        if z_min <= m.get("slice_z", 0.0) <= z_max
    ]
    if not band:
        band = slice_metrics
    if not band:
        return {}

    def _med(key: str) -> Optional[float]:
        vals = [m[key] for m in band if key in m and m[key] is not None]
        return float(np.median(vals)) if vals else None

    width = _med("body_width_mm")
    depth = _med("body_depth_mm")
    area = _med("body_area_mm2")
    if area is None and width is not None and depth is not None:
        area = width * depth

    return {
        "body_width_mm": width,
        "body_depth_mm": depth,
        "body_area_mm2": area,
        "body_com_x": _med("body_com_x"),
        "body_com_y": _med("body_com_y"),
        "body_com_z": _med("body_com_z"),
    }
