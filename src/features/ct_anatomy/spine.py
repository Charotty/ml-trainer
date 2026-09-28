"""Vertebral levels, lumbar lordosis, S1 tilt and disc heights.

Endplate tilt uses the supine scanner axis (+Z superior), not the standing
vertical. Lumbar lordosis is the Cobb angle between the L1 and S1 endplates
in the sagittal plane. Disc height is the gap between facing endplate voxels
in the midsagittal anterior and posterior bands.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.features.ct_anatomy.profiles import VERTEBRA_LEVELS
from src.features.ct_anatomy.qc import QC_APPROX, QC_MISSING, QC_OUT_OF_FOV
from src.features.ct_anatomy.volume import AnatomyVolume, mask_points_mm


def _body_mask(volume: AnatomyVolume, level: str) -> Optional[np.ndarray]:
    named = volume.mask(f"vertebrae_{level}")
    if named is None:
        return None
    body = volume.mask("vertebrae_body")
    if body is None or body.shape != named.shape:
        return named
    both = named & body
    if not np.any(both):
        return named
    return both


def fit_endplate_normal(points: np.ndarray) -> Optional[np.ndarray]:
    """Smallest-variance axis of a vertebral body, oriented toward +Z.

    For a slab-like vertebral body this is the endplate normal. Returns None
    when the body is not flat enough for a stable plane.
    """
    if len(points) < 20:
        return None
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)
    if evals[order[1]] < 3.0 * max(evals[order[0]], 1e-6):
        return None
    normal = evecs[:, order[0]].astype(float)
    if normal[2] < 0:
        normal = -normal
    return normal


def sagittal_tilt_deg(normal: np.ndarray) -> float:
    """Signed endplate angle from the axial plane, degrees.

    0: endplate horizontal (normal along +Z). Positive: normal tilts posterior (+Y).
    """
    return float(np.degrees(np.arctan2(float(normal[1]), float(normal[2]))))


def _disc_gap_mm(
    upper: np.ndarray,
    lower: np.ndarray,
    *,
    anterior: bool,
) -> Optional[float]:
    if len(upper) < 10 or len(lower) < 10:
        return None
    x_mid = 0.5 * (float(np.median(upper[:, 0])) + float(np.median(lower[:, 0])))
    x_span = max(float(upper[:, 0].max() - upper[:, 0].min()), 1.0)
    band = max(2.0, 0.15 * x_span)
    upper = upper[np.abs(upper[:, 0] - x_mid) <= band]
    lower = lower[np.abs(lower[:, 0] - x_mid) <= band]
    if len(upper) < 5 or len(lower) < 5:
        return None
    y_all = np.concatenate([upper[:, 1], lower[:, 1]])
    y_span = max(float(y_all.max() - y_all.min()), 1.0)
    width = max(2.0, 0.15 * y_span)
    cut = float(np.quantile(y_all, 0.25 if anterior else 0.75))
    if anterior:
        upper_band = upper[upper[:, 1] <= cut + width]
        lower_band = lower[lower[:, 1] <= cut + width]
    else:
        upper_band = upper[upper[:, 1] >= cut - width]
        lower_band = lower[lower[:, 1] >= cut - width]
    if len(upper_band) < 3 or len(lower_band) < 3:
        return None
    gap = float(upper_band[:, 2].min() - lower_band[:, 2].max())
    if gap <= 0:
        return None
    return gap


def measure_spine(volume: AnatomyVolume) -> Dict[str, object]:
    out: Dict[str, object] = {}
    centroids: Dict[str, np.ndarray] = {}
    for level in VERTEBRA_LEVELS:
        name = f"vertebrae_{level}"
        center = volume.centroid(name)
        if center is None:
            continue
        centroids[level] = center
        out[f"{name}_x"] = float(center[0])
        out[f"{name}_y"] = float(center[1])
        out[f"{name}_z"] = float(center[2])
        qc = QC_OUT_OF_FOV if volume.touches_border(name) else "ok"
        out[f"{name}_z_qc"] = qc

    if "L3" in centroids and "L4" in centroids:
        out["l3_l4_z"] = float(0.5 * (centroids["L3"][2] + centroids["L4"][2]))
        out["l3_l4_z_qc"] = "ok"
    else:
        out["l3_l4_z"] = None
        out["l3_l4_z_qc"] = QC_MISSING

    if "L4" in centroids and "L5" in centroids:
        out["l4_l5_z"] = float(0.5 * (centroids["L4"][2] + centroids["L5"][2]))
    else:
        out["l4_l5_z"] = None

    normals: Dict[str, np.ndarray] = {}
    for level in ("L1", "S1"):
        mask = _body_mask(volume, level)
        if mask is None:
            continue
        normal = fit_endplate_normal(mask_points_mm(mask, volume.affine))
        if normal is not None:
            normals[level] = normal

    if "L1" in normals and "S1" in normals:
        tilt_l1 = sagittal_tilt_deg(normals["L1"])
        tilt_s1 = sagittal_tilt_deg(normals["S1"])
        out["lumbar_lordosis_deg"] = abs(tilt_l1 - tilt_s1)
        out["s1_plate_tilt_deg"] = tilt_s1
        edge = volume.touches_border("vertebrae_L1") or volume.touches_border("vertebrae_S1")
        qc = QC_OUT_OF_FOV if edge else "ok"
        out["lumbar_lordosis_deg_qc"] = qc
        out["s1_plate_tilt_deg_qc"] = qc
    else:
        out["lumbar_lordosis_deg"] = None
        out["s1_plate_tilt_deg"] = None
        out["lumbar_lordosis_deg_qc"] = QC_MISSING
        out["s1_plate_tilt_deg_qc"] = QC_MISSING

    pairs = (("L3", "L4", "disc_l3l4"), ("L4", "L5", "disc_l4l5"))
    for upper_level, lower_level, key in pairs:
        upper_mask = _body_mask(volume, upper_level)
        lower_mask = _body_mask(volume, lower_level)
        if upper_mask is None or lower_mask is None:
            for side_name in ("anterior", "posterior"):
                out[f"{key}_{side_name}_height_mm"] = None
                out[f"{key}_{side_name}_height_mm_qc"] = QC_MISSING
            continue
        upper_pts = mask_points_mm(upper_mask, volume.affine)
        lower_pts = mask_points_mm(lower_mask, volume.affine)
        for side_name, anterior in (("anterior", True), ("posterior", False)):
            gap = _disc_gap_mm(upper_pts, lower_pts, anterior=anterior)
            feature = f"{key}_{side_name}_height_mm"
            out[feature] = gap
            if gap is None:
                out[f"{feature}_qc"] = QC_MISSING
            elif volume.touches_border(f"vertebrae_{upper_level}"):
                out[f"{feature}_qc"] = QC_OUT_OF_FOV
            else:
                out[f"{feature}_qc"] = "ok"

    # Combined disc mask is not required; if present it does not replace the
    # endplate gap, which matches the anterior/posterior disc-height columns.
    if volume.mask("intervertebral_discs") is None and out.get("disc_l3l4_anterior_height_mm") is None:
        out["disc_height_source"] = None
    else:
        out["disc_height_source"] = "endplate_gap"
        if volume.mask("intervertebral_discs") is None:
            out["disc_l3l4_anterior_height_mm_qc"] = out.get(
                "disc_l3l4_anterior_height_mm_qc", QC_APPROX
            )
    return out
