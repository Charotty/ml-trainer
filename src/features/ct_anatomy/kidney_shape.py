"""Kidney thirds, provisional rotation angles and contrast pedicle check.

Angles A/B/C are not a published clinical definition. They are the long-axis
tilt in the coronal and sagittal planes, plus the axial direction of the
medial concavity. A vessel length is reported only when aortic HU shows
contrast and a bright path reaches the hilum; otherwise the pedicle stays empty.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_erosion

from src.features.ct_anatomy.qc import QC_APPROX, QC_MISSING
from src.features.ct_anatomy.volume import PATIENT_Z, AnatomyVolume, mask_points_mm, voxel_axis_for

CONTRAST_HU = 150.0
AORTA_CONTRAST_HU = 100.0


def _long_axis(points: np.ndarray) -> Optional[np.ndarray]:
    if len(points) < 10:
        return None
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))].astype(float)
    if axis[2] < 0:
        axis = -axis
    return axis


def _hilum_vector(volume: AnatomyVolume, side: str, z_mm: float) -> Optional[np.ndarray]:
    mask = volume.mask(f"kidney_{side}")
    if mask is None:
        return None
    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    # Keep boundary voxels near the hilum slice.
    axis_z = volume.axial_index_for_z(z_mm)
    axis, _sign = voxel_axis_for(volume.affine, PATIENT_Z)
    selector = [slice(None), slice(None), slice(None)]
    lo = max(0, axis_z - 1)
    hi = min(mask.shape[axis], axis_z + 2)
    selector[axis] = slice(lo, hi)
    band = np.zeros_like(boundary)
    band[tuple(selector)] = boundary[tuple(selector)]
    points = volume.points(f"kidney_{side}")
    if len(points) == 0 or not np.any(band):
        return None
    center = points.mean(axis=0)
    boundary_pts = mask_points_mm(band, volume.affine)
    if len(boundary_pts) == 0:
        return None
    # Medial is toward the midline: −X for the left kidney, +X for the right.
    medial = -1.0 if side == "left" else 1.0
    medial_pts = boundary_pts[(boundary_pts[:, 0] - center[0]) * medial > 0]
    pool = medial_pts if len(medial_pts) else boundary_pts
    dist = np.linalg.norm(pool - center, axis=1)
    hilum = pool[int(np.argmin(dist))]
    return hilum - center


def _thirds(points: np.ndarray) -> Dict[str, np.ndarray]:
    z = points[:, 2]
    edges = np.linspace(float(z.min()), float(z.max()), 4)
    # Right edge is inclusive so the superior voxel is not dropped.
    edges[-1] = edges[-1] + 1e-3
    found = {}
    for name, lo, hi in (
        ("lower", edges[0], edges[1]),
        ("middle", edges[1], edges[2]),
        ("upper", edges[2], edges[3]),
    ):
        band = points[(z >= lo) & (z < hi)]
        if len(band):
            found[name] = band.mean(axis=0)
    return found


def _pedicle(
    volume: AnatomyVolume,
    hilum_point: Optional[np.ndarray],
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """Return length, origin angle, drop angle and status."""
    if hilum_point is None or volume.hu is None or volume.mask("aorta") is None:
        return None, None, None, "not_assessed"
    aorta_pts = volume.points("aorta")
    if len(aorta_pts) == 0:
        return None, None, None, "not_assessed"
    origin_i = int(np.argmin(np.linalg.norm(aorta_pts - hilum_point, axis=1)))
    origin = aorta_pts[origin_i]
    aorta_hu = np.array([volume.sample_hu(p) for p in aorta_pts[:: max(1, len(aorta_pts) // 30)]])
    if not np.isfinite(aorta_hu).any() or float(np.nanmean(aorta_hu)) < AORTA_CONTRAST_HU:
        return None, None, None, "no_contrast"
    samples = np.linspace(origin, hilum_point, 40)
    hus = np.array([volume.sample_hu(p) for p in samples])
    bright = np.isfinite(hus) & (hus >= CONTRAST_HU)
    if bright.mean() < 0.3:
        return None, None, None, "not_found"
    # Length follows the bright prefix from the aorta, not the renal parenchyma.
    last = int(np.where(bright)[0][-1])
    length = float(np.linalg.norm(samples[last] - origin))
    vec = hilum_point - origin
    origin_angle = float(np.degrees(np.arctan2(vec[2], vec[0])))
    drop_angle = float(np.degrees(np.arctan2(vec[2], vec[1] if abs(vec[1]) > 1e-3 else 1e-3)))
    return length, origin_angle, drop_angle, "approx"


def measure_kidney_shape(volume: AnatomyVolume) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for side in ("left", "right"):
        prefix = f"kidney_{side}"
        points = volume.points(f"kidney_{side}")
        if len(points) < 10:
            for name in ("upper", "middle", "lower"):
                for axis in "xyz":
                    key = f"{prefix}_third_{name}_{axis}"
                    out[key] = None
                    out[f"{key}_qc"] = QC_MISSING
            for suffix in ("rotation_a_deg", "rotation_b_deg", "rotation_c_deg"):
                out[f"{prefix}_{suffix}"] = None
                out[f"{prefix}_{suffix}_qc"] = QC_MISSING
            for suffix in ("pedicle_length_mm", "pedicle_origin_angle_deg", "pedicle_drop_angle_deg"):
                out[f"{prefix}_{suffix}"] = None
                out[f"{prefix}_{suffix}_qc"] = QC_MISSING
            out[f"{prefix}_pedicle_status"] = "not_assessed"
            continue

        for name, center in _thirds(points).items():
            for axis_i, axis_name in enumerate("xyz"):
                key = f"{prefix}_third_{name}_{axis_name}"
                out[key] = float(center[axis_i])
                out[f"{key}_qc"] = "ok"

        axis = _long_axis(points)
        z_mid = float(points[:, 2].mean())
        hilum_vec = _hilum_vector(volume, side, z_mid)
        if axis is None:
            out[f"{prefix}_rotation_a_deg"] = None
            out[f"{prefix}_rotation_b_deg"] = None
            out[f"{prefix}_rotation_a_deg_qc"] = QC_MISSING
            out[f"{prefix}_rotation_b_deg_qc"] = QC_MISSING
        else:
            out[f"{prefix}_rotation_a_deg"] = float(np.degrees(np.arctan2(axis[0], axis[2])))
            out[f"{prefix}_rotation_b_deg"] = float(np.degrees(np.arctan2(axis[1], axis[2])))
            out[f"{prefix}_rotation_a_deg_qc"] = QC_APPROX
            out[f"{prefix}_rotation_b_deg_qc"] = QC_APPROX
        if hilum_vec is None:
            out[f"{prefix}_rotation_c_deg"] = None
            out[f"{prefix}_rotation_c_deg_qc"] = QC_MISSING
            hilum_point = None
        else:
            out[f"{prefix}_rotation_c_deg"] = float(np.degrees(np.arctan2(hilum_vec[1], hilum_vec[0])))
            out[f"{prefix}_rotation_c_deg_qc"] = QC_APPROX
            hilum_point = points.mean(axis=0) + hilum_vec

        length, origin_angle, drop_angle, status = _pedicle(volume, hilum_point)
        out[f"{prefix}_pedicle_length_mm"] = length
        out[f"{prefix}_pedicle_origin_angle_deg"] = origin_angle
        out[f"{prefix}_pedicle_drop_angle_deg"] = drop_angle
        out[f"{prefix}_pedicle_status"] = status
        pedicle_qc = QC_APPROX if length is not None else QC_MISSING
        for suffix in ("pedicle_length_mm", "pedicle_origin_angle_deg", "pedicle_drop_angle_deg"):
            out[f"{prefix}_{suffix}_qc"] = pedicle_qc
    return out
