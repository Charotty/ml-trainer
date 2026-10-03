"""Perinephric fat thickness, density, stranding and Mayo Adhesive Probability score.

Dorsal and ventral thickness are fat runs (HU −190…−30) leaving the mid-kidney
surface: dorsal toward the back, ventral toward bowel or peritoneum. The first
few millimetres of non-fat are a partial-volume rim and do not end the run.

Lateral thickness is the gap from the lateral parenchyma to the liver (right)
or the spleen (left), which is where the manual caliper stops. A fat ray is
only the fallback when that organ mask is missing.

Density is the mean HU of the posterior fat pad on the hilum slice, including
soft tissue inside the pad (about −200…+50). It is not the mean of a fat-only
window around the whole kidney.

MAP score follows Davidiuk et al., J Urol 2014 (PMID 25192968): posterior fat
<1 cm / 1.0–1.9 cm / ≥2 cm scores 0 / 1 / 2, plus stranding 0 / 2 / 3.
The stranding bins below are a HU stand-in for the visual grade and must be
recalibrated against the Excel column before they are treated as the score.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.features.ct_anatomy.qc import QC_APPROX, QC_MISSING
from src.features.ct_anatomy.volume import PATIENT_X, PATIENT_Y, PATIENT_Z, AnatomyVolume, voxel_axis_for
from src.features.ct_geometry import voxels_to_patient_mm

FAT_HU = (-190.0, -30.0)
DENSITY_HU = (-200.0, 50.0)
STRANDING_HU = (-100.0, 0.0)
# Provisional: fraction of the posterior pad inside the stranding window.
STRANDING_MILD = 0.05
STRANDING_SEVERE = 0.20
_RIM_MM = 3.0
_RAYS = 7
_EXCLUDE = (
    "liver",
    "spleen",
    "vertebrae_body",
    "aorta",
    "inferior_vena_cava",
    "psoas_major_left",
    "psoas_major_right",
    "iliopsoas_left",
    "iliopsoas_right",
    "kidney_left",
    "kidney_right",
    "vertebrae_T11",
    "vertebrae_T12",
    "vertebrae_L1",
    "vertebrae_L2",
    "vertebrae_L3",
    "vertebrae_L4",
    "vertebrae_L5",
    "vertebrae_S1",
)


def map_points(dorsal_thickness_mm: Optional[float], stranding_fraction: Optional[float]) -> Optional[float]:
    if dorsal_thickness_mm is None or stranding_fraction is None:
        return None
    if dorsal_thickness_mm < 10.0:
        thickness_points = 0
    elif dorsal_thickness_mm < 20.0:
        thickness_points = 1
    else:
        thickness_points = 2
    if stranding_fraction < STRANDING_MILD:
        strand_points = 0
    elif stranding_fraction < STRANDING_SEVERE:
        strand_points = 2
    else:
        strand_points = 3
    return float(thickness_points + strand_points)


def _direction(side: str, kind: str) -> np.ndarray:
    if kind == "dorsal":
        return PATIENT_Y.copy()
    if kind == "ventral":
        return -PATIENT_Y
    # Lateral is away from the midline: left kidney toward +X, right toward −X.
    return PATIENT_X.copy() if side == "left" else -PATIENT_X


def _face_samples(surface: np.ndarray, direction: np.ndarray, n: int = _RAYS) -> np.ndarray:
    """Outer-face points. Interior voxels would start the ray inside the kidney."""
    if len(surface) == 0:
        return surface
    direction = direction / np.linalg.norm(direction)
    projections = surface @ direction
    face = surface[projections >= float(projections.max()) - 1.5]
    if len(face) <= n:
        return face
    axis = int(np.argmin(np.abs(direction)))
    order = np.argsort(face[:, axis])
    pick = np.linspace(0, len(order) - 1, n).astype(int)
    return face[order[pick]]


def _ray_from(
    volume: AnatomyVolume,
    start: np.ndarray,
    direction: np.ndarray,
    *,
    step_mm: float = 1.0,
    max_mm: float = 60.0,
) -> Optional[float]:
    """Length of the fat run beyond a short non-fat rim. 0 when no fat is met."""
    if volume.hu is None:
        return None
    direction = direction / np.linalg.norm(direction)
    fat_start = None
    fat_end = None
    n_steps = int(max_mm / step_mm)
    for i in range(1, n_steps + 1):
        dist = i * step_mm
        hu = volume.sample_hu(start + direction * dist)
        if not np.isfinite(hu):
            break
        if FAT_HU[0] <= hu <= FAT_HU[1]:
            if fat_start is None:
                fat_start = dist
            fat_end = dist
            continue
        if fat_start is None and dist <= _RIM_MM:
            continue
        break
    if fat_start is None or fat_end is None:
        return 0.0
    return float(fat_end - fat_start + step_mm)


def _median_ray(
    volume: AnatomyVolume,
    surface: np.ndarray,
    direction: np.ndarray,
) -> Optional[float]:
    samples = _face_samples(surface, direction)
    if len(samples) == 0:
        return None
    lengths = [_ray_from(volume, origin, direction) for origin in samples]
    finite = [value for value in lengths if value is not None]
    if not finite:
        return None
    return float(np.median(finite))


def _lateral_to_organ(kidney: np.ndarray, organ: np.ndarray, side: str) -> Optional[float]:
    """Gap from the lateral parenchyma to liver (right) or spleen (left)."""
    if len(kidney) == 0 or len(organ) == 0:
        return None
    z_mid = float(kidney[:, 2].mean())
    organ_band = organ[np.abs(organ[:, 2] - z_mid) <= 20.0]
    if len(organ_band) == 0:
        return None
    if side == "right":
        edge = float(kidney[:, 0].min())
        face = kidney[kidney[:, 0] <= edge + 1.0]
    else:
        edge = float(kidney[:, 0].max())
        face = kidney[kidney[:, 0] >= edge - 1.0]
    if len(face) == 0:
        return None
    dist, _idx = cKDTree(organ_band).query(face, k=1)
    return float(np.median(dist))


def _slice_hu(volume: AnatomyVolume, z_mm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(xyz_mm, hu, exclusion)`` for the axial slice nearest ``z_mm``."""
    empty = np.zeros((0, 3), dtype=float)
    if volume.hu is None:
        return empty, np.zeros(0, dtype=float), np.zeros(0, dtype=bool)
    axis, _sign = voxel_axis_for(volume.affine, PATIENT_Z)
    k = volume.axial_index_for_z(z_mm)
    slab = np.take(volume.hu, k, axis=axis)
    exclusion = np.zeros(slab.shape, dtype=bool)
    for name in _EXCLUDE:
        mask = volume.mask(name)
        if mask is None or mask.shape != volume.hu.shape:
            continue
        exclusion |= np.take(mask, k, axis=axis)
    coords = np.argwhere(np.isfinite(slab))
    if len(coords) == 0:
        return empty, np.zeros(0, dtype=float), np.zeros(0, dtype=bool)
    remaining = [a for a in range(3) if a != axis]
    ijk = np.zeros((len(coords), 3), dtype=float)
    ijk[:, remaining[0]] = coords[:, 0]
    ijk[:, remaining[1]] = coords[:, 1]
    ijk[:, axis] = k
    xyz = voxels_to_patient_mm(volume.affine, ijk)
    hu = slab[coords[:, 0], coords[:, 1]].astype(float)
    excluded = exclusion[coords[:, 0], coords[:, 1]]
    return xyz, hu, excluded


def _posterior_stats(volume: AnatomyVolume, side: str) -> Tuple[Optional[float], Optional[float]]:
    """Mean HU and stranding fraction of the posterior pad on the hilum slice."""
    kidney = volume.points(f"kidney_{side}")
    if volume.hu is None or len(kidney) == 0:
        return None, None
    z_mid = float(kidney[:, 2].mean())
    kidney_slice = volume.slice_points(f"kidney_{side}", z_mid)
    if len(kidney_slice) == 0:
        return None, None
    xyz, hu, excluded = _slice_hu(volume, z_mid)
    if len(xyz) == 0:
        return None, None
    y_back = float(kidney_slice[:, 1].max())
    x0, x1 = np.quantile(kidney_slice[:, 0], [0.2, 0.8])
    pad = (
        (xyz[:, 1] > y_back + 1.0)
        & (xyz[:, 1] <= y_back + 25.0)
        & (xyz[:, 0] >= x0)
        & (xyz[:, 0] <= x1)
        & ~excluded
    )
    values = hu[pad]
    values = values[np.isfinite(values) & (values >= DENSITY_HU[0]) & (values <= DENSITY_HU[1])]
    if len(values) == 0:
        return None, None
    density = float(values.mean())
    stranding = float(np.mean((values >= STRANDING_HU[0]) & (values <= STRANDING_HU[1])))
    return density, stranding


def measure_perirenal(volume: AnatomyVolume) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for side in ("left", "right"):
        prefix = f"kidney_{side}"
        kidney_name = f"kidney_{side}"
        if volume.mask(kidney_name) is None or volume.hu is None:
            for suffix in (
                "perirenal_dorsal_mm",
                "perirenal_ventral_mm",
                "perirenal_lateral_mm",
                "perirenal_hu",
                "perirenal_stranding",
                "map_score",
            ):
                out[f"{prefix}_{suffix}"] = None
                out[f"{prefix}_{suffix}_qc"] = QC_MISSING
            continue

        points = volume.points(kidney_name)
        z_mid = float(points[:, 2].mean())
        surface = volume.slice_points(kidney_name, z_mid)
        dorsal = _median_ray(volume, surface, _direction(side, "dorsal"))
        ventral = _median_ray(volume, surface, _direction(side, "ventral"))

        organ_name = "liver" if side == "right" else "spleen"
        lateral = _lateral_to_organ(points, volume.points(organ_name), side)
        lateral_qc = "ok" if lateral is not None else QC_APPROX
        if lateral is None:
            lateral = _median_ray(volume, surface, _direction(side, "lateral"))
            lateral_qc = QC_APPROX if lateral is not None else QC_MISSING

        density, stranding = _posterior_stats(volume, side)
        score = map_points(dorsal, stranding)
        values = {
            "perirenal_dorsal_mm": (dorsal, "ok" if dorsal is not None else QC_MISSING),
            "perirenal_ventral_mm": (ventral, "ok" if ventral is not None else QC_MISSING),
            "perirenal_lateral_mm": (lateral, lateral_qc),
            "perirenal_hu": (density, QC_APPROX if density is not None else QC_MISSING),
            "perirenal_stranding": (stranding, QC_APPROX if stranding is not None else QC_MISSING),
            "map_score": (score, QC_APPROX if score is not None else QC_MISSING),
        }
        for suffix, (value, qc) in values.items():
            key = f"{prefix}_{suffix}"
            out[key] = value
            out[f"{key}_qc"] = qc
    return out
