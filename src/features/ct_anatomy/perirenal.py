"""Perinephric fat thickness, density, stranding and Mayo Adhesive Probability score.

Thickness is the fat run (HU −190…−30) along a ray leaving the kidney surface
at the hilum slice: dorsal (+Y), ventral (−Y) and lateral (away from midline).

MAP score follows Davidiuk et al., J Urol 2014 (PMID 25192968): posterior fat
<1 cm / 1.0–1.9 cm / ≥2 cm scores 0 / 1 / 2, plus stranding 0 / 2 / 3.
The stranding bins below are a HU stand-in for the visual grade and must be
recalibrated against the Excel column before they are treated as the score.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.features.ct_anatomy.qc import QC_APPROX, QC_MISSING
from src.features.ct_anatomy.volume import PATIENT_X, PATIENT_Y, AnatomyVolume

FAT_HU = (-190.0, -30.0)
STRANDING_HU = (-100.0, 0.0)
# Provisional: fraction of the 5–10 mm shell inside the stranding window.
STRANDING_MILD = 0.05
STRANDING_SEVERE = 0.20


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


def _ray_thickness_mm(
    volume: AnatomyVolume,
    surface: np.ndarray,
    direction: np.ndarray,
    *,
    step_mm: float = 1.0,
    max_mm: float = 60.0,
) -> Optional[float]:
    if volume.hu is None or len(surface) == 0:
        return None
    direction = direction / np.linalg.norm(direction)
    projections = surface @ direction
    start = surface[int(np.argmax(projections))]
    length = 0.0
    seen_fat = False
    n_steps = int(max_mm / step_mm)
    for i in range(1, n_steps + 1):
        hu = volume.sample_hu(start + direction * (i * step_mm))
        if not np.isfinite(hu):
            break
        if FAT_HU[0] <= hu <= FAT_HU[1]:
            seen_fat = True
            length = i * step_mm
            continue
        break
    if not seen_fat:
        return 0.0
    return float(length)


def _shell_stats(volume: AnatomyVolume, kidney: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if volume.hu is None or not np.any(kidney):
        return None, None
    outside = distance_transform_edt(~kidney, sampling=volume.zooms)
    shell = (outside >= 5.0) & (outside <= 10.0)
    if not np.any(shell):
        return None, None
    hu = volume.hu[shell]
    finite = hu[np.isfinite(hu)]
    if len(finite) == 0:
        return None, None
    fat = finite[(finite >= FAT_HU[0]) & (finite <= FAT_HU[1])]
    density = float(fat.mean()) if len(fat) else None
    stranding = float(np.mean((finite >= STRANDING_HU[0]) & (finite <= STRANDING_HU[1])))
    return density, stranding


def measure_perirenal(volume: AnatomyVolume) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for side in ("left", "right"):
        prefix = f"kidney_{side}"
        kidney_mask = volume.mask(f"kidney_{side}")
        if kidney_mask is None or volume.hu is None:
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

        points = volume.points(f"kidney_{side}")
        z_mid = float(points[:, 2].mean())
        surface = volume.slice_points(f"kidney_{side}", z_mid)
        thicknesses = {}
        for kind in ("dorsal", "ventral", "lateral"):
            thicknesses[kind] = _ray_thickness_mm(volume, surface, _direction(side, kind))
        density, stranding = _shell_stats(volume, kidney_mask)
        score = map_points(thicknesses["dorsal"], stranding)
        # Stranding bins are not the radiologist grade, so MAP stays approximate.
        values = {
            "perirenal_dorsal_mm": thicknesses["dorsal"],
            "perirenal_ventral_mm": thicknesses["ventral"],
            "perirenal_lateral_mm": thicknesses["lateral"],
            "perirenal_hu": density,
            "perirenal_stranding": stranding,
            "map_score": score,
        }
        for suffix, value in values.items():
            key = f"{prefix}_{suffix}"
            out[key] = value
            out[f"{key}_qc"] = QC_APPROX if value is not None else QC_MISSING
    return out
