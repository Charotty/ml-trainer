"""Distances from kidney poles to rib, iliac crest, diaphragm and spine.

Diaphragm has no TotalSegmentator class. The right dome is the superior
surface of the liver and the left dome is the inferior surface of the left
lower lobe. Those two distances are marked ``approx``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from src.features.ct_anatomy.qc import QC_APPROX, QC_MISSING, QC_OUT_OF_FOV
from src.features.ct_anatomy.volume import AnatomyVolume


def _extreme_centroid(points: np.ndarray, *, superior: bool) -> Optional[np.ndarray]:
    if len(points) == 0:
        return None
    z = points[:, 2]
    limit = z.max() if superior else z.min()
    # Keep a single voxel layer. ±1 mm pulled in the neighbour and biased the gap.
    band = points[np.abs(z - limit) <= 0.51]
    if len(band) == 0:
        band = points
    return band.mean(axis=0)


def _euclidean(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def _min_xy_distance(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) == 0 or len(b) == 0:
        return None
    tree = cKDTree(b[:, :2])
    dist, _idx = tree.query(a[:, :2], k=1)
    return float(np.min(dist))


def measure_kidney_distances(volume: AnatomyVolume) -> Dict[str, object]:
    out: Dict[str, object] = {}
    spine_names = [f"vertebrae_L{i}" for i in range(1, 6)] + ["vertebrae_T12"]
    for side in ("left", "right"):
        kidney = volume.points(f"kidney_{side}")
        prefix = f"kidney_{side}"
        upper = _extreme_centroid(kidney, superior=True)
        lower = _extreme_centroid(kidney, superior=False)

        rib = volume.points(f"rib_{side}_11")
        rib_qc = "ok"
        if len(rib) == 0:
            rib = volume.points(f"rib_{side}_12")
            rib_qc = QC_APPROX
        rib_lower = _extreme_centroid(rib, superior=False)
        rib_dist = _euclidean(upper, rib_lower)

        hip = volume.points(f"hip_{side}")
        crest = _extreme_centroid(hip, superior=True)
        iliac_dist = _euclidean(lower, crest)

        if side == "right":
            dome = _extreme_centroid(volume.points("liver"), superior=True)
        else:
            dome = _extreme_centroid(volume.points("lung_lower_lobe_left"), superior=False)
        diaphragm = _euclidean(upper, dome)

        z_mid = float(kidney[:, 2].mean()) if len(kidney) else None
        medial = None
        if z_mid is not None:
            kidney_slice = volume.slice_points(f"kidney_{side}", z_mid)
            spine_parts = [volume.slice_points(name, z_mid) for name in spine_names]
            spine_parts = [part for part in spine_parts if len(part)]
            if spine_parts and len(kidney_slice):
                medial = _min_xy_distance(kidney_slice, np.concatenate(spine_parts, axis=0))

        values = {
            "upper_pole_to_rib11_mm": (rib_dist, rib_qc if rib_dist is not None else QC_MISSING),
            "lower_pole_to_iliac_crest_mm": (iliac_dist, "ok" if iliac_dist is not None else QC_MISSING),
            "upper_pole_to_diaphragm_mm": (diaphragm, QC_APPROX if diaphragm is not None else QC_MISSING),
            "medial_to_spine_mm": (medial, "ok" if medial is not None else QC_MISSING),
        }
        clipped = volume.touches_border(f"kidney_{side}")
        for suffix, (value, qc) in values.items():
            key = f"{prefix}_{suffix}"
            out[key] = value
            if value is not None and clipped and qc == "ok":
                qc = QC_OUT_OF_FOV
            out[f"{key}_qc"] = qc
    return out
