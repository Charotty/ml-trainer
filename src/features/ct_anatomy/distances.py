"""Distances from kidney poles to rib, iliac crest, diaphragm and spine.

Manual protocol (supine CT, vertebral coordinates):

* Rib 11 and diaphragm are cranio-caudal gaps along Z. The rib point is the
  medial end, where the rib leaves the vertebral body, not the anterior tip.
* The iliac gap is signed: negative when the crest is cranial to the lower pole.
* Medial-to-spine is the horizontal gap on the hilum slice from parenchyma to
  the vertebral body. Transverse processes are not the target.
* Diaphragm has no TotalSegmentator class. The gap is the column above the
  upper pole up to the liver (right) or the left lower lobe (left), so that
  distance stays ``approx``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.features.ct_anatomy.qc import QC_APPROX, QC_MISSING, QC_OUT_OF_FOV
from src.features.ct_anatomy.volume import AnatomyVolume

_COLUMN_RADIUS_MM = 18.0
_SPINE_NAMES = [f"vertebrae_L{i}" for i in range(1, 6)] + ["vertebrae_T12", "vertebrae_T11"]


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


def _rib_exit(rib: np.ndarray, vertebra: np.ndarray) -> Optional[np.ndarray]:
    """Point where the rib meets the vertebra, not the anterior rib tip."""
    if len(rib) == 0:
        return None
    if len(vertebra):
        dist, _idx = cKDTree(vertebra).query(rib, k=1)
        cutoff = max(float(np.quantile(dist, 0.15)), 3.0)
        band = rib[dist <= cutoff]
        if len(band) == 0:
            band = rib[dist <= float(dist.min()) + 2.0]
        return band.mean(axis=0)
    medial = np.abs(rib[:, 0])
    cutoff = float(np.quantile(medial, 0.15))
    band = rib[medial <= cutoff + 1.0]
    if len(band) == 0:
        band = rib
    return band.mean(axis=0)


def _column_surface_z(points: np.ndarray, pole: np.ndarray) -> Optional[float]:
    """Inferior surface of ``points`` in the column above ``pole``.

    Falls back to the superior surface when the structure sits entirely below
    the pole, so the signed gap can be negative.
    """
    if len(points) == 0:
        return None
    offset = points[:, :2] - pole[:2]
    near = points[np.linalg.norm(offset, axis=1) <= _COLUMN_RADIUS_MM]
    if len(near) == 0:
        return None
    above = near[near[:, 2] >= pole[2] - 1.0]
    if len(above) == 0:
        return float(near[:, 2].max())
    return float(above[:, 2].min())


def _vertebral_body_slice(volume: AnatomyVolume, z_mm: float) -> Tuple[np.ndarray, str]:
    """Vertebral body on the hilum slice. Named vertebrae are only a fallback."""
    body = volume.slice_points("vertebrae_body", z_mm)
    if len(body):
        return body, "ok"
    parts = [volume.slice_points(name, z_mm) for name in _SPINE_NAMES]
    parts = [part for part in parts if len(part)]
    if not parts:
        return np.zeros((0, 3), dtype=float), QC_MISSING
    spine = np.concatenate(parts, axis=0)
    # The body is the anterior part. Transverse processes sit lateral to it.
    cut = float(np.quantile(spine[:, 1], 0.6))
    anterior = spine[spine[:, 1] <= cut]
    if len(anterior) < 5:
        anterior = spine
    return anterior, QC_APPROX


def _medial_x_gap(kidney_slice: np.ndarray, body: np.ndarray, side: str) -> Optional[float]:
    """Horizontal parenchyma-to-body gap. Positive when the kidney is lateral."""
    if len(kidney_slice) == 0 or len(body) == 0:
        return None
    if side == "right":
        return float(body[:, 0].min() - kidney_slice[:, 0].max())
    return float(kidney_slice[:, 0].min() - body[:, 0].max())


def measure_kidney_distances(volume: AnatomyVolume) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for side in ("left", "right"):
        kidney = volume.points(f"kidney_{side}")
        prefix = f"kidney_{side}"
        upper = _extreme_centroid(kidney, superior=True)
        lower = _extreme_centroid(kidney, superior=False)

        rib = volume.points(f"rib_{side}_11")
        rib_level = "T11"
        rib_qc = "ok"
        if len(rib) == 0:
            rib = volume.points(f"rib_{side}_12")
            rib_level = "T12"
            rib_qc = QC_APPROX
        rib_exit = _rib_exit(rib, volume.points(f"vertebrae_{rib_level}"))
        rib_dist = None if upper is None or rib_exit is None else float(rib_exit[2] - upper[2])

        hip = volume.points(f"hip_{side}")
        iliac_dist = None
        if lower is not None and len(hip):
            # Positive when the lower pole is above the crest.
            iliac_dist = float(lower[2] - float(hip[:, 2].max()))

        dome_name = "liver" if side == "right" else "lung_lower_lobe_left"
        diaphragm = None
        if upper is not None:
            surface_z = _column_surface_z(volume.points(dome_name), upper)
            if surface_z is not None:
                diaphragm = float(surface_z - upper[2])

        z_mid = float(kidney[:, 2].mean()) if len(kidney) else None
        medial = None
        medial_qc = QC_MISSING
        if z_mid is not None:
            kidney_slice = volume.slice_points(f"kidney_{side}", z_mid)
            body, medial_qc = _vertebral_body_slice(volume, z_mid)
            medial = _medial_x_gap(kidney_slice, body, side)
            if medial is None:
                medial_qc = QC_MISSING

        values = {
            "upper_pole_to_rib11_mm": (rib_dist, rib_qc if rib_dist is not None else QC_MISSING),
            "lower_pole_to_iliac_crest_mm": (iliac_dist, "ok" if iliac_dist is not None else QC_MISSING),
            "upper_pole_to_diaphragm_mm": (diaphragm, QC_APPROX if diaphragm is not None else QC_MISSING),
            "medial_to_spine_mm": (medial, medial_qc if medial is not None else QC_MISSING),
        }
        clipped = volume.touches_border(f"kidney_{side}")
        for suffix, (value, qc) in values.items():
            key = f"{prefix}_{suffix}"
            out[key] = value
            if value is not None and clipped and qc == "ok":
                qc = QC_OUT_OF_FOV
            out[f"{key}_qc"] = qc
    return out
