"""In-memory CT grid: boolean masks and HU on one patient-mm affine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from src.features.ct_geometry import voxels_to_patient_mm

PATIENT_X = np.array([1.0, 0.0, 0.0])  # LPS: patient left
PATIENT_Y = np.array([0.0, 1.0, 0.0])  # posterior
PATIENT_Z = np.array([0.0, 0.0, 1.0])  # superior


def voxel_axis_for(affine: np.ndarray, patient_dir: np.ndarray) -> Tuple[int, int]:
    """Return ``(axis, sign)`` of the voxel axis most aligned with ``patient_dir``.

    ``sign`` is +1 when increasing the index moves along ``patient_dir``.
    """
    rot = np.asarray(affine, dtype=float)[:3, :3]
    dots = rot.T @ np.asarray(patient_dir, dtype=float)
    axis = int(np.argmax(np.abs(dots)))
    sign = 1 if dots[axis] >= 0 else -1
    return axis, sign


def mask_points_mm(
    mask: np.ndarray,
    affine: np.ndarray,
    *,
    limit: int = 200_000,
) -> np.ndarray:
    """Nx3 patient-mm coordinates of True voxels. Subsamples when huge."""
    idx = np.argwhere(np.asarray(mask, dtype=bool))
    if len(idx) == 0:
        return np.zeros((0, 3), dtype=float)
    if len(idx) > limit:
        step = int(np.ceil(len(idx) / limit))
        idx = idx[::step]
    return voxels_to_patient_mm(affine, idx)


def world_to_voxel(affine: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    hom = np.concatenate([np.asarray(xyz, dtype=float), np.ones(1)])
    return (np.linalg.inv(np.asarray(affine, dtype=float)) @ hom)[:3]


@dataclass
class AnatomyVolume:
    """Masks and optional HU sharing one affine. Axes are voxel indices."""

    affine: np.ndarray
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    hu: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.affine = np.asarray(self.affine, dtype=float)
        ref = None
        if self.masks:
            ref = next(iter(self.masks.values()))
        elif self.hu is not None:
            ref = self.hu
        if ref is None:
            raise ValueError("AnatomyVolume needs at least one mask or a HU array")
        self.shape: Tuple[int, int, int] = tuple(int(v) for v in np.shape(ref))
        self.zooms: Tuple[float, float, float] = tuple(
            float(np.linalg.norm(self.affine[:3, i])) for i in range(3)
        )

    def mask(self, name: str) -> Optional[np.ndarray]:
        value = self.masks.get(name)
        if value is None or not np.any(value):
            return None
        return np.asarray(value, dtype=bool)

    def points(self, name: str) -> np.ndarray:
        mask = self.mask(name)
        if mask is None:
            return np.zeros((0, 3), dtype=float)
        return mask_points_mm(mask, self.affine)

    def centroid(self, name: str) -> Optional[np.ndarray]:
        pts = self.points(name)
        if len(pts) == 0:
            return None
        return pts.mean(axis=0)

    def axial_index_for_z(self, z_mm: float) -> int:
        axis, _sign = voxel_axis_for(self.affine, PATIENT_Z)
        n = self.shape[axis]
        ijk = np.zeros((n, 3), dtype=float)
        ijk[:, axis] = np.arange(n)
        for other in range(3):
            if other != axis:
                ijk[:, other] = self.shape[other] / 2.0
        zs = voxels_to_patient_mm(self.affine, ijk)[:, 2]
        return int(np.argmin(np.abs(zs - float(z_mm))))

    def slice_points(self, name: str, z_mm: float) -> np.ndarray:
        """Patient-mm points of ``name`` on the axial slice nearest ``z_mm``."""
        mask = self.mask(name)
        if mask is None:
            return np.zeros((0, 3), dtype=float)
        axis, _sign = voxel_axis_for(self.affine, PATIENT_Z)
        k = self.axial_index_for_z(z_mm)
        slab = np.take(mask, k, axis=axis)
        coords = np.argwhere(slab)
        if len(coords) == 0:
            return np.zeros((0, 3), dtype=float)
        remaining = [a for a in range(3) if a != axis]
        ijk = np.zeros((len(coords), 3), dtype=float)
        ijk[:, remaining[0]] = coords[:, 0]
        ijk[:, remaining[1]] = coords[:, 1]
        ijk[:, axis] = k
        return voxels_to_patient_mm(self.affine, ijk)

    def sample_hu(self, xyz: np.ndarray) -> float:
        if self.hu is None:
            return float("nan")
        ijk = world_to_voxel(self.affine, xyz)
        rounded = np.rint(ijk).astype(int)
        if np.any(rounded < 0) or np.any(rounded >= np.array(self.shape)):
            return float("nan")
        return float(self.hu[tuple(rounded)])

    def touches_border(self, name: str, margin: int = 2) -> bool:
        mask = self.mask(name)
        if mask is None:
            return False
        idx = np.argwhere(mask)
        if len(idx) == 0:
            return False
        shape = np.array(mask.shape)
        return bool(np.any(idx.min(axis=0) <= margin) or np.any(idx.max(axis=0) >= shape - 1 - margin))
