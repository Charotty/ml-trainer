"""Skin-to-skin abdominal diameters and anterior wall thickness at L3–L4.

Excel measures these on the L3–L4 disc, not on a kidney slice. Width is the
left-right extent (patient X) and depth is the anterior-posterior extent
(patient Y). Wall thickness is the midline distance from anterior skin to the
abdominal cavity; rectus abdominis is only a fallback.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.features.ct_anatomy.qc import (
    QC_APPROX,
    QC_MISSING,
    QC_OK,
    QC_OUT_OF_FOV,
    QC_OUT_OF_RANGE,
    SHARED_RANGES,
)
from src.features.ct_anatomy.volume import PATIENT_Z, AnatomyVolume, voxel_axis_for
from src.features.ct_geometry import voxels_to_patient_mm

# Same limits as the clinical sanitizer. A cropped HU blob (about 120 mm on a
# whole-body scan) must not become body_width_mm / body_depth_mm.
WIDTH_RANGE = SHARED_RANGES["abd_width_l3l4_mm"]
DEPTH_RANGE = SHARED_RANGES["abd_depth_l3l4_mm"]
_ACCEPTED_BODY_QC = {QC_OK, QC_APPROX}


def _extent(points: np.ndarray, axis: int) -> Optional[float]:
    if len(points) == 0:
        return None
    return float(points[:, axis].max() - points[:, axis].min())


def _wall_thickness_mm(body: np.ndarray, inner: np.ndarray) -> Optional[float]:
    """Midline distance from the anterior skin to ``inner`` (cavity or muscle)."""
    if len(body) == 0 or len(inner) == 0:
        return None
    x_mid = float(np.median(inner[:, 0]))
    x_tol = max(3.0, 0.05 * float(body[:, 0].max() - body[:, 0].min()))
    body_col = body[np.abs(body[:, 0] - x_mid) <= x_tol]
    inner_col = inner[np.abs(inner[:, 0] - x_mid) <= x_tol]
    if len(body_col) == 0 or len(inner_col) == 0:
        return None
    skin_y = float(body_col[:, 1].min())
    # Cavity/muscle voxels at or posterior to the skin. Anterior is −Y.
    posterior = inner_col[inner_col[:, 1] >= skin_y - 1.0]
    if len(posterior) == 0:
        return None
    inner_y = float(posterior[:, 1].min())
    thickness = inner_y - skin_y
    if thickness < 0:
        return None
    return float(thickness)


def measure_abdomen(volume: AnatomyVolume, z_mm: Optional[float]) -> Dict[str, object]:
    keys = ("abd_width_l3l4_mm", "abd_depth_l3l4_mm", "abd_wall_thickness_mm")
    if z_mm is None:
        return {key: None for key in keys} | {f"{key}_qc": QC_MISSING for key in keys}

    body = volume.slice_points("body_trunc", z_mm)
    source_qc = "ok"
    if len(body) == 0 and volume.hu is not None:
        body = _hu_body_points(volume, z_mm)
        source_qc = QC_APPROX
    if len(body) == 0:
        return {key: None for key in keys} | {f"{key}_qc": QC_MISSING for key in keys}

    width = _extent(body, 0)
    depth = _extent(body, 1)
    cavity = volume.slice_points("abdominal_cavity", z_mm)
    wall_qc = source_qc
    wall = _wall_thickness_mm(body, cavity)
    if wall is None:
        rectus_l = volume.slice_points("rectus_abdominis_left", z_mm)
        rectus_r = volume.slice_points("rectus_abdominis_right", z_mm)
        rectus = np.concatenate([rectus_l, rectus_r], axis=0) if len(rectus_l) or len(rectus_r) else np.zeros((0, 3))
        wall = _wall_thickness_mm(body, rectus)
        if wall is not None:
            wall_qc = QC_APPROX

    edge = volume.touches_border("body_trunc")
    if edge and source_qc == "ok":
        source_qc = QC_OUT_OF_FOV
        if wall_qc == "ok":
            wall_qc = QC_OUT_OF_FOV

    return {
        "abd_width_l3l4_mm": width,
        "abd_width_l3l4_mm_qc": source_qc if width is not None else QC_MISSING,
        "abd_depth_l3l4_mm": depth,
        "abd_depth_l3l4_mm_qc": source_qc if depth is not None else QC_MISSING,
        "abd_wall_thickness_mm": wall,
        "abd_wall_thickness_mm_qc": wall_qc if wall is not None else QC_MISSING,
    }


def _finite(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _in_range(value: float, bounds: tuple) -> bool:
    return bounds[0] <= value <= bounds[1]


def kidney_level_z_from_row(row: Dict[str, object]) -> Optional[float]:
    """Mean cranio-caudal center of whichever kidneys are present."""
    zs = []
    for key in ("kidney_left_center_z", "kidney_right_center_z"):
        number = _finite(row.get(key))
        if number is not None:
            zs.append(number)
    if not zs:
        return None
    return float(np.mean(zs))


def kidney_level_z_from_volume(volume: AnatomyVolume) -> Optional[float]:
    zs = []
    for name in ("kidney_left", "kidney_right"):
        centroid = volume.centroid(name)
        if centroid is not None:
            zs.append(float(centroid[2]))
    if not zs:
        return None
    return float(np.mean(zs))


def widest_body_z(volume: AnatomyVolume) -> Optional[float]:
    """Axial level where the skin mask covers the most voxels."""
    mask = volume.mask("body_trunc")
    if mask is None:
        return None
    axis, _sign = voxel_axis_for(volume.affine, PATIENT_Z)
    other = tuple(a for a in range(3) if a != axis)
    counts = np.asarray(mask).sum(axis=other)
    if int(counts.max()) == 0:
        return None
    k = int(np.argmax(counts))
    ijk = np.zeros((1, 3), dtype=float)
    ijk[0, axis] = k
    return float(voxels_to_patient_mm(volume.affine, ijk)[0, 2])


def clinical_body_size_at(
    volume: AnatomyVolume,
    z_mm: float,
    *,
    source: str,
) -> Dict[str, object]:
    """Skin-to-skin width, depth, and width×depth at one axial level.

    Width is patient X and depth is patient Y, matching the Excel columns.
    The area is the bounding rectangle, not the voxel count inside the skin.
    """
    body = volume.slice_points("body_trunc", float(z_mm))
    if len(body) == 0:
        return {}
    width = float(body[:, 0].max() - body[:, 0].min())
    depth = float(body[:, 1].max() - body[:, 1].min())
    if volume.touches_border("body_trunc"):
        qc = QC_OUT_OF_FOV
    elif not _in_range(width, WIDTH_RANGE) or not _in_range(depth, DEPTH_RANGE):
        qc = QC_OUT_OF_RANGE
    elif source == "body_mask_l3l4":
        qc = QC_OK
    else:
        qc = QC_APPROX
    return {
        "body_mask_width_mm": width,
        "body_mask_depth_mm": depth,
        "body_mask_z_mm": float(z_mm),
        "body_mask_qc": qc,
        "body_mask_source": source,
    }


def mask_fields_from_abdomen(abdomen: Dict[str, object]) -> Dict[str, object]:
    """Copy an in-range L3–L4 skin measurement onto the model body-size keys."""
    width = _finite(abdomen.get("abd_width_l3l4_mm"))
    depth = _finite(abdomen.get("abd_depth_l3l4_mm"))
    qc = abdomen.get("abd_width_l3l4_mm_qc")
    if width is None or depth is None or qc not in _ACCEPTED_BODY_QC:
        return {}
    if not _in_range(width, WIDTH_RANGE) or not _in_range(depth, DEPTH_RANGE):
        return {}
    return {
        "body_mask_width_mm": width,
        "body_mask_depth_mm": depth,
        "body_mask_qc": qc,
        "body_mask_source": "body_mask_l3l4",
    }


def body_mask_fields(volume: AnatomyVolume, row: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Prefer L3–L4, then the kidney level, then the widest skin slice."""
    if row:
        from_l3 = mask_fields_from_abdomen(row)
        if from_l3:
            return from_l3
    z = kidney_level_z_from_row(row or {})
    source = "body_mask_kidney_z"
    if z is None:
        z = kidney_level_z_from_volume(volume)
    if z is None:
        z = widest_body_z(volume)
        source = "body_mask_widest_slice"
    if z is None or volume.mask("body_trunc") is None:
        return {}
    return clinical_body_size_at(volume, z, source=source)


def assign_clinical_body_size(row: Dict[str, object]) -> Dict[str, object]:
    """Write model columns from a skin mask that survived the clinical range.

    The HU extractor can leave a ~120 mm blob in ``body_width_mm``. This
    replaces that blob when a body-mask measurement is in range. An absent
    kidney does not block the write: one kidney, or none, still has a torso.
    """
    width = _finite(row.get("body_mask_width_mm"))
    depth = _finite(row.get("body_mask_depth_mm"))
    qc = row.get("body_mask_qc")
    if width is None or depth is None or qc not in _ACCEPTED_BODY_QC:
        return {}
    if not _in_range(width, WIDTH_RANGE) or not _in_range(depth, DEPTH_RANGE):
        return {}
    return {
        "body_width_mm": width,
        "body_depth_mm": depth,
        "body_area_mm2": float(width * depth),
        "body_size_source": row.get("body_mask_source") or "body_mask",
    }


def _hu_body_points(volume: AnatomyVolume, z_mm: float) -> np.ndarray:
    """Fallback body envelope from soft-tissue HU when the body task is missing."""
    if volume.hu is None:
        return np.zeros((0, 3), dtype=float)
    axis_z = volume.axial_index_for_z(z_mm)
    axis, _sign = voxel_axis_for(volume.affine, PATIENT_Z)
    slab = np.take(volume.hu, axis_z, axis=axis)
    tissue = (slab >= -300) & (slab <= 300)
    coords = np.argwhere(tissue)
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=float)
    remaining = [a for a in range(3) if a != axis]
    ijk = np.zeros((len(coords), 3), dtype=float)
    ijk[:, remaining[0]] = coords[:, 0]
    ijk[:, remaining[1]] = coords[:, 1]
    ijk[:, axis] = axis_z
    return voxels_to_patient_mm(volume.affine, ijk)
