"""Psoas cross-section at the kidney midpoint.

The manual thickness and area are taken on the same axial level as the other
renal measurements (about the middle of that kidney), not on the L3 centroid.
Psoas thickens caudally, so an L3 slice runs high.

Prefers ``psoas_major`` from the ``abdominal_muscles`` task. Falls back to
``iliopsoas`` from the ``total`` task, which includes iliacus and is marked
approximate.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.features.ct_anatomy.qc import QC_APPROX, QC_MISSING, QC_OUT_OF_FOV
from src.features.ct_anatomy.volume import PATIENT_Z, AnatomyVolume, voxel_axis_for


def _inplane_area_mm2(volume: AnatomyVolume) -> float:
    axis, _sign = voxel_axis_for(volume.affine, PATIENT_Z)
    spans = [volume.zooms[i] for i in range(3) if i != axis]
    return float(spans[0] * spans[1])


def measure_psoas(
    volume: AnatomyVolume,
    z_mm: Optional[float] = None,
    z_by_side: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, object]:
    """Measure each side at ``z_by_side[side]`` when given, otherwise at ``z_mm``."""
    out: Dict[str, object] = {}
    voxel_area = _inplane_area_mm2(volume)
    for side in ("left", "right"):
        area_key = f"kidney_{side}_psoas_area_cm2"
        thick_key = f"kidney_{side}_psoas_thickness_mm"
        side_z = z_by_side.get(side) if z_by_side is not None else None
        if z_by_side is not None:
            z_mm_side = side_z
        else:
            z_mm_side = z_mm
        if z_mm_side is None:
            out[area_key] = None
            out[thick_key] = None
            out[f"{area_key}_qc"] = QC_MISSING
            out[f"{thick_key}_qc"] = QC_MISSING
            continue
        major = volume.slice_points(f"psoas_major_{side}", z_mm_side)
        qc = "ok"
        points = major
        if len(points) == 0:
            points = volume.slice_points(f"iliopsoas_{side}", z_mm_side)
            qc = QC_APPROX
        if len(points) == 0:
            out[area_key] = None
            out[thick_key] = None
            out[f"{area_key}_qc"] = QC_MISSING
            out[f"{thick_key}_qc"] = QC_MISSING
            continue
        area_cm2 = len(points) * voxel_area / 100.0
        thickness = float(points[:, 1].max() - points[:, 1].min())
        name = f"psoas_major_{side}" if qc == "ok" else f"iliopsoas_{side}"
        if volume.touches_border(name) and qc == "ok":
            qc = QC_OUT_OF_FOV
        out[area_key] = float(area_cm2)
        out[thick_key] = thickness
        out[f"{area_key}_qc"] = qc
        out[f"{thick_key}_qc"] = qc
    return out
