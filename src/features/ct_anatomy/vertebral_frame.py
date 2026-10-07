"""Kidney thirds in the vertebral frame used by the Excel sheet.

The manual coordinates are not DICOM patient millimetres. The reader aligns
the study to the vertebral body and writes:

* X as an unsigned distance from that body's midline (both kidneys positive);
* Y and Z as signed offsets from the same body.

The origin is the vertebral body on the axial slice through the kidneys, not
the centre of the whole torso and not a rib or pelvic bone nearest the skin.
No constant is added: a remaining shift has to show up in the next comparison.

``measure_lyashchenko`` adds the protocol from Lyashchenko, Demin and Urazov:
OY through the spinous process, OX through the posterior point of the canal,
four origin/point variants, and fat, psoas and spine gap on that same slice.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.features.ct_anatomy.distances import _vertebral_body_slice
from src.features.ct_anatomy.kidney_shape import _thirds
from src.features.ct_anatomy.qc import QC_MISSING
from src.features.ct_anatomy.vertebral_axes import measure_lyashchenko
from src.features.ct_anatomy.volume import AnatomyVolume

# Same limits as the clinical body-size sanitizer. A cropped HU blob must not
# replace the skin-to-skin columns the comparison reads.
_WIDTH_RANGE = (200.0, 500.0)
_DEPTH_RANGE = (140.0, 400.0)


def _finite(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def measure_vertebral_frame(volume: AnatomyVolume) -> Dict[str, object]:
    """Unsigned X and signed Y/Z of each kidney third relative to one vertebra."""
    out: Dict[str, object] = {}
    thirds: Dict[str, Dict[str, np.ndarray]] = {}
    levels = []
    for side in ("left", "right"):
        points = volume.points(f"kidney_{side}")
        if len(points) < 10:
            thirds[side] = {}
            continue
        found = _thirds(points)
        thirds[side] = found
        middle = found.get("middle")
        if middle is not None:
            levels.append(float(middle[2]))

    origin = None
    origin_qc = QC_MISSING
    if levels:
        body, origin_qc = _vertebral_body_slice(volume, float(np.mean(levels)))
        if len(body):
            origin = body.mean(axis=0)

    out["vert_origin_qc"] = origin_qc if origin is not None else QC_MISSING
    if origin is None:
        out["vert_origin_x"] = None
        out["vert_origin_y"] = None
        out["vert_origin_z"] = None
    else:
        out["vert_origin_x"] = float(origin[0])
        out["vert_origin_y"] = float(origin[1])
        out["vert_origin_z"] = float(origin[2])

    for side in ("left", "right"):
        for name in ("upper", "middle", "lower"):
            point = thirds.get(side, {}).get(name)
            for axis_name, signed in (("x", False), ("y", True), ("z", True)):
                key = f"kidney_{side}_{name}_{axis_name}_vert"
                if origin is None or point is None:
                    out[key] = None
                    out[f"{key}_qc"] = QC_MISSING
                    continue
                axis = "xyz".index(axis_name)
                delta = float(point[axis] - origin[axis])
                out[key] = delta if signed else abs(delta)
                out[f"{key}_qc"] = origin_qc
    out.update(measure_lyashchenko(volume))
    return out


def publish_vertebral_frame(row: Dict[str, object]) -> Dict[str, object]:
    """Copy the vertebra origin and the skin size onto the columns readers use.

    ``spine_center_*_mm`` is the reference later passes subtract. It must be
    the vertebral body, so a missing spine is left empty instead of the torso
    centre. Skin width and depth replace the cropped HU blob in the median
    columns when they sit in the clinical range.
    """
    updates: Dict[str, object] = {}
    origin_x = _finite(row.get("vert_origin_x"))
    origin_y = _finite(row.get("vert_origin_y"))
    origin_z = _finite(row.get("vert_origin_z"))
    if origin_x is not None and origin_y is not None and origin_z is not None:
        updates["spine_center_x_mm"] = origin_x
        updates["spine_center_y_mm"] = origin_y
        updates["spine_center_z_mm"] = origin_z
        updates["spine_center_source"] = "vertebral_body_at_kidney"

    width = _finite(row.get("body_width_mm"))
    depth = _finite(row.get("body_depth_mm"))
    if width is None:
        width = _finite(row.get("abd_width_l3l4_mm"))
    if depth is None:
        depth = _finite(row.get("abd_depth_l3l4_mm"))
    if (
        width is not None
        and depth is not None
        and _WIDTH_RANGE[0] <= width <= _WIDTH_RANGE[1]
        and _DEPTH_RANGE[0] <= depth <= _DEPTH_RANGE[1]
    ):
        updates["body_width_mm"] = width
        updates["body_depth_mm"] = depth
        updates["body_width_mm_median"] = width
        updates["body_depth_mm_median"] = depth
        updates["body_area_mm2_median"] = float(width * depth)
        updates["body_size_source"] = row.get("body_size_source") or row.get("body_mask_source") or "l3l4_skin"
    return updates
