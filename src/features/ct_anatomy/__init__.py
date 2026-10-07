"""CT anatomy measurements beyond the kidney-only extractor.

The ``fast`` profile keeps kidney segmentation and adds a skin mask so
``body_width_mm``, ``body_depth_mm`` and ``body_area_mm2`` come from the
torso rather than a cropped HU blob. The ``full`` profile adds vertebral
levels, L3–L4 abdominal size, perinephric fat, psoas and pole-to-landmark
distances. See ``profiles`` for which TotalSegmentator tasks each
measurement needs.
"""

from src.features.ct_anatomy.extract import (
    attach_body_type,
    extract_anatomy_features,
    extract_anatomy_from_seg_dir,
)
from src.features.ct_anatomy.qc import body_type_from_bmi
from src.features.ct_anatomy.segmentation import run_anatomy_segmentation

__all__ = [
    "attach_body_type",
    "body_type_from_bmi",
    "extract_anatomy_features",
    "extract_anatomy_from_seg_dir",
    "run_anatomy_segmentation",
]
