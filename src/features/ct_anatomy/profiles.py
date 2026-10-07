"""TotalSegmentator task lists for the anatomy extractor.

Class names follow the ``total`` task in
https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py
Subtasks ``body``, ``vertebrae_body``, ``trunk_cavities`` and ``abdominal_muscles``
are Apache-2.0. ``tissue_types`` is intentionally omitted: it needs a non-commercial
license.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

PROFILE_FAST = "fast"
PROFILE_FULL = "full"

# Structures read from the default ``total`` task (per-class NIfTI files).
TOTAL_ROI: Tuple[str, ...] = (
    "kidney_right",
    "kidney_left",
    "vertebrae_T11",
    "vertebrae_T12",
    "vertebrae_L1",
    "vertebrae_L2",
    "vertebrae_L3",
    "vertebrae_L4",
    "vertebrae_L5",
    "vertebrae_S1",
    "sacrum",
    "rib_left_11",
    "rib_left_12",
    "rib_right_11",
    "rib_right_12",
    "hip_left",
    "hip_right",
    "iliopsoas_left",
    "iliopsoas_right",
    "liver",
    "spleen",
    "lung_lower_lobe_left",
    "lung_lower_lobe_right",
    "aorta",
    "inferior_vena_cava",
)

# (task, roi_subset or None, fast). Spine tasks stay at 1.5 mm.
# ``fast=True`` is the 3 mm model, used only where a rough envelope is enough.
TaskSpec = Tuple[str, Optional[Sequence[str]], bool]

FULL_TASKS: Tuple[TaskSpec, ...] = (
    # 3 mm total: the 1.5 mm model crashes the WSL GPU driver on an 8 GB card.
    ("total", TOTAL_ROI, True),
    ("body", None, True),
    # vertebrae_body rejects --fast; it runs on CPU (see segmentation.py).
    # trunk_cavities and abdominal_muscles are omitted: the full-resolution
    # models do not fit in WSL (GPU driver crash or the 15 GB RAM cap) and
    # this TotalSegmentator build refuses --fast for both.
    ("vertebrae_body", None, False),
    # ("trunk_cavities", None, False),
    # ("abdominal_muscles", None, False),
)

VERTEBRA_LEVELS: Tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "S1")

# Relative paths under the segmentation directory. First existing file wins.
MASK_FILES: dict[str, Tuple[str, ...]] = {
    "kidney_left": ("total/kidney_left.nii.gz", "kidney_left.nii.gz"),
    "kidney_right": ("total/kidney_right.nii.gz", "kidney_right.nii.gz"),
    "liver": ("total/liver.nii.gz",),
    "spleen": ("total/spleen.nii.gz",),
    "lung_lower_lobe_left": ("total/lung_lower_lobe_left.nii.gz",),
    "lung_lower_lobe_right": ("total/lung_lower_lobe_right.nii.gz",),
    "aorta": ("total/aorta.nii.gz",),
    "inferior_vena_cava": ("total/inferior_vena_cava.nii.gz",),
    "sacrum": ("total/sacrum.nii.gz",),
    "body_trunc": ("body/body_trunc.nii.gz", "body/body.nii.gz"),
    "vertebrae_body": ("vertebrae_body/vertebrae_body.nii.gz",),
    "intervertebral_discs": ("vertebrae_body/intervertebral_discs.nii.gz",),
    "abdominal_cavity": ("trunk_cavities/abdominal_cavity.nii.gz",),
    "iliopsoas_left": ("total/iliopsoas_left.nii.gz",),
    "iliopsoas_right": ("total/iliopsoas_right.nii.gz",),
    "psoas_major_left": ("abdominal_muscles/psoas_major_left.nii.gz",),
    "psoas_major_right": ("abdominal_muscles/psoas_major_right.nii.gz",),
    "rectus_abdominis_left": ("abdominal_muscles/rectus_abdominis_left.nii.gz",),
    "rectus_abdominis_right": ("abdominal_muscles/rectus_abdominis_right.nii.gz",),
}

for _level in ("T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1"):
    MASK_FILES[f"vertebrae_{_level}"] = (f"total/vertebrae_{_level}.nii.gz",)
for _side in ("left", "right"):
    for _rib in ("11", "12"):
        MASK_FILES[f"rib_{_side}_{_rib}"] = (f"total/rib_{_side}_{_rib}.nii.gz",)
    MASK_FILES[f"hip_{_side}"] = (f"total/hip_{_side}.nii.gz",)


def task_names(profile: str) -> List[str]:
    if profile != PROFILE_FULL:
        return []
    return [spec[0] for spec in FULL_TASKS]
