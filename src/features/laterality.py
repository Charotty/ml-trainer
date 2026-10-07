"""Kidney laterality contract for train/serve: present / absent / not_assessed.

``absent`` means a confirmed missing kidney (empty mask or clinician override).
``not_assessed`` means segmentation did not find the organ and the clinician has
not confirmed absence — do not impute a second kidney as if it were typical.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from src.features.phase1_schema import LEFT_TARGET_NAMES, RIGHT_TARGET_NAMES

LATERALITY_PRESENT = "present"
LATERALITY_ABSENT = "absent"
LATERALITY_NOT_ASSESSED = "not_assessed"
LATERALITY_VALUES = frozenset(
    {LATERALITY_PRESENT, LATERALITY_ABSENT, LATERALITY_NOT_ASSESSED}
)

KIDNEY_LEFT_PRESENT = "kidney_left_present"
KIDNEY_RIGHT_PRESENT = "kidney_right_present"
LATERALITY_COLUMNS: tuple[str, ...] = (KIDNEY_LEFT_PRESENT, KIDNEY_RIGHT_PRESENT)

MASK_STATUS_OK = "ok"
MASK_STATUS_EMPTY = "empty_mask"
MASK_STATUS_MISSING_FILE = "missing_file"
MASK_STATUS_PARSE_ERROR = "parse_error"
MASK_STATUS_NO_LABEL = "no_label_in_combined"

MASK_STATUS_COLUMNS: tuple[str, ...] = (
    "kidney_left_mask_status",
    "kidney_right_mask_status",
)

_SIDES = ("left", "right")


def _finite(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not pd.notna(number):
        return None
    return number


def normalize_laterality(value: object) -> str:
    """Map a raw flag to one of the three contract values."""
    if value is None:
        return LATERALITY_NOT_ASSESSED
    text = str(value).strip().lower()
    if text in LATERALITY_VALUES:
        return text
    if text in {"1", "1.0", "true", "yes", "y", "да"}:
        return LATERALITY_PRESENT
    if text in {"0", "0.0", "false", "no", "n", "нет"}:
        return LATERALITY_ABSENT
    return LATERALITY_NOT_ASSESSED


def side_detected(row: Mapping[str, Any], side: str) -> bool:
    """True when extraction produced a usable kidney on ``side``."""
    volume = _finite(row.get(f"kidney_{side}_volume_cm3"))
    if volume is not None and volume > 0:
        return True
    axes = [
        _finite(row.get(f"kidney_{side}_center_{axis}"))
        for axis in "xyz"
    ]
    return all(v is not None for v in axes)


def infer_side_laterality(row: Mapping[str, Any], side: str) -> str:
    """Infer laterality from geometry + mask status (no clinician override)."""
    status = str(row.get(f"kidney_{side}_mask_status") or "").strip().lower()
    if side_detected(row, side):
        return LATERALITY_PRESENT
    if status == MASK_STATUS_EMPTY:
        return LATERALITY_ABSENT
    return LATERALITY_NOT_ASSESSED


def infer_laterality_from_extraction(row: Mapping[str, Any]) -> dict[str, str]:
    flags = {side: infer_side_laterality(row, side) for side in _SIDES}
    detected = [side for side, status in flags.items() if status == LATERALITY_PRESENT]
    unknown = [side for side, status in flags.items() if status == LATERALITY_NOT_ASSESSED]
    # Legacy extraction_raw.json has no mask_status. If exactly one kidney was
    # measured, do not invent the other via the imputer.
    if len(detected) == 1 and len(unknown) == 1:
        other = unknown[0]
        if not str(row.get(f"kidney_{other}_mask_status") or "").strip():
            flags[other] = LATERALITY_ABSENT
    return flags


def laterality_from_row(row: Mapping[str, Any]) -> dict[str, str]:
    """Explicit QA flags win; otherwise infer from extraction."""
    inferred = infer_laterality_from_extraction(row)
    out: dict[str, str] = {}
    for side in _SIDES:
        key = f"kidney_{side}_present"
        if key in row and row[key] not in (None, ""):
            out[side] = normalize_laterality(row[key])
        else:
            out[side] = inferred[side]
    return out


def attach_laterality(row: dict[str, Any]) -> dict[str, Any]:
    """Write ``kidney_{side}_present`` onto a feature/extraction row."""
    out = dict(row)
    flags = laterality_from_row(out)
    out[KIDNEY_LEFT_PRESENT] = flags["left"]
    out[KIDNEY_RIGHT_PRESENT] = flags["right"]
    return out


def targets_for_side(side: str) -> tuple[str, ...]:
    if side == "left":
        return tuple(LEFT_TARGET_NAMES)
    if side == "right":
        return tuple(RIGHT_TARGET_NAMES)
    raise ValueError(f"Unknown kidney side {side!r}")


def laterality_from_excel_labels(
    *,
    left_volume: object = None,
    right_volume: object = None,
    left_delta_x: object = None,
    right_delta_x: object = None,
) -> dict[str, str]:
    """Map Excel completeness to laterality for QA overrides."""

    def _side(volume: object, delta: object) -> str:
        if _finite(volume) is not None or _finite(delta) is not None:
            return LATERALITY_PRESENT
        return LATERALITY_ABSENT

    return {
        "left": _side(left_volume, left_delta_x),
        "right": _side(right_volume, right_delta_x),
    }
