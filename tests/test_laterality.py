"""Laterality contract: present / absent / not_assessed."""

from src.features.laterality import (
    LATERALITY_ABSENT,
    LATERALITY_NOT_ASSESSED,
    LATERALITY_PRESENT,
    infer_laterality_from_extraction,
    laterality_from_excel_labels,
    laterality_from_row,
)
from src.api.cases.predictor import apply_laterality_gate


def test_empty_mask_is_absent_missing_file_is_not_assessed() -> None:
    flags = infer_laterality_from_extraction(
        {
            "kidney_right_volume_cm3": 242.6,
            "kidney_right_center_x": 94.0,
            "kidney_right_center_y": -4.0,
            "kidney_right_center_z": 1600.0,
            "kidney_right_mask_status": "ok",
            "kidney_left_mask_status": "empty_mask",
        }
    )
    assert flags["right"] == LATERALITY_PRESENT
    assert flags["left"] == LATERALITY_ABSENT

    flags2 = infer_laterality_from_extraction({"kidney_left_mask_status": "missing_file"})
    assert flags2["left"] == LATERALITY_NOT_ASSESSED
    assert flags2["right"] == LATERALITY_NOT_ASSESSED


def test_legacy_extraction_one_kidney_is_absent() -> None:
    flags = infer_laterality_from_extraction(
        {
            "kidney_right_volume_cm3": 242.6,
            "kidney_right_center_x": 94.0,
            "kidney_right_center_y": -4.0,
            "kidney_right_center_z": 1600.0,
        }
    )
    assert flags["right"] == LATERALITY_PRESENT
    assert flags["left"] == LATERALITY_ABSENT
    flags = laterality_from_excel_labels(
        left_volume=None,
        right_volume=265.6,
        left_delta_x=None,
        right_delta_x=-10.7,
    )
    assert flags == {"left": LATERALITY_ABSENT, "right": LATERALITY_PRESENT}


def test_absent_side_withholds_predictions() -> None:
    raw = {
        "kidney_left_delta_x": 1.0,
        "kidney_left_delta_y": 2.0,
        "kidney_left_delta_z": 3.0,
        "kidney_right_delta_x": 4.0,
        "kidney_right_delta_y": 5.0,
        "kidney_right_delta_z": 7.0,
    }
    gated, flags, withheld, notes = apply_laterality_gate(
        raw,
        {"kidney_left_present": "absent", "kidney_right_present": "present"},
    )
    assert flags["left"] == LATERALITY_ABSENT
    assert gated["kidney_left_delta_z"] is None
    assert gated["kidney_right_delta_z"] == 7.0
    assert "kidney_left_delta_x" in withheld
    assert any("отсутствует" in n for n in notes)


def test_explicit_qa_flag_wins_over_geometry() -> None:
    flags = laterality_from_row(
        {
            "kidney_left_present": "absent",
            "kidney_left_volume_cm3": 174.0,
            "kidney_left_center_x": 1.0,
            "kidney_left_center_y": 2.0,
            "kidney_left_center_z": 3.0,
        }
    )
    assert flags["left"] == LATERALITY_ABSENT
