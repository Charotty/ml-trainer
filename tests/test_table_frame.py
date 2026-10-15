"""CT vertebral points mapped into the Excel sheet frame."""

from __future__ import annotations

import pandas as pd

from src.features.ct_geometry import harmonize_ct_to_clinical_frame
from src.features.table_frame import (
    align_ct_to_table_frame,
    constant_target_baseline,
    flag_table_geometry_ood,
    geometry_ranges_from_frame,
)


def _cohort_row() -> dict:
    """Lateral middle points from the paired cohort (CT vertebral frame)."""
    return {
        "kidney_left_middle_x_vert": 77.5,
        "kidney_left_middle_y_vert": 0.4,
        "kidney_left_middle_z_vert": 4.0,
        "kidney_right_middle_x_vert": 72.8,
        "kidney_right_middle_y_vert": 4.3,
        "kidney_right_middle_z_vert": -3.8,
        "body_width_mm": 98.8,
        "body_depth_mm": 126.3,
    }


def test_vertebral_middle_lands_near_the_sheet():
    out = align_ct_to_table_frame(_cohort_row())
    assert out["feature_frame"] == "clinical_midpoint_signed_x"
    assert abs(out["kidney_left_supine_middle_z"] - (-58.6)) < 0.2
    assert abs(out["kidney_right_supine_middle_z"] - (-66.4)) < 0.2
    assert abs(out["spine_center_z"] - (-62.5)) < 0.2
    assert abs(out["kidney_left_center_x_rel"]) < 5.0
    assert abs(out["kidney_lr_sep_x"]) < 10.0
    assert out["kidney_left_to_spine_distance"] < 20.0
    assert out["kidney_left_supine_middle_y"] > 30.0
    assert out["body_width_mm"] is None
    assert out["body_depth_mm"] is None
    assert out["geometry_ood_x"] is False


def test_large_negative_y_is_flipped_before_the_level_shift():
    row = _cohort_row()
    row["kidney_left_middle_y_vert"] = -30.0
    out = align_ct_to_table_frame(row)
    assert out["table_frame_y_flipped"] == ["left"]
    assert out["kidney_left_supine_middle_y"] > 60.0


def test_missing_vert_is_left_for_the_patient_harmonizer():
    raw = {
        "kidney_left_center_x": -80.0,
        "kidney_left_center_y": 10.0,
        "kidney_left_center_z": 20.0,
        "kidney_right_center_x": 70.0,
        "kidney_right_center_y": 12.0,
        "kidney_right_center_z": 18.0,
    }
    out = align_ct_to_table_frame(raw)
    assert "feature_frame" not in out
    harmonized = harmonize_ct_to_clinical_frame(out)
    assert harmonized["feature_frame"] == "clinical_midpoint_signed_x"


def test_harmonize_does_not_undo_the_table_frame():
    aligned = align_ct_to_table_frame(_cohort_row())
    again = harmonize_ct_to_clinical_frame(aligned)
    assert again["kidney_left_supine_middle_z"] == aligned["kidney_left_supine_middle_z"]
    assert again["spine_center_z"] == aligned["spine_center_z"]


def test_geometry_ranges_and_constant_baseline():
    frame = pd.DataFrame(
        {
            "kidney_left_supine_middle_z": list(range(-80, -20)),
            "kidney_left_to_spine_distance": [8.0] * 60,
            "kidney_left_delta_z": [-15.0, -5.0, 5.0] * 20,
        }
    )
    ranges = geometry_ranges_from_frame(frame)
    assert ranges["kidney_left_supine_middle_z"][0] < -70
    assert ranges["kidney_left_supine_middle_z"][1] > -30
    baseline = constant_target_baseline(frame)
    assert baseline["kidney_left_delta_z"]["mean_mm"] == -5.0
    assert baseline["kidney_left_delta_z"]["mae_mm"] > 0

    ood, reasons = flag_table_geometry_ood(
        {"kidney_left_supine_middle_z": 10.0, "kidney_left_to_spine_distance": 8.0},
        ranges,
    )
    assert ood is True
    assert any("supine_middle_z" in reason for reason in reasons)
    inside, _ = flag_table_geometry_ood(
        {"kidney_left_supine_middle_z": -50.0, "kidney_left_to_spine_distance": 8.0},
        ranges,
    )
    assert inside is False
