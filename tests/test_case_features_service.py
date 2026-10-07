"""Unit tests for case feature persistence and PATCH allowlist."""

from __future__ import annotations

from src.api.cases.features_service import (
    LEAKAGE_SPAN_COLUMNS,
    PATCH_ALLOWED_COLUMNS,
    SAFE_QA_CLINICAL_COLUMNS,
    merge_base_features,
    persistable_base_columns,
)
from src.features.displacement_axis_features import ANATOMICAL_FEATURES


def test_persistable_base_keeps_lordosis_and_spans() -> None:
    cols = persistable_base_columns()
    for name in SAFE_QA_CLINICAL_COLUMNS:
        assert name in cols
    for name in ("lumbar_lordosis_deg", "s1_plate_tilt_deg", "abd_wall_thickness_mm"):
        assert name in cols
    for leaked in LEAKAGE_SPAN_COLUMNS:
        assert leaked not in cols


def test_merge_accepts_clinic_and_rejects_delta_span() -> None:
    merged = merge_base_features(
        {"bmi": None, "body_width_mm": 90.0},
        {
            "bmi": 23.8,
            "lumbar_lordosis_deg": 69.3,
            "kidney_left_z_span_supine_mm": 90.4,
            "kidney_left_z_delta_span_mm": 11.0,
            "not_a_feature": 1.0,
        },
    )
    assert merged["bmi"] == 23.8
    assert merged["lumbar_lordosis_deg"] == 69.3
    assert merged["kidney_left_z_span_supine_mm"] == 90.4
    assert "kidney_left_z_delta_span_mm" not in merged
    assert "not_a_feature" not in merged
    assert "lumbar_lordosis_deg" in PATCH_ALLOWED_COLUMNS
    assert "bmi" in PATCH_ALLOWED_COLUMNS
    for name in ANATOMICAL_FEATURES:
        assert name in PATCH_ALLOWED_COLUMNS
    for leaked in LEAKAGE_SPAN_COLUMNS:
        assert leaked not in PATCH_ALLOWED_COLUMNS


def test_merge_accepts_laterality_strings() -> None:
    merged = merge_base_features(
        {"kidney_left_present": "not_assessed"},
        {"kidney_left_present": "absent", "kidney_right_present": "present"},
    )
    assert merged["kidney_left_present"] == "absent"
    assert merged["kidney_right_present"] == "present"
    assert "kidney_left_present" in PATCH_ALLOWED_COLUMNS
