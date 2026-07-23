"""Tests for patient-LPS geometry helpers."""

import numpy as np
import pytest

from src.features.ct_geometry import (
    aggregate_body_at_z_band,
    harmonize_ct_to_clinical_frame,
    kidney_features_from_mask,
    mask_extent_patient_mm,
    merge_spine_relative,
    patient_kidney_side,
    sanitize_body_size_for_clinical_model,
)


def test_patient_kidney_side_lps():
    assert patient_kidney_side(50.0, 10.0) == "left"
    assert patient_kidney_side(-20.0, 10.0) == "right"


def test_kidney_features_from_mask_affine():
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    affine[:3, 3] = [100.0, -200.0, -50.0]
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 3:7, 4:9] = True
    zooms = (1.0, 1.0, 2.0)
    out = kidney_features_from_mask(mask, affine, zooms, "kidney_left")
    assert out["kidney_left_volume_cm3"] > 0
    assert out["kidney_left_length_mm"] > 0
    assert "kidney_left_center_x" in out


def test_merge_spine_relative_distances():
    features = {
        "kidney_left_center_x": 110.0,
        "kidney_left_center_y": -190.0,
        "kidney_left_center_z": -40.0,
        "body_com_x": 105.0,
        "body_com_y": -195.0,
        "body_com_z": -42.0,
    }
    merged = merge_spine_relative(features, 100.0, -200.0, -50.0)
    assert merged["kidney_left_center_x_rel"] == pytest.approx(10.0)
    assert merged["kidney_left_to_spine_distance"] == pytest.approx(
        np.sqrt(10 ** 2 + 10 ** 2 + 10 ** 2)
    )


def test_aggregate_body_at_z_band():
    metrics = [
        {"slice_z": 10.0, "body_width_mm": 200.0, "body_depth_mm": 150.0},
        {"slice_z": 20.0, "body_width_mm": 220.0, "body_depth_mm": 160.0},
        {"slice_z": 80.0, "body_width_mm": 999.0, "body_depth_mm": 999.0},
    ]
    band = aggregate_body_at_z_band(metrics, 9.0, 21.0)
    assert band["body_width_mm"] == pytest.approx(210.0)
    assert band["body_depth_mm"] == pytest.approx(155.0)


def test_harmonize_ct_to_clinical_frame_matches_train_scale():
    # Komarov-like LPS centers: signed X, large vertebral-spine distances.
    features = {
        "kidney_left_center_x": -81.671,
        "kidney_left_center_y": -8.884,
        "kidney_left_center_z": -533.419,
        "kidney_right_center_x": 62.671,
        "kidney_right_center_y": 6.821,
        "kidney_right_center_z": -555.032,
        "body_width_mm": 121.0,
        "body_depth_mm": 123.0,
    }
    out = harmonize_ct_to_clinical_frame(features)
    assert out["kidney_left_to_spine_distance"] < 40.0
    assert out["kidney_right_to_spine_distance"] < 40.0
    assert abs(out["kidney_left_center_x_rel"]) < 20.0
    assert abs(out["kidney_right_center_x_rel"]) < 20.0
    assert out["feature_frame"] == "clinical_midpoint_unsigned_x"


def test_sanitize_body_size_drops_fov_crop():
    out = sanitize_body_size_for_clinical_model(
        {"body_width_mm": 121.0, "body_depth_mm": 123.0, "body_area_mm2": 1.0}
    )
    assert out["body_width_mm"] is None
    assert out["body_depth_mm"] is None
    assert out["body_area_mm2"] is None
