"""Tests for patient-LPS geometry helpers and the signed clinical X contract."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.coordinate_harmonization import compare_x_feature_distributions
from src.features.ct_geometry import (
    FEATURE_FRAME_CLINICAL_SIGNED_X,
    aggregate_body_at_z_band,
    clinical_x_from_patient_x,
    flag_geometry_ood_x,
    flip_lps_ras,
    harmonize_ct_to_clinical_frame,
    kidney_features_from_mask,
    merge_spine_relative,
    patient_kidney_side,
    sanitize_body_size_for_clinical_model,
    signed_rel_from_clinical_pair,
    unsigned_x_from_kidney_midpoint,
)

ROOT = Path(__file__).resolve().parents[1]
CLINICAL_X_CSV = ROOT / "tests" / "fixtures" / "clinical_x_features.csv"

# Komarov-like LPS/RAS-style centers (left negative, right positive).
KOMAROV = {
    "kidney_left_center_x": -81.671,
    "kidney_left_center_y": -8.884,
    "kidney_left_center_z": -533.419,
    "kidney_right_center_x": 62.671,
    "kidney_right_center_y": 6.821,
    "kidney_right_center_z": -555.032,
    "body_width_mm": 320.0,
    "body_depth_mm": 200.0,
}

# Golden: unsigned laterality vs origin 0, then signed midpoint encoding.
KOMAROV_LEFT_REL_X = 9.5
KOMAROV_RIGHT_REL_X = -9.5
KOMAROV_LR_SEP_X = -19.0


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


def test_old_unsigned_from_kidney_midpoint_always_equal():
    """Defect: abs(x - kidney_mid) makes left and right X algebraically identical."""
    for left_x, right_x in (
        (-70.0, 70.0),
        (-80.0, 60.0),
        (90.0, 230.0),
        (-81.671, 62.671),
        (80.0, -60.0),
    ):
        lu, ru = unsigned_x_from_kidney_midpoint(left_x, right_x)
        assert lu == pytest.approx(ru)
        left_rel = lu - 0.5 * (lu + ru)
        right_rel = ru - 0.5 * (lu + ru)
        assert left_rel == pytest.approx(0.0)
        assert right_rel == pytest.approx(0.0)
        assert (ru - lu) == pytest.approx(0.0)


def test_old_formula_zeros_asymmetric_relative_x():
    """Asymmetric kidneys still collapse under abs-from-kidney-midpoint."""
    left_x, right_x = -80.0, 60.0
    lu, ru = unsigned_x_from_kidney_midpoint(left_x, right_x)
    left_sup = np.array([lu, 0.0, 0.0])
    right_sup = np.array([ru, 0.0, 0.0])
    left_rel, right_rel, _ = signed_rel_from_clinical_pair(left_sup, right_sup)
    assert left_rel[0] == pytest.approx(0.0)
    assert right_rel[0] == pytest.approx(0.0)


def test_new_formula_symmetric_about_origin_is_legitimately_zero():
    out = harmonize_ct_to_clinical_frame(
        {
            "kidney_left_center_x": -70.0,
            "kidney_left_center_y": 10.0,
            "kidney_left_center_z": 5.0,
            "kidney_right_center_x": 70.0,
            "kidney_right_center_y": -4.0,
            "kidney_right_center_z": -5.0,
        }
    )
    assert out["kidney_left_center_x_rel"] == pytest.approx(0.0)
    assert out["kidney_right_center_x_rel"] == pytest.approx(0.0)
    assert out["kidney_lr_sep_x"] == pytest.approx(0.0)
    # Y/Z stay signed and non-zero.
    assert out["kidney_left_center_y_rel"] != pytest.approx(0.0)
    assert out["kidney_lr_sep_z"] != pytest.approx(0.0)


def test_new_formula_asymmetric_preserves_signed_x():
    out = harmonize_ct_to_clinical_frame(
        {
            "kidney_left_center_x": -80.0,
            "kidney_left_center_y": 0.0,
            "kidney_left_center_z": 0.0,
            "kidney_right_center_x": 60.0,
            "kidney_right_center_y": 0.0,
            "kidney_right_center_z": 0.0,
        }
    )
    assert out["kidney_left_center_x_rel"] == pytest.approx(10.0)
    assert out["kidney_right_center_x_rel"] == pytest.approx(-10.0)
    assert out["kidney_lr_sep_x"] == pytest.approx(-20.0)
    assert out["kidney_left_center_x_rel"] == pytest.approx(-out["kidney_right_center_x_rel"])
    assert out["feature_frame"] == FEATURE_FRAME_CLINICAL_SIGNED_X
    assert out["geometry_ood_x"] is False


def test_golden_komarov_expected_coords():
    out = harmonize_ct_to_clinical_frame(KOMAROV)
    assert out["kidney_left_center_x_rel"] == pytest.approx(KOMAROV_LEFT_REL_X)
    assert out["kidney_right_center_x_rel"] == pytest.approx(KOMAROV_RIGHT_REL_X)
    assert out["kidney_lr_sep_x"] == pytest.approx(KOMAROV_LR_SEP_X)
    assert abs(out["kidney_left_center_x_rel"]) < 20.0
    assert abs(out["kidney_right_center_x_rel"]) < 20.0
    assert out["kidney_left_to_spine_distance"] < 40.0
    assert out["kidney_right_to_spine_distance"] < 40.0
    assert out["kidney_left_center_x_rel"] != 0.0
    assert out["kidney_lr_sep_x"] != 0.0
    assert out["feature_frame"] == FEATURE_FRAME_CLINICAL_SIGNED_X
    assert out["geometry_ood_x"] is False


def test_golden_vertebral_spine_asymmetric():
    """Midline is the vertebra, not the kidney-kidney midpoint."""
    out = harmonize_ct_to_clinical_frame(
        {
            "kidney_left_center_x": 90.0,
            "kidney_left_center_y": 2.0,
            "kidney_left_center_z": 1.0,
            "kidney_right_center_x": 230.0,
            "kidney_right_center_y": -2.0,
            "kidney_right_center_z": -1.0,
            "spine_center_x": 150.0,
        }
    )
    # |90-150|=60, |230-150|=80 → rel ±10, sep = +20
    assert out["kidney_left_center_x_rel"] == pytest.approx(-10.0)
    assert out["kidney_right_center_x_rel"] == pytest.approx(10.0)
    assert out["kidney_lr_sep_x"] == pytest.approx(20.0)
    assert out["mid_sagittal_x_source"] == "spine_center_x"
    # Old helper would still zero this pair.
    lu, ru = unsigned_x_from_kidney_midpoint(90.0, 230.0)
    assert lu == pytest.approx(ru)


def test_round_trip_rel_plus_spine_recovers_clinical_x():
    left_x, right_x, midline = -80.0, 60.0, 0.0
    left_clin, right_clin = clinical_x_from_patient_x(left_x, right_x, midline)
    left_rel, right_rel, spine = signed_rel_from_clinical_pair(
        np.array([left_clin, 3.0, -4.0]),
        np.array([right_clin, -1.0, 2.0]),
    )
    assert (left_rel[0] + spine[0]) == pytest.approx(left_clin)
    assert (right_rel[0] + spine[0]) == pytest.approx(right_clin)
    assert (right_rel[0] - left_rel[0]) == pytest.approx(right_clin - left_clin)
    out = harmonize_ct_to_clinical_frame(
        {
            "kidney_left_center_x": left_x,
            "kidney_left_center_y": 3.0,
            "kidney_left_center_z": -4.0,
            "kidney_right_center_x": right_x,
            "kidney_right_center_y": -1.0,
            "kidney_right_center_z": 2.0,
        }
    )
    assert out["kidney_left_center_x_rel"] + out["spine_center_x"] == pytest.approx(left_clin)
    assert out["kidney_right_center_x_rel"] + out["spine_center_x"] == pytest.approx(right_clin)


def test_excel_supine_x_round_trip_matches_train_rel():
    """Clinical table X (both ~+70 mm) round-trips to the labeled relative X."""
    left_sup, right_sup = 74.7, 66.8
    out = harmonize_ct_to_clinical_frame(
        {
            "kidney_left_center_x": left_sup,
            "kidney_left_center_y": 15.1,
            "kidney_left_center_z": -85.7,
            "kidney_right_center_x": right_sup,
            "kidney_right_center_y": 26.5,
            "kidney_right_center_z": -103.4,
        }
    )
    assert out["kidney_left_center_x_rel"] == pytest.approx(3.95)
    assert out["kidney_right_center_x_rel"] == pytest.approx(-3.95)
    assert out["kidney_lr_sep_x"] == pytest.approx(-7.9)


def test_harmonize_is_idempotent_when_feature_frame_set():
    first = harmonize_ct_to_clinical_frame(KOMAROV)
    second = harmonize_ct_to_clinical_frame(first)
    assert second["kidney_left_center_x_rel"] == pytest.approx(first["kidney_left_center_x_rel"])
    assert second["kidney_lr_sep_x"] == pytest.approx(first["kidney_lr_sep_x"])
    assert second["feature_frame"] == FEATURE_FRAME_CLINICAL_SIGNED_X


def test_lps_ras_round_trip_and_clinical_x_invariant():
    lps = np.array([[80.0, -30.0, 100.0], [-60.0, 10.0, 90.0]])
    ras = flip_lps_ras(lps)
    assert ras[0, 0] == pytest.approx(-80.0)
    assert ras[0, 1] == pytest.approx(30.0)
    assert np.allclose(flip_lps_ras(ras), lps)

    lps_row = harmonize_ct_to_clinical_frame(
        {
            "kidney_left_center_x": 80.0,
            "kidney_left_center_y": -30.0,
            "kidney_left_center_z": 100.0,
            "kidney_right_center_x": -60.0,
            "kidney_right_center_y": 10.0,
            "kidney_right_center_z": 90.0,
        }
    )
    ras_row = harmonize_ct_to_clinical_frame(
        {
            "kidney_left_center_x": -80.0,
            "kidney_left_center_y": 30.0,
            "kidney_left_center_z": 100.0,
            "kidney_right_center_x": 60.0,
            "kidney_right_center_y": -10.0,
            "kidney_right_center_z": 90.0,
        }
    )
    assert lps_row["kidney_left_center_x_rel"] == pytest.approx(ras_row["kidney_left_center_x_rel"])
    assert lps_row["kidney_right_center_x_rel"] == pytest.approx(ras_row["kidney_right_center_x_rel"])
    assert lps_row["kidney_lr_sep_x"] == pytest.approx(ras_row["kidney_lr_sep_x"])


def test_x_rel_sign_and_non_zero_variability():
    offsets = (-20.0, -10.0, 5.0, 15.0, 25.0)
    left_rels = []
    seps = []
    for d in offsets:
        out = harmonize_ct_to_clinical_frame(
            {
                "kidney_left_center_x": -70.0 - d,
                "kidney_left_center_y": 1.0,
                "kidney_left_center_z": 0.0,
                "kidney_right_center_x": 70.0,
                "kidney_right_center_y": -1.0,
                "kidney_right_center_z": 0.0,
            }
        )
        left_rels.append(out["kidney_left_center_x_rel"])
        seps.append(out["kidney_lr_sep_x"])
        assert out["kidney_left_center_x_rel"] == pytest.approx(-out["kidney_right_center_x_rel"])
    assert np.std(left_rels) > 1.0
    assert np.std(seps) > 1.0
    assert any(v > 0 for v in left_rels) and any(v < 0 for v in left_rels)


def test_raw_lps_midpoint_rel_is_ood():
    raw = {
        "kidney_left_center_x_rel": -72.0,
        "kidney_right_center_x_rel": 72.0,
        "kidney_lr_sep_x": 144.0,
        "kidney_left_to_spine_distance": 180.0,
        "kidney_right_to_spine_distance": 180.0,
    }
    is_ood, reasons = flag_geometry_ood_x(raw)
    assert is_ood
    assert reasons


def test_x_distribution_gate_harmonized_matches_clinical():
    clinical = pd.read_csv(CLINICAL_X_CSV)
    infer = pd.DataFrame([harmonize_ct_to_clinical_frame(KOMAROV)])
    report = compare_x_feature_distributions(clinical, infer)
    assert report["compatible"], report
    assert not report["wild_divergence"], report


def test_x_distribution_gate_fails_raw_lps_without_ood_flag():
    clinical = pd.read_csv(CLINICAL_X_CSV)
    raw = pd.DataFrame(
        [
            {
                "kidney_left_center_x_rel": -72.0,
                "kidney_right_center_x_rel": 72.0,
                "kidney_lr_sep_x": 144.0,
                "geometry_ood_x": False,
            }
        ]
    )
    report = compare_x_feature_distributions(clinical, raw)
    assert report["wild_divergence"]
    assert not report["ood_flagged"]
    assert not report["compatible"]


def test_x_distribution_gate_allows_ood_when_flagged():
    clinical = pd.read_csv(CLINICAL_X_CSV)
    raw = pd.DataFrame(
        [
            {
                "kidney_left_center_x_rel": -72.0,
                "kidney_right_center_x_rel": 72.0,
                "kidney_lr_sep_x": 144.0,
                "geometry_ood_x": True,
            }
        ]
    )
    report = compare_x_feature_distributions(clinical, raw)
    assert report["wild_divergence"]
    assert report["ood_flagged"]
    assert report["compatible"]


def test_sanitize_body_size_drops_fov_crop():
    out = sanitize_body_size_for_clinical_model(
        {"body_width_mm": 121.0, "body_depth_mm": 123.0, "body_area_mm2": 1.0}
    )
    assert out["body_width_mm"] is None
    assert out["body_depth_mm"] is None
    assert out["body_area_mm2"] is None
