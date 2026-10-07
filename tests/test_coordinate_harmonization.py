"""Tests for DICOM -> Vybor coordinate harmonization."""

from pathlib import Path

import pandas as pd

from src.features.coordinate_harmonization import (
    build_reference_stats,
    harmonize_dataframe,
)
from src.features.phase1_schema import normalize_dataframe

ROOT = Path(__file__).resolve().parents[1]


def test_y_flip_moves_na_y_rel_toward_vybor():
    ref_path = ROOT / "data" / "vybor_from_xlsx.csv"
    if not ref_path.exists():
        ref_path = ROOT / "data" / "vybor_unified_features.csv"
    ref = normalize_dataframe(pd.read_csv(ref_path))
    reference = build_reference_stats(ref)

    na_raw = pd.read_csv("data/na_spine_full.csv", nrows=50)
    na_raw = na_raw[na_raw["status"] == "extracted"].head(20)
    aligned = harmonize_dataframe(na_raw, reference, source_kind="na_spine")

    ref_y = ref["kidney_left_center_y_rel"].median()
    raw_y = normalize_dataframe(na_raw)["kidney_left_center_y_rel"].median()
    ali_y = aligned["kidney_left_center_y_rel"].median()

    assert raw_y > 0
    assert abs(ali_y - ref_y) < abs(raw_y - ref_y)


def test_aligned_body_width_closer_to_vybor():
    ref_path = ROOT / "data" / "vybor_from_xlsx.csv"
    if not ref_path.exists():
        ref_path = ROOT / "data" / "vybor_unified_features.csv"
    ref = normalize_dataframe(pd.read_csv(ref_path))
    reference = build_reference_stats(ref)
    na_raw = pd.read_csv("data/na_spine_full.csv")
    na_raw = na_raw[na_raw["status"] == "extracted"].head(30)
    aligned = harmonize_dataframe(na_raw, reference, source_kind="na_spine")

    ref_w = ref["body_width_mm"].median()
    raw_w = normalize_dataframe(na_raw)["body_width_mm"].median()
    ali_w = aligned["body_width_mm"].median()

    assert abs(ali_w - ref_w) < abs(raw_w - ref_w)


def test_aligned_length_not_zeroed_when_vybor_missing():
    ref_path = ROOT / "data" / "vybor_from_xlsx.csv"
    if not ref_path.exists():
        ref_path = ROOT / "data" / "vybor_unified_features.csv"
    ref = normalize_dataframe(pd.read_csv(ref_path))
    reference = build_reference_stats(ref)
    na_raw = pd.read_csv("data/na_spine_full.csv")
    na_raw = na_raw[na_raw["status"] == "extracted"].head(30)
    aligned = harmonize_dataframe(na_raw, reference, source_kind="na_spine")

    raw_len = normalize_dataframe(na_raw)["kidney_left_length_mm"].median()
    ali_len = aligned["kidney_left_length_mm"].median()
    assert ali_len > 50
    assert abs(ali_len - raw_len) < abs(raw_len)


def test_vybor_identity_passthrough():
    ref_path = ROOT / "data" / "vybor_from_xlsx.csv"
    if not ref_path.exists():
        ref_path = ROOT / "data" / "vybor_unified_features.csv"
    ref = normalize_dataframe(pd.read_csv(ref_path))
    reference = build_reference_stats(ref)
    out = harmonize_dataframe(ref.head(5), reference, source_kind="vybor")
    assert (out["harmonization_applied"] == "identity").all()
    pd.testing.assert_series_equal(
        ref.head(5)["kidney_left_center_x_rel"].reset_index(drop=True),
        out["kidney_left_center_x_rel"].reset_index(drop=True),
        check_names=False,
    )


def test_harmonize_skips_iqr_rescale_for_collapsed_signed_x():
    """Zero-IQR X rel must not be mapped onto the clinical train median."""
    ref = pd.DataFrame(
        {
            "kidney_left_center_x_rel": [1.5, 2.0, -1.0, 3.0, 0.5, 4.0, -2.0, 1.0],
            "kidney_right_center_x_rel": [-1.5, -2.0, 1.0, -3.0, -0.5, -4.0, 2.0, -1.0],
            "kidney_left_center_y_rel": [0.0] * 8,
            "kidney_right_center_y_rel": [0.0] * 8,
            "kidney_left_center_z_rel": [0.0] * 8,
            "kidney_right_center_z_rel": [0.0] * 8,
            "body_width_mm": [300.0] * 8,
            "body_depth_mm": [200.0] * 8,
            "spine_center_x": [70.0] * 8,
            "spine_center_y": [20.0] * 8,
            "spine_center_z": [-50.0] * 8,
            "body_com_x": [72.0] * 8,
            "body_com_y": [22.0] * 8,
            "body_com_z": [-50.0] * 8,
        }
    )
    reference = build_reference_stats(ref)
    collapsed = pd.DataFrame(
        {
            "kidney_left_center_x_rel": [0.0] * 8,
            "kidney_right_center_x_rel": [0.0] * 8,
            "kidney_left_center_y_rel": [1.0] * 8,
            "kidney_right_center_y_rel": [-1.0] * 8,
            "kidney_left_center_z_rel": [0.0] * 8,
            "kidney_right_center_z_rel": [0.0] * 8,
            "body_width_mm": [310.0] * 8,
            "body_depth_mm": [210.0] * 8,
            "spine_center_x": [0.0] * 8,
            "spine_center_y": [0.0] * 8,
            "spine_center_z": [0.0] * 8,
            "body_com_x": [0.0] * 8,
            "body_com_y": [0.0] * 8,
            "body_com_z": [0.0] * 8,
        }
    )
    aligned = harmonize_dataframe(collapsed, reference, source_kind="dicom_lps")
    assert aligned["kidney_left_center_x_rel"].abs().max() < 1e-9
    assert aligned["kidney_right_center_x_rel"].abs().max() < 1e-9

