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
