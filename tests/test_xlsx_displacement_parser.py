"""Tests for xlsx displacement parser."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.excel_displacement_adapter import convert_excel_displacement_df
from src.data.xlsx_displacement_parser import (
    DEFAULT_XLSX_PATH,
    attach_clinical_extras,
    build_vybor_from_xlsx,
    parse_xlsx_raw_table,
)
from src.features.phase1_schema import CLINICAL_DEMOGRAPHIC_FEATURES, TARGET_NAMES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not DEFAULT_XLSX_PATH.exists(), reason="Main xlsx not present")
def test_xlsx_raw_table_has_patients():
    raw = parse_xlsx_raw_table(DEFAULT_XLSX_PATH)
    assert len(raw) >= 90
    assert "fio" in raw.columns
    assert "left_delta_middle_x" in raw.columns


@pytest.mark.skipif(not DEFAULT_XLSX_PATH.exists(), reason="Main xlsx not present")
def test_build_vybor_from_xlsx_clinical_rows():
    df = build_vybor_from_xlsx(DEFAULT_XLSX_PATH, boku_path=None)
    assert len(df) >= 80
    for col in TARGET_NAMES:
        assert col in df.columns
        assert df[col].notna().all()


@pytest.mark.skipif(not DEFAULT_XLSX_PATH.exists(), reason="Main xlsx not present")
def test_build_vybor_from_xlsx_no_merge_suffix_columns():
    """Regression: attach_clinical_extras used to rename lumbar_lordosis_deg,
    s1_plate_tilt_deg and abd_wall_thickness_mm into _x/_y suffixed pairs on
    pandas merge collision, silently dropping them from the model contract.
    """
    df = build_vybor_from_xlsx(DEFAULT_XLSX_PATH, boku_path=None)
    for col in ("lumbar_lordosis_deg", "s1_plate_tilt_deg", "abd_wall_thickness_mm"):
        assert col in df.columns, f"{col} missing from canonical Vybor columns"
        assert not any(c.endswith(f"{col}_x") or c.endswith(f"{col}_y") for c in df.columns)
        assert df[col].notna().mean() > 0.5, f"{col} mostly NaN after merge fix"


@pytest.mark.skipif(not DEFAULT_XLSX_PATH.exists(), reason="Main xlsx not present")
def test_build_vybor_from_xlsx_has_clinical_demographic_columns():
    df = build_vybor_from_xlsx(DEFAULT_XLSX_PATH, boku_path=None)
    # sex/age/bmi/body_type are recorded for essentially every patient;
    # has_previous_surgery may be sparsely documented ("-" for unknown), so
    # only require the column to exist and carry at least some signal.
    for col in ("sex", "age", "bmi", "body_type"):
        assert col in df.columns
        assert df[col].notna().mean() > 0.5, f"{col} mostly NaN"
    assert "has_previous_surgery" in df.columns
    assert df["has_previous_surgery"].notna().sum() > 0


@pytest.mark.skipif(not DEFAULT_XLSX_PATH.exists(), reason="Main xlsx not present")
def test_build_vybor_from_xlsx_width_greater_than_depth_on_average():
    """Regression: abd_width_l3l4_mm (col 19, transverse) and
    abd_depth_l3l4_mm (col 18, AP/sagittal) used to be swapped in ``_COL``,
    which silently inverted body_width_mm/body_depth_mm relative to the
    width=X(transverse)/depth=Y(AP) convention used everywhere else (see
    scripts/inference/enhanced_ct_extractor.py). Transverse abdominal extent
    is anatomically larger than AP extent for the vast majority of patients.
    """
    df = build_vybor_from_xlsx(DEFAULT_XLSX_PATH, boku_path=None)
    both = df[["body_width_mm", "body_depth_mm"]].dropna()
    assert len(both) > 10
    assert (both["body_width_mm"] > both["body_depth_mm"]).mean() > 0.8


def test_attach_clinical_extras_coalesces_instead_of_suffixing():
    """Unit-level check of the merge fix without requiring the real xlsx file."""
    raw = pd.DataFrame(
        [
            {
                "fio": "Ivanov A.",
                "lumbar_lordosis_deg": 12.0,
                "s1_plate_tilt_deg": 5.0,
                "abd_wall_thickness_mm": 8.0,
                "kidney_left_z_span_supine_mm": 20.0,
            },
            {
                "fio": "Petrov B.",
                "lumbar_lordosis_deg": 15.0,
                "s1_plate_tilt_deg": 6.0,
                "abd_wall_thickness_mm": 9.0,
                "kidney_left_z_span_supine_mm": 22.0,
            },
        ]
    )
    converted = pd.DataFrame(
        [
            {
                "full_name": "Ivanov A.",
                "lumbar_lordosis_deg": np.nan,  # simulate NaN from convert step
                "s1_plate_tilt_deg": 5.0,
                "abd_wall_thickness_mm": 8.0,
            },
            {
                "full_name": "Petrov B.",
                "lumbar_lordosis_deg": 15.0,
                "s1_plate_tilt_deg": 6.0,
                "abd_wall_thickness_mm": 9.0,
            },
        ]
    )
    out = attach_clinical_extras(converted, raw)
    assert "lumbar_lordosis_deg_x" not in out.columns
    assert "lumbar_lordosis_deg_y" not in out.columns
    assert out["lumbar_lordosis_deg"].tolist() == [12.0, 15.0]  # NaN coalesced from raw
    assert out["s1_plate_tilt_deg"].tolist() == [5.0, 6.0]
    assert out["kidney_left_z_span_supine_mm"].tolist() == [20.0, 22.0]
