"""Tests for xlsx displacement parser."""

from pathlib import Path

import pytest

from src.data.xlsx_displacement_parser import (
    DEFAULT_XLSX_PATH,
    build_vybor_from_xlsx,
    parse_xlsx_raw_table,
)
from src.features.phase1_schema import TARGET_NAMES

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
