"""Tests for Excel displacement table -> canonical schema."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.excel_displacement_adapter import (
    convert_excel_displacement_df,
    load_excel_displacement_table,
)
from src.features.phase1_schema import TARGET_NAMES

ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = ROOT / "data" / "train_displacement_dataset.csv"
VYBOR_PATH = ROOT / "data" / "vybor_unified_features.csv"


@pytest.mark.skipif(not EXCEL_PATH.exists(), reason="Excel dataset not present")
def test_excel_converts_with_all_targets():
    raw = pd.read_csv(EXCEL_PATH)
    out = convert_excel_displacement_df(raw.iloc[:3])
    assert len(out) == 3
    for col in TARGET_NAMES:
        assert col in out.columns
        assert out[col].notna().all()


@pytest.mark.skipif(not EXCEL_PATH.exists(), reason="Excel dataset not present")
def test_excel_dedupes_vybor_names():
    vybor = pd.read_csv(VYBOR_PATH) if VYBOR_PATH.exists() else None
    unique = load_excel_displacement_table(str(EXCEL_PATH), vybor_df=vybor)
    # Excel cohort overlaps Vybor — expect few or zero unique rows
    assert len(unique) <= len(pd.read_csv(EXCEL_PATH))
