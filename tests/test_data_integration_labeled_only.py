"""KiTS19/DICOM must not appear as regression targets in train/val splits."""

from pathlib import Path

import pandas as pd
import pytest

from src.features.phase1_schema import TARGET_NAMES
from src.models.data_integration_fix import DataIntegrationFix, PROXY_TARGET_SOURCES

ROOT = Path(__file__).resolve().parents[1]
VYBOR = ROOT / "data" / "vybor_from_xlsx.csv"
KITS = ROOT / "data" / "kits19_medical_grade_features.csv"
DICOM = ROOT / "data" / "dicom_medical_features.csv"


@pytest.mark.skipif(not VYBOR.exists(), reason="vybor_from_xlsx.csv missing")
def test_labeled_only_train_excludes_proxy_sources():
    fixer = DataIntegrationFix(
        vybor_path=VYBOR,
        kits19_path=KITS if KITS.exists() else None,
        dicom_path=DICOM if DICOM.exists() else None,
        excel_path=None,
        training_mode="labeled_only",
    )
    master_df, train_df, val_df, _ = fixer.run()

    proxy_in_master = master_df[master_df["source"].isin(PROXY_TARGET_SOURCES)]
    if len(proxy_in_master):
        assert proxy_in_master[TARGET_NAMES].isna().all().all()

    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        if "source" not in split_df.columns:
            continue
        bad = split_df[split_df["source"].isin(PROXY_TARGET_SOURCES)]
        assert len(bad) == 0, f"{split_name} must not contain KiTS19/DICOM rows"

    assert set(train_df["source"].unique()).issubset({"Vybor", "Excel"})


@pytest.mark.skipif(not VYBOR.exists(), reason="vybor_from_xlsx.csv missing")
def test_extended_mode_still_clinical_only_for_targets():
    """Deprecated extended modes must not resurrect KiTS/DICOM proxy deltas in y."""
    fixer = DataIntegrationFix(
        vybor_path=VYBOR,
        kits19_path=KITS if KITS.exists() else None,
        dicom_path=DICOM if DICOM.exists() else None,
        excel_path=None,
        training_mode="clinical_xlsx_extended",
    )
    _, train_df, _, _ = fixer.run()
    if "source" in train_df.columns:
        assert not set(train_df["source"].unique()) & PROXY_TARGET_SOURCES
