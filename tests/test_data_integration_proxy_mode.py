"""proxy_weighted_extended includes KiTS/DICOM proxy targets in train."""

from pathlib import Path

import pytest

from src.features.phase1_schema import TARGET_NAMES
from src.models.data_integration_fix import DataIntegrationFix, PROXY_TARGET_SOURCES

ROOT = Path(__file__).resolve().parents[1]
VYBOR = ROOT / "data" / "vybor_from_xlsx.csv"
KITS = ROOT / "data" / "harmonized" / "kits19_medical_grade_features_aligned.csv"
KITS_FALLBACK = ROOT / "data" / "kits19_medical_grade_features.csv"
DICOM_PSEUDO = ROOT / "data" / "harmonized" / "dicom_medical_features_pseudolabeled_proxy.csv"


@pytest.mark.skipif(not VYBOR.exists(), reason="vybor_from_xlsx.csv missing")
def test_proxy_weighted_mode_includes_kits_in_train():
    kits_path = KITS if KITS.exists() else KITS_FALLBACK
    if not kits_path.exists():
        pytest.skip("KiTS features missing")

    dicom_path = DICOM_PSEUDO if DICOM_PSEUDO.exists() else None
    fixer = DataIntegrationFix(
        vybor_path=VYBOR,
        kits19_path=kits_path,
        dicom_path=dicom_path,
        excel_path=None,
        training_mode="proxy_weighted_extended",
    )
    master_df, train_df, val_df, _ = fixer.run()

    kits_master = master_df[master_df["source"] == "KiTS19"]
    if len(kits_master):
        assert kits_master[TARGET_NAMES].notna().any().any(), "KiTS proxy targets must be kept"

    train_sources = set(train_df["source"].unique()) if "source" in train_df.columns else set()
    assert "KiTS19" in train_sources or len(kits_master) == 0

    val_sources = set(val_df["source"].unique()) if "source" in val_df.columns else set()
    assert not val_sources & PROXY_TARGET_SOURCES, "val must be clinical Vybor only"

    if "sample_weight" in train_df.columns:
        assert (train_df.loc[train_df["source"] == "Vybor", "sample_weight"] == 1.0).all()
