"""Tests for external CT enrichment."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.ct_external_enrichment import (
    compute_anatomical_extras,
    compute_spans_from_upper_lower,
    enrich_external_ct_frame,
    recompute_body_com_offset,
)
from src.features.phase1_schema import normalize_dataframe

ROOT = Path(__file__).resolve().parents[1]
CLINICAL = ROOT / "data" / "vybor_from_xlsx.csv"
DICOM = ROOT / "data" / "harmonized" / "dicom_medical_features_aligned.csv"


def test_spans_from_upper_lower():
    df = pd.DataFrame(
        {
            "kidney_left_upper_z": [10.0],
            "kidney_left_lower_z": [4.0],
            "kidney_right_upper_y": [8.0],
            "kidney_right_lower_y": [2.0],
        }
    )
    out = compute_spans_from_upper_lower(df)
    assert out.loc[0, "kidney_left_z_span_supine_mm"] == pytest.approx(6.0)
    assert out.loc[0, "kidney_right_y_span_supine_mm"] == pytest.approx(6.0)


def test_body_com_offset_when_degenerate():
    df = pd.DataFrame(
        {
            "spine_center_x": [0.0],
            "spine_center_y": [0.0],
            "spine_center_z": [0.0],
            "body_com_x": [0.0],
            "body_com_y": [0.0],
            "body_com_z": [0.0],
            "body_depth_mm": [200.0],
            "body_width_mm": [300.0],
            "lumbar_lordosis_deg": [30.0],
            "s1_plate_tilt_deg": [10.0],
        }
    )
    out = recompute_body_com_offset(df)
    assert out.loc[0, "body_com_y"] == pytest.approx(200.0 * 0.06 * np.cos(np.deg2rad(30.0)), rel=1e-3)


def test_anatomical_extras_from_rel():
    df = pd.DataFrame(
        {
            "spine_center_x": [100.0],
            "spine_center_y": [50.0],
            "spine_center_z": [30.0],
            "kidney_left_center_x_rel": [-20.0],
            "kidney_left_center_y_rel": [5.0],
            "kidney_left_center_z_rel": [2.0],
            "kidney_right_center_x_rel": [20.0],
            "kidney_right_center_y_rel": [4.0],
            "kidney_right_center_z_rel": [1.0],
        }
    )
    out = compute_anatomical_extras(df)
    assert out.loc[0, "kidney_left_supine_middle_x"] == pytest.approx(80.0)
    assert out.loc[0, "kidney_lr_sep_x"] == pytest.approx(40.0)


@pytest.mark.skipif(not CLINICAL.exists() or not DICOM.exists(), reason="data missing")
def test_enrich_raises_dicom_coverage():
    clinical = normalize_dataframe(pd.read_csv(CLINICAL)).head(20)
    dicom = normalize_dataframe(pd.read_csv(DICOM)).head(30)
    enriched, meta = enrich_external_ct_frame(
        dicom,
        clinical_reference=clinical,
        source_id="dicom",
    )
    assert "projection_lookup_join" in meta["steps"]
    assert any(c.startswith("proj_sup_") for c in enriched.columns)
    assert enriched.filter(like="proj_sup_").notna().any().any()
