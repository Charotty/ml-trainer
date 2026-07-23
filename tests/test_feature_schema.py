"""Regression tests for canonical Phase 1 feature schema normalization."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.phase1_schema import (
    BASE_FEATURES,
    CLINICAL_DEMOGRAPHIC_FEATURES,
    ENGINEERED_FEATURES,
    OPTIONAL_METADATA_COLUMNS,
    TARGET_NAMES,
    encode_patient_position,
    normalize_dataframe,
    validate_base_features,
)


def test_encode_patient_position_dicom_codes():
    assert encode_patient_position("HFS") == 1
    assert encode_patient_position("supine") == 1
    assert encode_patient_position(None) == 1


def test_normalize_dicom_extractor_aliases():
    raw = pd.DataFrame([{
        "body_com_x_mm": 10.0,
        "body_com_y_mm": 20.0,
        "body_com_z_mm": 30.0,
        "spine_center_x_mm": 1.0,
        "spine_center_y_mm": 2.0,
        "spine_center_z_mm": 3.0,
        "body_width_mm_median": 250.0,
        "body_depth_mm_median": 180.0,
        "body_area_mm2_median": 45000.0,
        "kidney_left_vs_spine_x": -45.0,
        "kidney_left_vs_spine_y": 12.0,
        "kidney_left_vs_spine_z": -5.0,
        "kidney_right_vs_spine_x": 50.0,
        "kidney_right_vs_spine_y": 10.0,
        "kidney_right_vs_spine_z": -4.0,
        "kidney_left_volume_cm3": 140.0,
        "kidney_right_volume_cm3": 150.0,
        "kidney_left_length_mm": 100.0,
        "kidney_right_length_mm": 102.0,
        "patient_position": "HFS",
    }])
    out = normalize_dataframe(raw)
    assert out["body_com_x"].iloc[0] == 10.0
    assert out["body_width_mm"].iloc[0] == 250.0
    assert out["kidney_left_center_x_rel"].iloc[0] == -45.0
    assert out["patient_position_encoded"].iloc[0] == 1
    for col in BASE_FEATURES:
        assert col in out.columns


def test_validate_base_features_threshold():
    partial = pd.DataFrame([{BASE_FEATURES[0]: 1.0}])
    result = validate_base_features(partial, min_present_ratio=0.8)
    assert not result.is_valid
    assert BASE_FEATURES[0] in result.present_base


def test_schema_lists_match_training_contract():
    assert len(BASE_FEATURES) == 23
    assert len(ENGINEERED_FEATURES) == 13
    assert len(TARGET_NAMES) == 6


def test_clinical_demographic_features_are_model_inputs_not_optional_metadata():
    """sex/age/bmi/body_type must be real model inputs, not just optional
    metadata — this is the feature-rework contract requested for the
    dicexe/Vybor xlsx clinical/tabular columns."""
    assert CLINICAL_DEMOGRAPHIC_FEATURES == [
        "sex",
        "age",
        "bmi",
        "body_type",
        "has_previous_surgery",
    ]
    for col in CLINICAL_DEMOGRAPHIC_FEATURES:
        assert col not in OPTIONAL_METADATA_COLUMNS
