"""Tests for canonical feature pipeline module."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.pipeline import (
    build_inference_matrix,
    normalize_raw_features,
    print_canonical_flow,
)
from src.features.phase1_schema import BASE_FEATURES, CLINICAL_DEMOGRAPHIC_FEATURES, TARGET_NAMES


def test_normalize_raw_features_alias():
    df = pd.DataFrame([{"body_com_x_mm": 1.0, "kidney_left_vs_spine_x": -10.0}])
    out = normalize_raw_features(df)
    assert out["body_com_x"].iloc[0] == 1.0
    assert out["kidney_left_center_x_rel"].iloc[0] == -10.0


def test_build_inference_matrix_requires_feature_names():
    sys.path.insert(0, str(ROOT / "models" / "phase1"))
    from adaptive_ensemble import AdaptiveEnsembleTrainer

    trainer = AdaptiveEnsembleTrainer()
    row = {c: 1.0 for c in BASE_FEATURES}
    df = pd.DataFrame([row])
    with pytest.raises(RuntimeError, match="feature_names is empty"):
        build_inference_matrix(trainer, df)


def test_build_feature_matrix_includes_clinical_demographics():
    """Feature rework: sex/age/bmi/body_type from the Vybor xlsx must reach
    the model's feature matrix, not just live unused in the CSV."""
    sys.path.insert(0, str(ROOT / "models" / "phase1"))
    from adaptive_ensemble import AdaptiveEnsembleTrainer

    trainer = AdaptiveEnsembleTrainer(enrichment_mode="none")
    row = {c: 1.0 for c in BASE_FEATURES}
    row.update({t: 0.0 for t in TARGET_NAMES})
    row.update(
        {"sex": 1.0, "age": 45.0, "bmi": 24.5, "body_type": 0.0, "has_previous_surgery": 0.0}
    )
    df = pd.DataFrame([row, row])
    _, _, all_feature_cols, _ = trainer._build_feature_matrix(df)
    for col in CLINICAL_DEMOGRAPHIC_FEATURES:
        assert col in all_feature_cols


def test_print_canonical_flow(capsys):
    print_canonical_flow()
    out = capsys.readouterr().out
    assert "enhanced_ct_extractor" in out
    assert "adaptive_ensemble" in out
