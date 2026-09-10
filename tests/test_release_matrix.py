"""Block 7 focused test matrix (fast; no DICOM GPU, no n=87 nested OOF).

Companion coverage (do not duplicate here):
- geometry: tests/test_ct_geometry.py
- schema sync: tests/test_feature_schema.py
- preprocessing: tests/test_pipeline_preprocessing.py
- weighting: tests/test_model_justification.py
- artifact JSON cards: tests/test_artifact_manifest.py
- runtime predict_df/predict_targets: tests/test_runtime_predictor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.kidney_displacement_api import _performance_from_training_meta
from src.features.ct_geometry import flag_geometry_ood_x
from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES
from src.features.pipeline import predict_targets
from src.models.artifact_manifest import validate_manifest
from src.models.ensemble import AdaptiveEnsembleTrainer
from src.models.runtime import (
    ArtifactSchemaError,
    RuntimePredictor,
    bundle_from_payload,
    predict_df,
    validate_runtime_payload,
    warn_if_sklearn_mismatch,
)

ARCHIVE = ROOT / "models" / "archive"
CLINICAL_X_CSV = ROOT / "tests" / "fixtures" / "clinical_x_features.csv"


def _toy_row() -> dict:
    row = {c: float(i + 1) for i, c in enumerate(BASE_FEATURES)}
    row.update({t: 0.5 for t in TARGET_NAMES})
    return row


def _fitted_payload() -> dict:
    df = pd.DataFrame([_toy_row(), _toy_row(), _toy_row()])
    trainer = AdaptiveEnsembleTrainer(
        enrichment_mode="none",
        estimator_profile="tiny",
        encode_categoricals=False,
        yz_target_boost=False,
    )
    trainer.feature_names = list(BASE_FEATURES)
    X = trainer.build_inference_matrix(df)
    imputer = SimpleImputer(strategy="median").fit(X)
    X_imp = imputer.transform(X)
    scaler = StandardScaler().fit(X_imp)
    X_scaled = scaler.transform(X_imp)
    models = {}
    for target in TARGET_NAMES:
        est = DummyRegressor(strategy="mean")
        est.fit(X_scaled, df[target].values)
        models[target] = est
    return {
        "artifact_schema_version": "1.1.0",
        "feature_schema_version": "phase1_v1",
        "estimator_type": "ridge",
        "preprocessing_contract": {
            "scaler": "StandardScaler",
            "scaler_type": "StandardScaler",
            "imputer_type": "SimpleImputer",
        },
        "models": models,
        "scaler": scaler,
        "imputer": imputer,
        "feature_names": list(BASE_FEATURES),
        "target_names": list(TARGET_NAMES),
        "enrichment_mode": "none",
        "z_head": "ensemble",
        "z_driver_names": [],
        "na_trend_store": None,
        "encode_categoricals": False,
    }


def test_runtime_equivalence_offline_predict_df_and_pipeline() -> None:
    payload = _fitted_payload()
    runtime = RuntimePredictor.from_payload(payload)
    df = pd.DataFrame([_toy_row()])
    from_df = runtime.predict_df(df)
    from_row = runtime.predict_row(_toy_row())
    via_module = predict_df(runtime.bundle, df)
    via_pipeline = predict_targets(None, payload, _toy_row())
    via_targets = runtime.predict_targets(_toy_row())
    for col in TARGET_NAMES:
        assert from_df[col].iloc[0] == pytest.approx(from_row[col], rel=1e-12, abs=1e-12)
        assert from_row[col] == pytest.approx(via_pipeline[col], rel=1e-12, abs=1e-12)
        assert from_row[col] == pytest.approx(via_module[col].iloc[0], rel=1e-12, abs=1e-12)
        assert from_row[col] == pytest.approx(via_targets[col], rel=1e-12, abs=1e-12)


def test_golden_toy_row_is_constant_mean() -> None:
    """Deterministic golden prediction; DummyRegressor mean of 0.5 labels."""
    runtime = RuntimePredictor.from_payload(_fitted_payload())
    pred = runtime.predict_row(_toy_row())
    for col in TARGET_NAMES:
        assert pred[col] == pytest.approx(0.5, abs=1e-6)


def test_golden_clinical_x_fixture_signed_and_in_distribution() -> None:
    clinical = pd.read_csv(CLINICAL_X_CSV)
    assert len(clinical) >= 5
    for _, row in clinical.iterrows():
        left = float(row["kidney_left_center_x_rel"])
        right = float(row["kidney_right_center_x_rel"])
        sep = float(row["kidney_lr_sep_x"])
        assert left == pytest.approx(-right, abs=1e-6)
        assert sep == pytest.approx(right - left, abs=1e-6)
        is_ood, _ = flag_geometry_ood_x(row.to_dict())
        assert not is_ood


def test_missing_features_hard_fail() -> None:
    runtime = RuntimePredictor.from_payload(_fitted_payload())
    with pytest.raises(ValueError, match="Missing required base features"):
        runtime.predict_row({"age": 65})


def test_incompatible_artifact_schema_hard_fails() -> None:
    payload = _fitted_payload()
    payload["artifact_schema_version"] = "9.9.9"
    with pytest.raises(ArtifactSchemaError, match="Incompatible artifact_schema_version"):
        validate_runtime_payload(payload)


def test_corrupted_payload_not_a_mapping() -> None:
    with pytest.raises(ArtifactSchemaError, match="must be a mapping"):
        validate_runtime_payload([1, 2, 3])  # type: ignore[arg-type]


def test_incomplete_payload_empty_models() -> None:
    payload = _fitted_payload()
    payload["models"] = {}
    with pytest.raises(ArtifactSchemaError, match="models"):
        validate_runtime_payload(payload)


def test_incomplete_payload_missing_feature_names() -> None:
    payload = _fitted_payload()
    del payload["feature_names"]
    with pytest.raises(ArtifactSchemaError, match="feature_names"):
        validate_runtime_payload(payload)


def test_corrupted_joblib_file_refused(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.pkl"
    bad.write_bytes(b"not-a-valid-joblib-payload")
    with pytest.raises(Exception):
        RuntimePredictor.load(bad)


def test_ood_coordinates_flag_on_raw_lps() -> None:
    is_ood, reasons = flag_geometry_ood_x(
        {
            "kidney_left_center_x_rel": -72.0,
            "kidney_right_center_x_rel": 72.0,
            "kidney_lr_sep_x": 144.0,
            "kidney_left_to_spine_distance": 180.0,
            "kidney_right_to_spine_distance": 180.0,
        }
    )
    assert is_ood
    assert reasons


def test_sklearn_mismatch_warns_when_recorded_differs() -> None:
    payload = _fitted_payload()
    payload["sklearn_version"] = "9.9.9"
    with pytest.warns(UserWarning, match="sklearn"):
        warn_if_sklearn_mismatch(payload)


def test_sklearn_match_is_silent() -> None:
    import sklearn
    import warnings as _warnings

    payload = _fitted_payload()
    payload["sklearn_version"] = sklearn.__version__
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        warn_if_sklearn_mismatch(payload)
    assert caught == []


def test_model_info_does_not_invent_mae() -> None:
    payload = _fitted_payload()
    perf = _performance_from_training_meta(payload)
    assert perf["average_mae_mm"] is None
    assert perf["status"] == "unavailable"
    assert "8.52" not in str(perf)
    assert "2.14" not in str(perf)


def test_model_info_uses_training_meta_when_present() -> None:
    payload = _fitted_payload()
    payload["training_meta"] = {"performance": {"average_mae_mm": 12.3, "average_r2": 0.01}}
    perf = _performance_from_training_meta(payload)
    assert perf["status"] == "from_training_meta"
    assert perf["average_mae_mm"] == 12.3


def test_archive_cards_are_not_winners() -> None:
    cards = sorted(ARCHIVE.glob("adaptive_ensemble_clinical_honest_f1*.json"))
    assert len(cards) == 2
    counts = set()
    for path in cards:
        data = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest(data)
        assert data["production_winner"] is False
        assert data["research_only"] is True
        counts.add(data["feature_count"])
    assert counts == {111, 121}


def test_bundle_from_incomplete_scaler_fails() -> None:
    payload = _fitted_payload()
    payload["scaler"] = object()
    with pytest.raises(ArtifactSchemaError, match="scaler"):
        bundle_from_payload(payload)
