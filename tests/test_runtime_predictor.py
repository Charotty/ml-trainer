"""Unified runtime predictor: schema hard-fail and predict_df/predict_targets equivalence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES
from src.features.pipeline import predict_targets
from src.models.ensemble import AdaptiveEnsembleTrainer
from src.models.runtime import (
    ArtifactSchemaError,
    RuntimePredictor,
    bundle_from_payload,
    predict_df,
    validate_runtime_payload,
)
from src.models.span_anchor_calibrator import SpanAnchorCalibrator


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
    for i, target in enumerate(TARGET_NAMES):
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


def test_missing_categorical_encoder_hard_fails() -> None:
    payload = _fitted_payload()
    payload["feature_names"] = list(BASE_FEATURES) + ["sex_code_1", "sex_missing"]
    payload["encode_categoricals"] = True
    with pytest.raises(ArtifactSchemaError, match="categorical_encoder"):
        validate_runtime_payload(payload)


def test_reconstructed_encoder_satisfies_contract() -> None:
    from src.models.runtime import reconstruct_categorical_encoder

    payload = _fitted_payload()
    payload["feature_names"] = list(BASE_FEATURES) + ["sex_code_1", "sex_missing"]
    payload["encode_categoricals"] = True
    payload["categorical_encoder"] = reconstruct_categorical_encoder(payload["feature_names"])
    validate_runtime_payload(payload)
    assert payload["categorical_encoder"].fitted_ is True
    payload = _fitted_payload()
    payload["artifact_schema_version"] = "9.9.9"
    with pytest.raises(ArtifactSchemaError, match="Incompatible artifact_schema_version"):
        validate_runtime_payload(payload)


def test_incompatible_feature_schema_hard_fails() -> None:
    payload = _fitted_payload()
    payload["feature_schema_version"] = "other_v9"
    with pytest.raises(ArtifactSchemaError, match="Incompatible feature_schema_version"):
        validate_runtime_payload(payload)


def test_missing_scaler_hard_fails() -> None:
    payload = _fitted_payload()
    payload["scaler"] = None
    with pytest.raises(ArtifactSchemaError, match="scaler"):
        validate_runtime_payload(payload)


def test_predict_df_and_predict_targets_match() -> None:
    payload = _fitted_payload()
    runtime = RuntimePredictor.from_payload(payload)
    df = pd.DataFrame([_toy_row()])
    from_df = runtime.predict_df(df)
    from_row = runtime.predict_row(_toy_row())
    via_pipeline = predict_targets(None, payload, _toy_row())
    for col in TARGET_NAMES:
        assert from_df[col].iloc[0] == pytest.approx(from_row[col], rel=1e-12, abs=1e-12)
        assert from_row[col] == pytest.approx(via_pipeline[col], rel=1e-12, abs=1e-12)


def test_side_aware_calibrator_left_right() -> None:
    left = SpanAnchorCalibrator(side="left")
    right = SpanAnchorCalibrator(side="right")
    assert left.target == "kidney_left_delta_z"
    assert right.target == "kidney_right_delta_z"
    df = pd.DataFrame([_toy_row()])
    left.fit(df, [1.0], [1.0])
    right.fit(df, [2.0], [2.0])
    assert left.transform(df, [1.0]).shape == (1,)
    assert right.transform(df, [2.0]).shape == (1,)


def test_src_ensemble_import_without_phase1_on_path() -> None:
    from src.models.ensemble import AdaptiveEnsembleTrainer as T

    trainer = T(enrichment_mode="none", estimator_profile="tiny")
    trainer.feature_names = list(BASE_FEATURES)
    X = trainer.build_inference_matrix(pd.DataFrame([_toy_row()]))
    assert X.shape[1] == len(BASE_FEATURES)


def test_missing_features_raise_on_predict() -> None:
    runtime = RuntimePredictor.from_payload(_fitted_payload())
    with pytest.raises(ValueError, match="Missing required base features"):
        runtime.predict_df(pd.DataFrame([{"bmi": 22.0}]))


def test_sklearn_version_mismatch_warns_on_from_payload() -> None:
    payload = _fitted_payload()
    payload["sklearn_version"] = "0.0.1"
    with pytest.warns(UserWarning, match="sklearn"):
        RuntimePredictor.from_payload(payload)

