"""Canonical Phase 1 feature pipeline — single API for prepare / infer.

Use this module instead of calling ``AdaptiveEnsembleTrainer`` engineering
methods directly from API scripts or one-off notebooks.

Flow:
  DICOM  → enhanced_ct_extractor (normalize_record)
  tables → normalize_dataframe (phase1_schema)
  train  → AdaptiveEnsembleTrainer.prepare_training_data_split
  infer  → build_inference_matrix → imputer → scaler → model
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

from .phase1_schema import (
    BASE_FEATURES,
    SCHEMA_VERSION,
    TARGET_NAMES,
    normalize_dataframe,
    normalize_record,
    validate_base_features,
)

# A model artifact trained with almost-constant columns can contain a
# StandardScaler scale close to floating-point noise. Dividing a valid CT
# value by it makes linear ensemble members produce astronomically large
# displacements. These limits are inference guards; ordinary in-distribution
# training values are unaffected.
_MIN_SCALER_SCALE = 1e-6
_MAX_ABS_SCALED_FEATURE = 12.0

# Re-export for callers that import from pipeline only
__all__ = [
    "SCHEMA_VERSION",
    "BASE_FEATURES",
    "TARGET_NAMES",
    "normalize_raw_features",
    "normalize_raw_record",
    "validate_raw_features",
    "build_inference_matrix",
    "apply_model_preprocessing",
    "predict_targets",
    "predict_quantiles",
]


def normalize_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Alias: map extractor / CSV columns to canonical BASE_FEATURES."""
    return normalize_dataframe(df)


def normalize_raw_record(record: Mapping[str, object]) -> Dict[str, object]:
    """Alias: normalize a single patient dict."""
    return normalize_record(record)


def validate_raw_features(df: pd.DataFrame, *, min_present_ratio: float = 0.8):
    return validate_base_features(df, min_present_ratio=min_present_ratio)


def build_inference_matrix(
    trainer: Any,
    patient_data: Union[Mapping[str, object], pd.DataFrame],
    *,
    feature_names: Optional[list] = None,
) -> np.ndarray:
    """Train-time feature engineering aligned to saved ``feature_names``."""
    if feature_names is not None:
        trainer.feature_names = list(feature_names)
    if isinstance(patient_data, pd.DataFrame):
        df = patient_data
    else:
        df = pd.DataFrame([dict(patient_data)])
    df = normalize_dataframe(df)
    return trainer.build_inference_matrix(df)


def apply_model_preprocessing(
    X: np.ndarray,
    model_data: Mapping[str, Any],
) -> np.ndarray:
    """Apply persisted preprocessing safely for out-of-distribution CT inputs.

    Near-constant training features contain no usable patient signal. Their
    standardized value is fixed at zero instead of dividing by numerical
    noise. Remaining standardized values are clipped to prevent an
    out-of-distribution coordinate from dominating linear voters.
    """
    imputer = model_data.get("imputer")
    if imputer is not None:
        X = imputer.transform(X)
    scaler = model_data["scaler"]
    X_scaled = scaler.transform(X)

    scale = np.asarray(getattr(scaler, "scale_", []), dtype=float)
    if scale.shape == (X_scaled.shape[1],):
        near_constant = ~np.isfinite(scale) | (np.abs(scale) < _MIN_SCALER_SCALE)
        if near_constant.any():
            X_scaled[:, near_constant] = 0.0

    return np.nan_to_num(
        np.clip(X_scaled, -_MAX_ABS_SCALED_FEATURE, _MAX_ABS_SCALED_FEATURE),
        nan=0.0,
        posinf=_MAX_ABS_SCALED_FEATURE,
        neginf=-_MAX_ABS_SCALED_FEATURE,
    )


def predict_targets(
    trainer: Any,
    model_data: Mapping[str, Any],
    patient_data: Union[Mapping[str, object], pd.DataFrame],
) -> Dict[str, float]:
    """Full inference via the shared ``src.models.runtime`` facade."""
    from src.models.runtime import RuntimePredictor

    payload = dict(model_data)
    if trainer is not None:
        payload.setdefault("feature_names", getattr(trainer, "feature_names", None))
        payload.setdefault("enrichment_mode", getattr(trainer, "enrichment_mode", "projection"))
        store = getattr(trainer, "na_trend_store", None)
        if payload.get("na_trend_store") is None and store is not None:
            payload["na_trend_store"] = store.to_dict() if hasattr(store, "to_dict") else store
        payload.setdefault("z_head", getattr(trainer, "z_head", "ensemble"))
        payload.setdefault("z_driver_names", getattr(trainer, "z_driver_names", None))
        payload.setdefault("categorical_encoder", getattr(trainer, "categorical_encoder_", None))
        payload.setdefault("encode_categoricals", getattr(trainer, "encode_categoricals", False))
    return RuntimePredictor.from_payload(payload).predict_targets(patient_data)


def predict_quantiles(
    trainer: Any,
    model_data: Mapping[str, Any],
    patient_data: Union[Mapping[str, object], pd.DataFrame],
) -> Dict[str, Dict[str, float]]:
    """Return P10/P50/P90 intervals per target when quantile_model is saved."""
    from src.models.runtime import RuntimePredictor

    payload = dict(model_data)
    if trainer is not None:
        payload.setdefault("feature_names", getattr(trainer, "feature_names", None))
        payload.setdefault("enrichment_mode", getattr(trainer, "enrichment_mode", "projection"))
        store = getattr(trainer, "na_trend_store", None)
        if payload.get("na_trend_store") is None and store is not None:
            payload["na_trend_store"] = store.to_dict() if hasattr(store, "to_dict") else store
    return RuntimePredictor.from_payload(payload).predict_quantiles(patient_data)


def print_canonical_flow() -> None:
    """Stdout summary of supported commands (for CLI ``info`` subcommand)."""
    lines = [
        f"Phase 1 feature pipeline ({SCHEMA_VERSION})",
        "",
        "1. Extract (DICOM -> canonical base columns):",
        "   python scripts/inference/enhanced_ct_extractor.py <dicom_root> --output out.csv",
        "",
        "2. Integrate sources -> data/processed/:",
        "   python src/models/data_integration_fix.py",
        "",
        "3. Train ensemble:",
        "   python -c \"from src.models.ensemble import AdaptiveEnsembleTrainer\"",
        "   # or: python models/phase1/adaptive_ensemble.py (legacy shim)",
        "",
        "4. Validate (smoke + metrics):",
        "   python scripts/run_phase1_pipeline.py validate --run-id RUN_ID",
        "",
        "5. Inference API:",
        "   uvicorn src.api.kidney_displacement_api:app --port 8000",
        "",
        "Orchestrator: scripts/run_phase1_pipeline.py",
        "Docs:         docs/PHASE1_PIPELINE_RUNBOOK.md",
        "Schema:       config/phase1_feature_schema.yaml",
        "Code:         src/features/phase1_schema.py + src/features/pipeline.py",
        "",
        "Legacy (do not use for new work):",
        "  - scripts/inference/dicom_feature_extractor.py",
        "  - scripts/inference/extract_from_dicom.py",
        "  - scripts/inference/convert_single_file.py",
        "  - src/data/prepare_dataset.py",
        "  - models/phase1/train_lasso.py, train_ridge.py, target_specific_ensemble.py",
    ]
    print("\n".join(lines))
