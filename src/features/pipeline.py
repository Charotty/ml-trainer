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
    """Apply persisted imputer (if any) and scaler from a saved ensemble pickle."""
    imputer = model_data.get("imputer")
    if imputer is not None:
        X = imputer.transform(X)
    scaler = model_data["scaler"]
    return scaler.transform(X)


def predict_targets(
    trainer: Any,
    model_data: Mapping[str, Any],
    patient_data: Union[Mapping[str, object], pd.DataFrame],
) -> Dict[str, float]:
    """Full inference: normalize → engineer → imputer → scaler → per-target predict."""
    feature_names = model_data["feature_names"]
    X = build_inference_matrix(trainer, patient_data, feature_names=feature_names)
    X_scaled = apply_model_preprocessing(X, model_data)
    predictions: Dict[str, float] = {}
    for target_name, model in model_data["models"].items():
        predictions[target_name] = float(model.predict(X_scaled)[0])
    return predictions


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
        "   python models/phase1/adaptive_ensemble.py",
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
