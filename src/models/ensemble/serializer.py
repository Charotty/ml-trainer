"""Artifact serializer: persist trainer state with an explicit schema contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib

from src.features.phase1_schema import SCHEMA_VERSION
from src.models.artifact_manifest import (
    ARTIFACT_SCHEMA_VERSION,
    extract_from_payload,
)

# Keep in sync with src.features.pipeline inference guards (avoid import cycle).
_MIN_SCALER_SCALE = 1e-6
_MAX_ABS_SCALED_FEATURE = 12.0

PACKAGE_STATUS = "production"

# Runtime payload schema (embedded in the joblib dict, distinct from JSON cards).
RUNTIME_PAYLOAD_SCHEMA_VERSION = "1.1.0"
SUPPORTED_RUNTIME_SCHEMA_VERSIONS = frozenset({"1.0.0", "1.1.0"})
SUPPORTED_FEATURE_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION, "phase1_v1"})

PREPROCESSING_CONTRACT: dict[str, Any] = {
    "imputer": "SimpleImputer(strategy=median)|None",
    "scaler": "StandardScaler",
    "near_constant_scale_floor": _MIN_SCALER_SCALE,
    "max_abs_scaled_feature": _MAX_ABS_SCALED_FEATURE,
    "notes": (
        "Imputer (if present) then scaler. Near-constant scaler columns are "
        "zeroed; remaining values are clipped. No silent RF fallback."
    ),
}


def estimator_type_from_payload(payload: Mapping[str, Any]) -> str:
    z_head = str(payload.get("z_head") or "ensemble")
    kind = str(payload.get("model_kind") or "ensemble")
    if z_head == "quantile_v7":
        return "quantile_v7"
    if kind in {"rf", "gbt", "ridge", "mean", "median"}:
        return kind
    return "adaptive_ensemble"


def preprocessing_contract_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    scaler = payload.get("scaler")
    imputer = payload.get("imputer")
    contract = dict(PREPROCESSING_CONTRACT)
    contract["scaler_type"] = type(scaler).__name__ if scaler is not None else None
    contract["imputer_type"] = type(imputer).__name__ if imputer is not None else None
    contract["imputer_strategy"] = getattr(imputer, "strategy", None) if imputer is not None else None
    return contract


def build_runtime_payload(
    trainer: Any,
    *,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a versioned joblib payload. Does not mark a production winner."""
    payload: dict[str, Any] = {
        "artifact_schema_version": RUNTIME_PAYLOAD_SCHEMA_VERSION,
        "feature_schema_version": SCHEMA_VERSION,
        "estimator_type": estimator_type_from_payload(
            {
                "z_head": getattr(trainer, "z_head", "ensemble"),
                "model_kind": getattr(trainer, "model_kind", "ensemble"),
            }
        ),
        "preprocessing_contract": dict(PREPROCESSING_CONTRACT),
        "models": trainer.trained_models,
        "scaler": trainer.scaler,
        "imputer": trainer.imputer,
        "feature_names": list(trainer.feature_names),
        "target_names": list(trainer.target_names),
        "required_features": list(getattr(trainer, "required_features", [])),
        "target_columns": list(getattr(trainer, "target_columns", [])),
        "adaptive_weights": getattr(trainer, "adaptive_weights", {}),
        "best_models": getattr(trainer, "best_models", {}),
        "z_head": getattr(trainer, "z_head", "ensemble"),
        "z_driver_names": list(getattr(trainer, "z_driver_names", []) or []),
        "enrichment_mode": getattr(trainer, "enrichment_mode", "projection"),
        "model_kind": getattr(trainer, "model_kind", "ensemble"),
        "encode_categoricals": bool(getattr(trainer, "encode_categoricals", False)),
        "categorical_encoder": getattr(trainer, "categorical_encoder_", None),
        "schema_version": ARTIFACT_SCHEMA_VERSION,
    }
    try:
        import sklearn

        payload["sklearn_version"] = sklearn.__version__
    except Exception:  # pragma: no cover
        pass
    store = getattr(trainer, "na_trend_store", None)
    if store is not None and hasattr(store, "to_dict"):
        payload["na_trend_store"] = store.to_dict()
    payload["preprocessing_contract"] = preprocessing_contract_from_payload(payload)
    payload.update(extract_from_payload(payload))
    if extras:
        payload.update(dict(extras))
    return payload


def save_trainer(trainer: Any, filepath: str | Path, *, extras: Mapping[str, Any] | None = None) -> Path:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_runtime_payload(trainer, extras=extras)
    joblib.dump(payload, path)
    print(f"Model saved to {path}")
    print(f"Saved {len(payload.get('models') or {})} trained models")
    print(f"Features: {len(payload.get('feature_names') or [])}, Targets: {len(payload.get('target_names') or [])}")
    return path
