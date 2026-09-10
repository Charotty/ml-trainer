"""Unified runtime predictor: one train/inference predict path.

Merges ``scripts.validation.common.predict_df`` and
``src.features.pipeline.predict_targets``. CT Workbench, kidney API :8000,
and validation scripts must call this module.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.fold_categoricals import ALL_MISS_COLS, FoldCategoricalEncoder
from src.features.phase1_schema import (
    BASE_FEATURES,
    SCHEMA_VERSION,
    normalize_dataframe,
    validate_base_features,
)
from src.features.pipeline import apply_model_preprocessing
from src.models.ensemble.features import FeatureTransformer, inference_transformer_from_payload
from src.models.ensemble.serializer import (
    SUPPORTED_FEATURE_SCHEMA_VERSIONS,
    SUPPORTED_RUNTIME_SCHEMA_VERSIONS,
    estimator_type_from_payload,
    preprocessing_contract_from_payload,
)
from src.models.side_z_predictor import LEFT_Z, RIGHT_Z
from src.models.z_quantile_v7 import Z_TARGETS, predict_quantile_z

PACKAGE_STATUS = "production"

DEFAULT_MODEL_PATH_STR = "models/adaptive_ensemble_clinical_honest.pkl"
LEGACY_MODEL_NAME = "adaptive_ensemble.pkl"
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Garbage/empty rows fail; partial CT extraction (some NaNs) is still allowed.
_MIN_INFERENCE_BASE_FEATURE_RATIO = 0.2


class ArtifactSchemaError(ValueError):
    """Incompatible or incomplete runtime artifact schema. No silent fallback."""


def default_model_path() -> Path:
    env = os.environ.get("MODEL_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _REPO_ROOT / DEFAULT_MODEL_PATH_STR


def warn_if_legacy_model(model_path: Path | str | None) -> None:
    if model_path is None:
        return
    path = Path(model_path)
    if path.name == LEGACY_MODEL_NAME:
        warnings.warn(
            f"Using legacy model path '{path}'. Prefer canonical "
            f"'{DEFAULT_MODEL_PATH_STR}'.",
            UserWarning,
            stacklevel=2,
        )


def _major_minor(version: str) -> tuple[int, int] | None:
    parts = str(version).strip().split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def recorded_sklearn_version(payload: Mapping[str, Any]) -> str | None:
    """Return sklearn version recorded on the artifact, if any."""
    recorded = payload.get("sklearn_version")
    if recorded:
        return str(recorded)
    meta = payload.get("training_meta")
    if isinstance(meta, Mapping):
        nested = meta.get("sklearn_version")
        if nested:
            return str(nested)
        env = meta.get("environment")
        if isinstance(env, Mapping) and env.get("sklearn"):
            return str(env["sklearn"])
    return None


def warn_if_sklearn_mismatch(payload: Mapping[str, Any]) -> None:
    """Warn when installed sklearn major.minor differs from the artifact card.

    Joblib payloads must come from trusted sources. A pin mismatch
    (cards recorded 1.9 vs requirements.txt ``scikit-learn<1.6``) can
    fail unpickle or silently change predictions.
    """
    recorded = recorded_sklearn_version(payload)
    if not recorded:
        return
    try:
        import sklearn

        installed = sklearn.__version__
    except Exception:  # pragma: no cover
        return
    rec_mm = _major_minor(recorded)
    inst_mm = _major_minor(installed)
    if rec_mm is None or inst_mm is None or rec_mm == inst_mm:
        return
    warnings.warn(
        f"sklearn {installed} differs from artifact-recorded {recorded}; "
        "joblib unpickle may fail or change predictions. "
        "Load joblib only from trusted sources and match training/inference pins.",
        UserWarning,
        stacklevel=2,
    )


_ENCODED_DUMMY_PREFIXES: tuple[str, ...] = ("sex_code_", "body_type_code_")
_ENCODED_MISSING_NAMES: frozenset[str] = frozenset(
    f"{col}_missing" for col in ALL_MISS_COLS
)


def payload_requires_categorical_encoder(feature_names: object) -> bool:
    names = [str(n) for n in (feature_names or [])]
    return any(
        name.startswith(_ENCODED_DUMMY_PREFIXES) or name in _ENCODED_MISSING_NAMES
        for name in names
    )


def reconstruct_categorical_encoder(feature_names: object) -> FoldCategoricalEncoder:
    """Rebuild the fold encoder from persisted dummy/missing column names.

    ``FoldCategoricalEncoder.fit`` only records column names from constants; it
    does not learn data-dependent parameters. Reconstructing it restores the
    train/serve contract without retraining the heads.
    """
    names = [str(n) for n in (feature_names or [])]
    encoder = FoldCategoricalEncoder()
    encoder.dummy_names = [
        name for name in names if name.startswith(_ENCODED_DUMMY_PREFIXES)
    ]
    encoder.missing_names = [name for name in names if name in _ENCODED_MISSING_NAMES]
    encoder.fitted_ = True
    return encoder


def _require(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload or payload[key] is None:
        raise ArtifactSchemaError(
            f"Artifact missing required inference field {key!r}; refusing to guess."
        )
    return payload[key]


def validate_runtime_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Hard-fail on incompatible schema. Never silently swap estimators."""
    if not isinstance(payload, Mapping):
        raise ArtifactSchemaError("Artifact payload must be a mapping")

    schema_ver = payload.get("artifact_schema_version")
    if schema_ver is not None:
        schema_ver = str(schema_ver)
        if schema_ver not in SUPPORTED_RUNTIME_SCHEMA_VERSIONS:
            raise ArtifactSchemaError(
                f"Incompatible artifact_schema_version {schema_ver!r}; "
                f"supported {sorted(SUPPORTED_RUNTIME_SCHEMA_VERSIONS)}. "
                "No silent fallback."
            )

    feature_schema = payload.get("feature_schema_version")
    if feature_schema is not None and str(feature_schema) not in SUPPORTED_FEATURE_SCHEMA_VERSIONS:
        raise ArtifactSchemaError(
            f"Incompatible feature_schema_version {feature_schema!r}; "
            f"expected one of {sorted(SUPPORTED_FEATURE_SCHEMA_VERSIONS)}. "
            "No silent fallback."
        )

    _require(payload, "feature_names")
    _require(payload, "models")
    _require(payload, "scaler")
    if not isinstance(payload["feature_names"], (list, tuple)):
        raise ArtifactSchemaError("feature_names must be a list")
    if not isinstance(payload["models"], Mapping) or not payload["models"]:
        raise ArtifactSchemaError("models must be a non-empty mapping")
    scaler = payload["scaler"]
    if not hasattr(scaler, "transform"):
        raise ArtifactSchemaError("scaler must expose transform(); refusing to skip preprocessing")

    if schema_ver == "1.1.0":
        _require(payload, "feature_schema_version")
        _require(payload, "estimator_type")
        contract = payload.get("preprocessing_contract")
        if not isinstance(contract, Mapping):
            raise ArtifactSchemaError(
                "artifact_schema_version 1.1.0 requires preprocessing_contract"
            )
        if contract.get("scaler") is None and contract.get("scaler_type") is None:
            raise ArtifactSchemaError("preprocessing_contract missing scaler type")

    if payload_requires_categorical_encoder(payload.get("feature_names")):
        encoder = payload.get("categorical_encoder")
        if encoder is None or not bool(getattr(encoder, "fitted_", False)):
            raise ArtifactSchemaError(
                "Artifact feature_names include encoded categoricals "
                "(sex_code_*/body_type_code_/*_missing) but categorical_encoder "
                "is missing or not fitted; refusing to impute a silent majority class."
            )
        if "encode_categoricals" in payload and not bool(payload.get("encode_categoricals")):
            raise ArtifactSchemaError(
                "encode_categoricals is False while encoded categorical columns "
                "are in feature_names; refusing to skip the encoder."
            )
    if "z_driver_names" not in payload:
        raise ArtifactSchemaError(
            "Artifact missing required inference field 'z_driver_names'; refusing to guess."
        )
    return payload


def _calibrator_target(calibrator: Any, fallback: str) -> str:
    target = getattr(calibrator, "target", None)
    if callable(target):
        try:
            target = calibrator.target()
        except TypeError:
            pass
    if isinstance(target, property):
        target = None
    if not target:
        target = getattr(type(calibrator), "TARGET", None) or fallback
    return str(target)


def _apply_side_z(pred_df: pd.DataFrame, side_z: Any, X_scaled: np.ndarray) -> None:
    if not side_z:
        return
    left_m = side_z.get("left") if isinstance(side_z, dict) else getattr(side_z, "left", None)
    right_m = side_z.get("right") if isinstance(side_z, dict) else getattr(side_z, "right", None)
    if left_m is not None and getattr(left_m, "fitted_", False):
        pred_df[LEFT_Z] = left_m.predict(X_scaled)
    if right_m is not None and getattr(right_m, "fitted_", False):
        pred_df[RIGHT_Z] = right_m.predict(X_scaled)


def _apply_calibrators(
    pred_df: pd.DataFrame,
    df_norm: pd.DataFrame,
    left_cal: Any,
    right_cal: Any,
) -> None:
    if left_cal is not None and hasattr(left_cal, "transform"):
        target = _calibrator_target(left_cal, "kidney_left_delta_z")
        if target in pred_df.columns:
            pred_df[target] = left_cal.transform(df_norm, pred_df[target].values)
    if right_cal is not None and hasattr(right_cal, "transform"):
        target = _calibrator_target(right_cal, "kidney_right_delta_z")
        if target in pred_df.columns:
            pred_df[target] = right_cal.transform(df_norm, pred_df[target].values)


def _apply_multitask_blend(
    pred_df: pd.DataFrame,
    X_scaled: np.ndarray,
    multitask: Any,
    blend: Mapping[str, Any] | None,
    target_names: List[str],
) -> None:
    if multitask is None or not getattr(multitask, "fitted_", False):
        return
    cfg = blend or {"z": 0.35, "xy": 0.15}
    z_blend = float(cfg.get("z", 0.35))
    xy_blend = float(cfg.get("xy", 0.15))
    mt = multitask.predict(X_scaled)
    names = list(getattr(multitask, "target_names", None) or target_names)
    for j, tgt in enumerate(names):
        if tgt not in pred_df.columns:
            continue
        if tgt.endswith("_z") and z_blend <= 0.0:
            continue
        w = z_blend if tgt.endswith("_z") else xy_blend
        pred_df[tgt] = (1.0 - w) * pred_df[tgt].values + w * mt[:, j]


@dataclass
class PredictorBundle:
    """Loaded artifact view used by validation scripts and RuntimePredictor."""

    mode: str
    feature_names: List[str]
    target_names: List[str]
    scaler: StandardScaler
    models: Dict[str, object]
    imputer: Optional[Any] = None
    left_z_calibrator: Optional[Any] = None
    right_z_calibrator: Optional[Any] = None
    side_z_models: Optional[dict] = None
    multitask_model: Optional[Any] = None
    multitask_blend: Optional[dict] = None
    quantile_model: Optional[Any] = None
    z_head: str = "ensemble"
    z_driver_names: Optional[List[str]] = None
    enrichment_mode: str = "projection"
    na_trend_store: Optional[Dict[str, Any]] = None
    artifact_schema_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    estimator_type: Optional[str] = None
    preprocessing_contract: Optional[Dict[str, Any]] = None
    encode_categoricals: bool = False
    categorical_encoder: Optional[Any] = None


def bundle_from_payload(payload: Mapping[str, Any], *, mode: str = "pretrained_adaptive_ensemble") -> PredictorBundle:
    validate_runtime_payload(payload)
    return PredictorBundle(
        mode=mode,
        feature_names=list(payload["feature_names"]),
        target_names=list(payload.get("target_names", payload["models"].keys())),
        scaler=payload["scaler"],
        models=dict(payload["models"]),
        imputer=payload.get("imputer"),
        left_z_calibrator=payload.get("left_z_calibrator"),
        right_z_calibrator=payload.get("right_z_calibrator"),
        side_z_models=payload.get("side_z_models"),
        multitask_model=payload.get("multitask_model"),
        multitask_blend=payload.get("multitask_blend"),
        quantile_model=payload.get("quantile_model"),
        z_head=payload.get("z_head", "ensemble"),
        z_driver_names=payload.get("z_driver_names"),
        enrichment_mode=payload.get("enrichment_mode", "projection"),
        na_trend_store=payload.get("na_trend_store"),
        artifact_schema_version=payload.get("artifact_schema_version"),
        feature_schema_version=payload.get("feature_schema_version", SCHEMA_VERSION),
        estimator_type=payload.get("estimator_type") or estimator_type_from_payload(payload),
        preprocessing_contract=(
            dict(payload["preprocessing_contract"])
            if isinstance(payload.get("preprocessing_contract"), Mapping)
            else preprocessing_contract_from_payload(payload)
        ),
        encode_categoricals=bool(payload.get("encode_categoricals", False)),
        categorical_encoder=payload.get("categorical_encoder"),
    )


def load_model_bundle(model_path: Path | str) -> PredictorBundle:
    model_path = Path(model_path)
    warn_if_legacy_model(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    payload = joblib.load(model_path)
    if not isinstance(payload, Mapping):
        raise ArtifactSchemaError(f"Artifact at {model_path} is not a mapping")
    warn_if_sklearn_mismatch(payload)
    return bundle_from_payload(payload)


def predict_from_bundle(bundle: PredictorBundle, df: pd.DataFrame) -> pd.DataFrame:
    """Canonical dataframe predict used by APIs and validation."""
    df_norm = normalize_dataframe(df)
    coverage = validate_base_features(
        df_norm, min_present_ratio=_MIN_INFERENCE_BASE_FEATURE_RATIO
    )
    if not coverage.is_valid:
        missing = ", ".join(coverage.missing_base[:12])
        raise ValueError(
            "Missing required base features for inference "
            f"({len(coverage.present_base)}/{len(BASE_FEATURES)} present; "
            f"need ≥{_MIN_INFERENCE_BASE_FEATURE_RATIO:.0%}). Missing include: {missing}"
        )
    transformer = inference_transformer_from_payload(
        {
            "feature_names": bundle.feature_names,
            "enrichment_mode": bundle.enrichment_mode,
            "na_trend_store": bundle.na_trend_store,
            "encode_categoricals": bundle.encode_categoricals,
            "categorical_encoder": bundle.categorical_encoder,
        }
    )
    X = transformer.build_inference_matrix(df_norm)
    model_data = {"imputer": bundle.imputer, "scaler": bundle.scaler}
    X_scaled = apply_model_preprocessing(X, model_data)
    X_imputed = bundle.imputer.transform(X) if bundle.imputer is not None else X

    z_head = getattr(bundle, "z_head", "ensemble")
    z_drivers = getattr(bundle, "z_driver_names", None) or []
    rows: Dict[str, np.ndarray] = {}
    for target_name, model in bundle.models.items():
        if z_head == "quantile_v7" and target_name in Z_TARGETS and z_drivers:
            rows[target_name] = predict_quantile_z(
                model, X_imputed, bundle.feature_names, z_drivers
            )
        else:
            rows[target_name] = model.predict(X_scaled)
    pred_df = pd.DataFrame(rows, index=df.index)

    _apply_side_z(pred_df, getattr(bundle, "side_z_models", None), X_scaled)
    _apply_calibrators(
        pred_df,
        df_norm,
        bundle.left_z_calibrator,
        getattr(bundle, "right_z_calibrator", None),
    )
    _apply_multitask_blend(
        pred_df,
        X_scaled,
        getattr(bundle, "multitask_model", None),
        getattr(bundle, "multitask_blend", None),
        list(bundle.target_names),
    )
    return pred_df


def predict_df(bundle: PredictorBundle, df: pd.DataFrame) -> pd.DataFrame:
    if bundle.mode != "pretrained_adaptive_ensemble":
        X_scaled = bundle.scaler.transform(normalize_dataframe(df)[bundle.feature_names].values)
        rows = {name: model.predict(X_scaled) for name, model in bundle.models.items()}
        return pd.DataFrame(rows, index=df.index)
    return predict_from_bundle(bundle, df)


@dataclass
class RuntimePredictor:
    """Installable production predictor. No sys.path, no scripts.validation."""

    model_path: Path
    payload: Dict[str, Any]
    bundle: PredictorBundle
    transformer: FeatureTransformer = field(repr=False)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        model_path: Path | str | None = None,
    ) -> "RuntimePredictor":
        bundle = bundle_from_payload(payload)
        warn_if_sklearn_mismatch(payload)
        transformer = inference_transformer_from_payload(payload)
        return cls(
            model_path=Path(model_path) if model_path is not None else Path(DEFAULT_MODEL_PATH_STR),
            payload=dict(payload),
            bundle=bundle,
            transformer=transformer,
        )

    @classmethod
    def load(cls, model_path: Path | str | None = None) -> "RuntimePredictor":
        path = Path(model_path or default_model_path())
        warn_if_legacy_model(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Production model not found: {path}. "
                "Run: python scripts/data/train_clinical_honest.py --z-head ensemble"
            )
        payload = joblib.load(path)
        if not isinstance(payload, Mapping):
            raise ArtifactSchemaError(f"Artifact at {path} is not a mapping")
        return cls.from_payload(payload, model_path=path)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return predict_from_bundle(self.bundle, df)

    def predict_row(self, row: Mapping[str, Any]) -> Dict[str, float]:
        df = normalize_dataframe(pd.DataFrame([dict(row)]))
        pred = self.predict_df(df)
        return {col: float(pred[col].iloc[0]) for col in pred.columns}

    def predict_targets(
        self,
        patient_data: Union[Mapping[str, object], pd.DataFrame],
    ) -> Dict[str, float]:
        if isinstance(patient_data, pd.DataFrame):
            pred = self.predict_df(patient_data)
            return {col: float(pred[col].iloc[0]) for col in pred.columns}
        return self.predict_row(patient_data)

    def predict_quantiles(
        self,
        patient_data: Union[Mapping[str, object], pd.DataFrame],
    ) -> Dict[str, Dict[str, float]]:
        quantile = self.payload.get("quantile_model")
        if quantile is None or not getattr(quantile, "fitted_", False):
            return {}
        if isinstance(patient_data, pd.DataFrame):
            patient_df = patient_data
        else:
            patient_df = pd.DataFrame([dict(patient_data)])
        X = self.transformer.build_inference_matrix(normalize_dataframe(patient_df))
        X_scaled = apply_model_preprocessing(
            X, {"imputer": self.bundle.imputer, "scaler": self.bundle.scaler}
        )
        return quantile.predict_all(X_scaled)

    def enrichment_mode(self) -> str:
        return str(self.payload.get("enrichment_mode", "na_trends"))

    def feature_count(self) -> int:
        return len(self.payload.get("feature_names", []))
