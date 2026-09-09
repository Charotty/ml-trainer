"""Dump and validate model-artifact manifests (provenance / model cards).

Block 0 freeze: every candidate ``.pkl`` is paired with a JSON card that records
hashes, features, ensemble weights, and an explicit ban on mixing historical
OOF MAE figures (8.40 / 8.49 / 8.52 mm) until corrected nested OOF exists.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

ARTIFACT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = frozenset({"1.0.0", "1.1.0"})
CURRENT_ARTIFACT_SCHEMA_VERSION = "1.1.0"

OOF_METRICS_DO_NOT_MIX_NOTE = (
    "Historical GKF-OOF MAE figures 8.40 mm, 8.49 mm, and 8.52 mm were produced "
    "under mixed evaluation protocols (different feature sets, projection vs "
    "na_trends, non-nested OOF). They MUST NOT be mixed, compared across "
    "candidates, or cited as production evidence until a corrected nested "
    "GroupKFold OOF exists for each artifact."
)

HISTORICAL_OOF_MAE_MM_DO_NOT_MIX: tuple[float, float, float] = (8.40, 8.49, 8.52)

# Compact hyperparameter keys worth persisting on the card.
_ESTIMATOR_PARAM_KEYS = frozenset(
    {
        "n_estimators",
        "max_depth",
        "max_features",
        "min_samples_leaf",
        "min_samples_split",
        "n_jobs",
        "random_state",
        "alpha",
        "max_iter",
        "learning_rate",
        "subsample",
        "loss",
    }
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "artifact_schema_version",
    "schema_version",
    "feature_schema_version",
    "candidate_id",
    "status",
    "production_winner",
    "artifact_filename",
    "sha256",
    "size_bytes",
    "feature_count",
    "feature_names",
    "ensemble_weights",
    "estimator_types",
    "calibrators_present",
    "training_meta",
    "python_version",
    "sklearn_version",
    "git_commit",
    "run_id",
    "oof_metrics_note",
    "historical_oof_mae_mm_do_not_mix",
)

_SHA256_HEX_LEN = 64


class ManifestValidationError(ValueError):
    """Raised when a model-card JSON document fails schema checks."""


def sha256_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Return uppercase SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _estimator_param_subset(estimator: Any) -> dict[str, Any]:
    if estimator is None or not hasattr(estimator, "get_params"):
        return {"type": type(estimator).__name__}
    params = estimator.get_params(deep=False)
    subset = {k: _jsonable(params[k]) for k in sorted(params) if k in _ESTIMATOR_PARAM_KEYS}
    subset["type"] = type(estimator).__name__
    return subset


def extract_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pull card fields that live inside a joblib payload (no file hashing)."""
    feature_names = [str(name) for name in list(payload.get("feature_names") or [])]
    models = payload.get("models") or {}
    ensemble_weights: dict[str, Any] = {}
    estimator_types: dict[str, Any] = {}
    estimator_params: dict[str, Any] = {}

    for target, model in models.items():
        names: list[str] = []
        named = getattr(model, "named_estimators", None)
        estimators_spec = getattr(model, "estimators", None)
        if named:
            names = list(named.keys())
        elif estimators_spec:
            names = [str(item[0]) for item in estimators_spec]

        fitted = list(getattr(model, "estimators_", None) or [])
        types = [type(est).__name__ for est in fitted]
        if names and len(names) == len(types):
            estimator_types[str(target)] = dict(zip(names, types))
        elif names:
            estimator_types[str(target)] = {
                name: type(est).__name__
                for name, est in (named.items() if named else [])
            } or names
        else:
            estimator_types[str(target)] = types

        weights = getattr(model, "weights", None)
        if weights is not None and names and len(names) == len(list(weights)):
            ensemble_weights[str(target)] = {
                name: float(weight) for name, weight in zip(names, weights)
            }
        elif weights is not None:
            ensemble_weights[str(target)] = [float(weight) for weight in weights]
        else:
            ensemble_weights[str(target)] = None

        source_estimators = named or (
            {item[0]: item[1] for item in estimators_spec} if estimators_spec else {}
        )
        if source_estimators:
            estimator_params[str(target)] = {
                str(name): _estimator_param_subset(est)
                for name, est in source_estimators.items()
            }

    calibrators_present = {
        "left_z_calibrator": payload.get("left_z_calibrator") is not None,
        "right_z_calibrator": payload.get("right_z_calibrator") is not None,
    }
    training_meta = _jsonable(payload.get("training_meta") or {})
    na_store = payload.get("na_trend_store")
    na_meta: dict[str, Any] | None = None
    if isinstance(na_store, Mapping):
        na_meta = {
            "spine_rows": na_store.get("spine_rows"),
            "boku_rows": na_store.get("boku_rows"),
            "kits_rows": na_store.get("kits_rows"),
            "include_kits": na_store.get("include_kits"),
        }

    scaler = payload.get("scaler")
    imputer = payload.get("imputer")
    return {
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "target_names": [str(name) for name in list(payload.get("target_names") or models.keys())],
        "ensemble_weights": ensemble_weights,
        "estimator_types": estimator_types,
        "estimator_params": estimator_params,
        "calibrators_present": calibrators_present,
        "training_meta": training_meta,
        "z_head": payload.get("z_head"),
        "enrichment_mode": payload.get("enrichment_mode"),
        "z_driver_names": list(payload.get("z_driver_names") or []),
        "preprocessor": {
            "scaler_type": type(scaler).__name__ if scaler is not None else None,
            "imputer_type": type(imputer).__name__ if imputer is not None else None,
            "imputer_strategy": getattr(imputer, "strategy", None) if imputer is not None else None,
        },
        "na_trend_store": na_meta,
    }


def build_manifest(
    *,
    candidate_id: str,
    artifact_filename: str,
    sha256: str,
    size_bytes: int,
    payload_fields: Mapping[str, Any],
    git_commit: str,
    python_version: str,
    sklearn_version: str,
    run_id: str = "PLACEHOLDER",
    status: str = "candidate",
    production_winner: bool = False,
    joblib_version: str | None = None,
    git_blob: str | None = None,
    source: str | None = None,
    alias_path: str = "models/adaptive_ensemble_clinical_honest.pkl",
    feature_schema_version: str = "phase1_v1",
    data_hashes: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a v1.0.0 model card. Does not mark either candidate as winner."""
    card: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": feature_schema_version,
        "candidate_id": candidate_id,
        "status": status,
        "production_winner": bool(production_winner),
        "research_only": True,
        "artifact_filename": artifact_filename,
        "alias_path": alias_path,
        "source": source,
        "sha256": str(sha256).upper(),
        "size_bytes": int(size_bytes),
        "git_commit": git_commit,
        "git_blob": git_blob,
        "run_id": run_id,
        "python_version": python_version,
        "sklearn_version": sklearn_version,
        "joblib_version": joblib_version,
        "data_hashes": dict(data_hashes or {}),
        "folds": None,
        "oof_predictions": None,
        "aggregated_metrics": None,
        "subgroup_metrics": None,
        "clinical_metrics": None,
        "tail_metrics": None,
        "worst_cases": None,
        "conformal": None,
        "oof_protocol": None,
        "oof_status": "not_computed",
        "oof_weight_mode": None,
        "historical_oof_mae_mm_do_not_mix": list(HISTORICAL_OOF_MAE_MM_DO_NOT_MIX),
        "oof_metrics_note": OOF_METRICS_DO_NOT_MIX_NOTE,
    }
    card.update(payload_fields)
    if extra:
        card.update(extra)
    return card


def oof_placeholder(*, protocol: str | None = None) -> dict[str, Any]:
    """Schema-compatible empty OOF block (no 87-patient retrain required)."""
    return {
        "folds": None,
        "oof_predictions": None,
        "aggregated_metrics": None,
        "subgroup_metrics": None,
        "clinical_metrics": None,
        "tail_metrics": None,
        "worst_cases": None,
        "conformal": None,
        "oof_protocol": protocol,
        "oof_status": "not_computed",
        "oof_weight_mode": None,
    }


def apply_oof_to_manifest(
    card: dict[str, Any],
    oof_fields: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Copy nested-OOF fields onto a model card. Missing payload stays placeholder."""
    if not oof_fields:
        card.setdefault("oof_status", "not_computed")
        card.setdefault("folds", None)
        card.setdefault("oof_predictions", None)
        card.setdefault("aggregated_metrics", None)
        card.setdefault("subgroup_metrics", None)
        card.setdefault("clinical_metrics", None)
        card.setdefault("tail_metrics", None)
        card.setdefault("worst_cases", None)
        card.setdefault("conformal", None)
        card.setdefault("oof_protocol", None)
        card.setdefault("oof_weight_mode", None)
        return card
    card["folds"] = oof_fields.get("folds")
    card["oof_predictions"] = oof_fields.get("oof_predictions")
    card["aggregated_metrics"] = oof_fields.get("aggregated_metrics")
    card["oof_protocol"] = oof_fields.get("oof_protocol")
    card["oof_status"] = oof_fields.get("oof_status", "computed")
    card["oof_weight_mode"] = oof_fields.get("oof_weight_mode")
    for key in (
        "subgroup_metrics",
        "clinical_metrics",
        "tail_metrics",
        "worst_cases",
        "conformal",
    ):
        if key in oof_fields:
            card[key] = oof_fields.get(key)
    return card


def _require_type(data: Mapping[str, Any], field: str, expected: type | tuple[type, ...]) -> None:
    value = data[field]
    if not isinstance(value, expected):
        type_name = (
            expected.__name__
            if isinstance(expected, type)
            else " | ".join(item.__name__ for item in expected)
        )
        raise ManifestValidationError(
            f"Field '{field}' must be {type_name}, got {type(value).__name__}"
        )


def validate_manifest(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a model-card mapping. Returns ``data`` on success."""
    if not isinstance(data, Mapping):
        raise ManifestValidationError("Manifest must be a JSON object")

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ManifestValidationError("Missing required fields: " + ", ".join(missing))

    schema_ver = data["artifact_schema_version"]
    if schema_ver not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ManifestValidationError(
            f"Unsupported artifact_schema_version {schema_ver!r}; "
            f"supported {sorted(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS)}"
        )
    if data["schema_version"] != schema_ver:
        raise ManifestValidationError(
            f"schema_version {data['schema_version']!r} must match "
            f"artifact_schema_version {schema_ver!r}"
        )
    _require_type(data, "feature_schema_version", str)
    if not str(data["feature_schema_version"]).strip():
        raise ManifestValidationError("feature_schema_version must be non-empty")
    if schema_ver == "1.1.0":
        estimator = data.get("estimator_type")
        if not isinstance(estimator, str) or not estimator.strip():
            raise ManifestValidationError(
                "artifact_schema_version 1.1.0 requires estimator_type"
            )
        contract = data.get("preprocessing_contract", data.get("preprocessor"))
        if not isinstance(contract, dict):
            raise ManifestValidationError(
                "artifact_schema_version 1.1.0 requires preprocessing_contract"
            )

    _require_type(data, "candidate_id", str)
    _require_type(data, "status", str)
    _require_type(data, "production_winner", bool)
    _require_type(data, "artifact_filename", str)
    _require_type(data, "sha256", str)
    _require_type(data, "size_bytes", int)
    _require_type(data, "feature_count", int)
    _require_type(data, "feature_names", list)
    _require_type(data, "ensemble_weights", (dict, list))
    _require_type(data, "estimator_types", (dict, list))
    _require_type(data, "calibrators_present", dict)
    _require_type(data, "training_meta", dict)
    _require_type(data, "python_version", str)
    _require_type(data, "sklearn_version", str)
    _require_type(data, "git_commit", str)
    _require_type(data, "run_id", str)
    _require_type(data, "oof_metrics_note", str)
    _require_type(data, "historical_oof_mae_mm_do_not_mix", list)

    sha = data["sha256"].strip()
    if len(sha) != _SHA256_HEX_LEN or any(ch not in "0123456789abcdefABCDEF" for ch in sha):
        raise ManifestValidationError("sha256 must be a 64-character hexadecimal digest")

    if data["size_bytes"] < 0:
        raise ManifestValidationError("size_bytes must be >= 0")

    names = data["feature_names"]
    if any(not isinstance(name, str) for name in names):
        raise ManifestValidationError("feature_names must be a list of strings")
    if data["feature_count"] != len(names):
        raise ManifestValidationError(
            f"feature_count ({data['feature_count']}) != len(feature_names) ({len(names)})"
        )

    cals = data["calibrators_present"]
    for key in ("left_z_calibrator", "right_z_calibrator"):
        if key not in cals:
            raise ManifestValidationError(f"calibrators_present missing '{key}'")
        if not isinstance(cals[key], bool):
            raise ManifestValidationError(f"calibrators_present.{key} must be bool")

    note = data["oof_metrics_note"]
    for token in ("8.40", "8.49", "8.52", "MUST NOT"):
        if token not in note:
            raise ManifestValidationError(
                "oof_metrics_note must explicitly forbid mixing "
                "8.40/8.49/8.52 until corrected nested OOF exists "
                f"(missing {token!r})"
            )

    historical = data["historical_oof_mae_mm_do_not_mix"]
    expected = list(HISTORICAL_OOF_MAE_MM_DO_NOT_MIX)
    coerced = [float(item) for item in historical]
    if coerced != expected:
        raise ManifestValidationError(
            f"historical_oof_mae_mm_do_not_mix must be {expected}, got {historical}"
        )

    if not data["candidate_id"].strip():
        raise ManifestValidationError("candidate_id must be non-empty")
    if not data["run_id"].strip():
        raise ManifestValidationError("run_id must be non-empty")
    if not data["git_commit"].strip():
        raise ManifestValidationError("git_commit must be non-empty")

    return data


def dump_manifest(
    path: Path | str,
    data: Mapping[str, Any],
    *,
    validate: bool = True,
) -> Path:
    """Write a validated model-card JSON file (UTF-8, trailing newline)."""
    payload: Mapping[str, Any] = dict(data)
    if validate:
        validate_manifest(payload)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def load_manifest(path: Path | str, *, validate: bool = True) -> dict[str, Any]:
    """Load a model-card JSON file and optionally validate it."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ManifestValidationError("Manifest JSON root must be an object")
    if validate:
        validate_manifest(raw)
    return raw


def hash_data_file(path: Path | str, *, label: str | None = None) -> dict[str, Any]:
    """Hash a dataset file if present; record absence otherwise."""
    file_path = Path(path)
    key = label or str(file_path).replace("\\", "/")
    if not file_path.is_file():
        return {
            key: {
                "present": False,
                "sha256": None,
                "size_bytes": None,
            }
        }
    return {
        key: {
            "present": True,
            "sha256": sha256_file(file_path),
            "size_bytes": file_path.stat().st_size,
        }
    }


def iter_required_fields() -> Iterable[str]:
    return REQUIRED_FIELDS


def is_valid_manifest(data: Mapping[str, Any]) -> bool:
    try:
        validate_manifest(data)
    except ManifestValidationError:
        return False
    return True


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CURRENT_ARTIFACT_SCHEMA_VERSION",
    "HISTORICAL_OOF_MAE_MM_DO_NOT_MIX",
    "ManifestValidationError",
    "OOF_METRICS_DO_NOT_MIX_NOTE",
    "SUPPORTED_ARTIFACT_SCHEMA_VERSIONS",
    "apply_oof_to_manifest",
    "build_manifest",
    "dump_manifest",
    "extract_from_payload",
    "hash_data_file",
    "is_valid_manifest",
    "iter_required_fields",
    "load_manifest",
    "oof_placeholder",
    "sha256_bytes",
    "sha256_file",
    "validate_manifest",
]
