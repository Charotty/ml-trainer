"""Production model loading and prediction for Cases API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.features.laterality import (
    LATERALITY_ABSENT,
    LATERALITY_NOT_ASSESSED,
    laterality_from_row,
    targets_for_side,
)
from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES, normalize_dataframe
from src.models.runtime import RuntimePredictor, default_model_path

DEFAULT_MODEL_PATH = default_model_path()
MODEL_ID = "adaptive_ensemble_clinical_honest"
MAX_ABS_DELTA_MM = 80.0
IMPUTED_IMPORTANCE_THRESHOLD = 0.40


def _estimator_importances(model: Any, n_features: int) -> Optional[np.ndarray]:
    for attr in ("feature_importances_", "coef_"):
        value = getattr(model, attr, None)
        if value is None:
            continue
        arr = np.abs(np.asarray(value, dtype=float)).reshape(-1)
        if arr.shape[0] == n_features:
            return arr / arr.sum() if arr.sum() > 0 else arr
    estimators = getattr(model, "estimators_", None) or getattr(model, "named_estimators_", None)
    if isinstance(estimators, dict):
        estimators = list(estimators.values())
    if estimators:
        parts = [_estimator_importances(est, n_features) for est in estimators]
        parts = [p for p in parts if p is not None]
        if parts:
            return np.mean(parts, axis=0)
    return None


def imputed_importance_share(
    *,
    payload: Mapping[str, Any] | None,
    all_features: Mapping[str, Any] | None,
    target: str,
) -> Optional[float]:
    """Share of tree importance sitting on features that are NaN at serve time."""
    if not payload or not all_features:
        return None
    names = list(payload.get("feature_names") or [])
    models = payload.get("models") or {}
    model = models.get(target)
    if not names or model is None:
        return None
    imp = _estimator_importances(model, len(names))
    if imp is None:
        return None
    nan_mask = np.array(
        [
            val is None or (isinstance(val, float) and (not np.isfinite(val) or pd.isna(val)))
            for val in (all_features.get(name) for name in names)
        ],
        dtype=bool,
    )
    return float(imp[nan_mask].sum())


def assess_prediction_sanity(
    predictions: Dict[str, Any],
    *,
    max_abs_mm: float = MAX_ABS_DELTA_MM,
    all_features: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    imputed_share_threshold: float = IMPUTED_IMPORTANCE_THRESHOLD,
) -> tuple[bool, List[str]]:
    """Return (ok, warnings) for implausible magnitudes and imputed-feature load."""
    warnings: List[str] = []
    for name, value in predictions.items():
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            warnings.append(f"{name}: значение не является числом")
            continue
        if not np.isfinite(number):
            warnings.append(f"{name}: нечисловое значение ({value})")
        elif abs(number) > max_abs_mm:
            warnings.append(
                f"{name}: |Δ|={abs(number):.1f} мм превышает порог {max_abs_mm:.0f} мм"
            )
        share = imputed_importance_share(
            payload=payload, all_features=all_features, target=name
        )
        if share is not None and share >= imputed_share_threshold:
            warnings.append(
                f"{name}: {share:.0%} важности признаков импьютированы "
                f"(порог {imputed_share_threshold:.0%})"
            )
    return (len(warnings) == 0), warnings


def apply_laterality_gate(
    predictions: Dict[str, float],
    row: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, str], List[str], List[str]]:
    """Null absent-side targets; warn on not_assessed. Does not invent a kidney."""
    flags = laterality_from_row(row)
    out: Dict[str, Any] = dict(predictions)
    withheld: List[str] = []
    notes: List[str] = []
    for side, status in flags.items():
        if status == LATERALITY_ABSENT:
            for target in targets_for_side(side):
                out[target] = None
                withheld.append(target)
            notes.append(
                f"kidney_{side}: отсутствует — прогноз по стороне не выдаётся"
            )
        elif status == LATERALITY_NOT_ASSESSED:
            notes.append(
                f"kidney_{side}: не подтверждена сегментацией — низкая достоверность"
            )
    for target in TARGET_NAMES:
        out.setdefault(target, predictions.get(target))
    return out, flags, withheld, notes


def finalize_case_predictions(
    *,
    row: Mapping[str, Any],
    predictions: Dict[str, float],
    all_features: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], bool, List[str], Dict[str, str], List[str]]:
    gated, flags, withheld, notes = apply_laterality_gate(predictions, row)
    sanity_ok, warnings = assess_prediction_sanity(
        gated, all_features=all_features, payload=payload
    )
    return gated, sanity_ok, notes + warnings, flags, withheld


@dataclass
class ProductionPredictor:
    runtime: RuntimePredictor

    @property
    def model_path(self) -> Path:
        return self.runtime.model_path

    @property
    def payload(self) -> Dict[str, Any]:
        return self.runtime.payload

    @property
    def bundle(self):
        return self.runtime.bundle

    @classmethod
    def load(cls, model_path: Path | None = None) -> ProductionPredictor:
        return cls(runtime=RuntimePredictor.load(model_path))

    def predict_row(self, row: Dict[str, Any]) -> Dict[str, float]:
        return self.runtime.predict_row(row)

    def enrichment_mode(self) -> str:
        return self.runtime.enrichment_mode()

    def feature_count(self) -> int:
        return self.runtime.feature_count()


def compute_feature_coverage(
    all_features: Dict[str, Any],
    feature_names: List[str],
) -> tuple[float, List[str]]:
    missing: List[str] = []
    present = 0
    for name in feature_names:
        val = all_features.get(name)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            missing.append(name)
        else:
            present += 1
    pct = 100.0 * present / len(feature_names) if feature_names else 0.0
    return pct, missing


def base_features_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    df = normalize_dataframe(pd.DataFrame([row]))
    return {col: _json_safe(df[col].iloc[0]) for col in BASE_FEATURES if col in df.columns}


def _json_safe(val: Any) -> Any:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(val, "item"):
        return val.item()
    return val
