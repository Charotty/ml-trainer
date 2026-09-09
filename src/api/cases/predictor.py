"""Production model loading and prediction for Cases API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.features.phase1_schema import BASE_FEATURES, normalize_dataframe
from src.models.runtime import RuntimePredictor, default_model_path

DEFAULT_MODEL_PATH = default_model_path()
MODEL_ID = "adaptive_ensemble_clinical_honest"
MAX_ABS_DELTA_MM = 80.0


def assess_prediction_sanity(
    predictions: Dict[str, float],
    *,
    max_abs_mm: float = MAX_ABS_DELTA_MM,
) -> tuple[bool, List[str]]:
    """Return (ok, warnings) for clinically implausible displacement magnitudes."""
    warnings: List[str] = []
    for name, value in predictions.items():
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
    return (len(warnings) == 0), warnings


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
