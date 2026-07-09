"""Production model loading and prediction for Cases API."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models" / "phase1"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "validation"))

from common import PredictorBundle, predict_df  # noqa: E402
from src.features.na_trend_features import NaTrendStore  # noqa: E402
from src.features.phase1_schema import BASE_FEATURES, normalize_dataframe  # noqa: E402

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "adaptive_ensemble_clinical_honest.pkl"
MODEL_ID = "adaptive_ensemble_clinical_honest"


@dataclass
class ProductionPredictor:
    model_path: Path
    payload: Dict[str, Any]
    bundle: Any

    @classmethod
    def load(cls, model_path: Path | None = None) -> ProductionPredictor:
        path = Path(model_path or DEFAULT_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Production model not found: {path}. "
                "Run: python scripts/data/train_clinical_honest.py --z-head ensemble"
            )
        payload = joblib.load(path)
        store_dict = payload.get("na_trend_store")
        bundle = PredictorBundle(
            mode="pretrained_adaptive_ensemble",
            feature_names=list(payload["feature_names"]),
            target_names=list(payload.get("target_names", payload["models"].keys())),
            scaler=payload["scaler"],
            models=payload["models"],
            imputer=payload.get("imputer"),
            left_z_calibrator=payload.get("left_z_calibrator"),
            right_z_calibrator=payload.get("right_z_calibrator"),
            side_z_models=payload.get("side_z_models"),
            multitask_model=payload.get("multitask_model"),
            multitask_blend=payload.get("multitask_blend"),
            quantile_model=payload.get("quantile_model"),
            z_head=payload.get("z_head", "ensemble"),
            z_driver_names=payload.get("z_driver_names"),
        )
        bundle.enrichment_mode = payload.get("enrichment_mode", "na_trends")
        bundle.na_trend_store = store_dict
        return cls(model_path=path, payload=payload, bundle=bundle)

    def predict_row(self, row: Dict[str, Any]) -> Dict[str, float]:
        df = normalize_dataframe(pd.DataFrame([row]))
        pred = predict_df(self.bundle, df)
        return {col: float(pred[col].iloc[0]) for col in pred.columns}

    def enrichment_mode(self) -> str:
        return str(self.payload.get("enrichment_mode", "na_trends"))

    def feature_count(self) -> int:
        return len(self.payload.get("feature_names", []))


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
