"""Feature engineering for a single case row."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models" / "phase1"))

from adaptive_ensemble import AdaptiveEnsembleTrainer  # noqa: E402
from src.features.ct_external_enrichment import (  # noqa: E402
    SPAN_COLS,
    compute_anatomical_extras,
    compute_spans_from_upper_lower,
    fill_clinical_drivers_from_reference,
)
from src.features.ct_geometry import (  # noqa: E402
    harmonize_ct_to_clinical_frame,
    sanitize_body_size_for_clinical_model,
)
from src.features.na_trend_features import NaTrendStore  # noqa: E402
from src.features.phase1_schema import (  # noqa: E402
    BASE_FEATURES,
    CLINICAL_DEMOGRAPHIC_FEATURES,
    normalize_dataframe,
)

from .predictor import _json_safe, compute_feature_coverage  # noqa: E402

CLINICAL_REFERENCE_PATH = REPO_ROOT / "data" / "vybor_from_xlsx.csv"

_KIDNEY_POLE_PREFIXES = (
    "kidney_left_upper_",
    "kidney_left_lower_",
    "kidney_right_upper_",
    "kidney_right_lower_",
)
_ABSOLUTE_CENTER_KEYS = (
    "kidney_left_center_x",
    "kidney_left_center_y",
    "kidney_left_center_z",
    "kidney_right_center_x",
    "kidney_right_center_y",
    "kidney_right_center_z",
)
_PASS_THROUGH_FEATURE_KEYS = list(
    dict.fromkeys(
        [
            *BASE_FEATURES,
            *CLINICAL_DEMOGRAPHIC_FEATURES,
            *_ABSOLUTE_CENTER_KEYS,
            "patient_position",
            "scan_position",
            "feature_frame",
        ]
    )
)


def _apply_ct_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    out = compute_spans_from_upper_lower(df)
    out = compute_anatomical_extras(out)
    if CLINICAL_REFERENCE_PATH.exists() and any(
        col not in out.columns or out[col].isna().all() for col in SPAN_COLS
    ):
        out = fill_clinical_drivers_from_reference(out, pd.read_csv(CLINICAL_REFERENCE_PATH))
    return out


def build_features_from_base(
    base_row: Dict[str, Any],
    *,
    feature_names: List[str],
    enrichment_mode: str = "na_trends",
    na_trend_store: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, List[str]]:
    """Return base_features, all_features, coverage_pct, missing."""
    # Normalize first (aliases / abs→rel), then overwrite geometry with the
    # clinical Excel/Vybor frame so LPS vertebral-spine distances do not leak in.
    df = normalize_dataframe(pd.DataFrame([base_row]))
    row = harmonize_ct_to_clinical_frame(df.iloc[0].to_dict())
    row = sanitize_body_size_for_clinical_model(row)
    df = _apply_ct_enrichment(pd.DataFrame([row]))
    store = NaTrendStore.from_dict(na_trend_store) if na_trend_store else NaTrendStore.fit(include_kits=False)
    trainer = AdaptiveEnsembleTrainer(
        enrichment_mode=enrichment_mode,
        na_trend_store=store,
    )
    trainer.feature_names = list(feature_names)
    matrix = trainer.build_inference_matrix(df)
    all_features: Dict[str, Any] = {}
    for i, name in enumerate(feature_names):
        all_features[name] = _json_safe(matrix[0, i])
    keep_cols = list(
        dict.fromkeys([*BASE_FEATURES, *CLINICAL_DEMOGRAPHIC_FEATURES, "feature_frame"])
    )
    base_out = {
        col: _json_safe(df[col].iloc[0]) for col in keep_cols if col in df.columns
    }
    coverage, missing = compute_feature_coverage(all_features, feature_names)
    return base_out, all_features, coverage, missing


def merge_base_features(
    existing: Dict[str, Any],
    overrides: Dict[str, float],
) -> Dict[str, Any]:
    merged = dict(existing)
    allowed = set(BASE_FEATURES) | set(CLINICAL_DEMOGRAPHIC_FEATURES)
    for key, val in overrides.items():
        if key in allowed:
            merged[key] = float(val)
    return merged


def extraction_row_to_base_features(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Map extract_from_dicom row dict to model inputs (base + clinical + absolutes)."""
    df = normalize_dataframe(pd.DataFrame([extracted]))
    pole_cols = [
        col
        for col in df.columns
        if any(col.startswith(prefix) for prefix in _KIDNEY_POLE_PREFIXES)
    ]
    cols = list(dict.fromkeys([*_PASS_THROUGH_FEATURE_KEYS, *pole_cols]))
    row = {col: _json_safe(df[col].iloc[0]) for col in cols if col in df.columns}
    # Preserve absolute LPS centers even if normalize renamed nothing.
    for key in _ABSOLUTE_CENTER_KEYS:
        if key in extracted and key not in row:
            row[key] = _json_safe(extracted[key])
    for key in CLINICAL_DEMOGRAPHIC_FEATURES:
        if key in extracted and (key not in row or row[key] is None):
            row[key] = _json_safe(extracted[key])
    return row
