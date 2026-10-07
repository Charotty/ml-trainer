"""Feature engineering for a single case row."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

from src.models.ensemble import AdaptiveEnsembleTrainer  # noqa: E402
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
from src.features.displacement_axis_features import ANATOMICAL_FEATURES  # noqa: E402
from src.features.laterality import (  # noqa: E402
    LATERALITY_COLUMNS,
    MASK_STATUS_COLUMNS,
    attach_laterality,
    normalize_laterality,
)
from src.features.phase1_schema import (  # noqa: E402
    BASE_FEATURES,
    CLINICAL_DEMOGRAPHIC_FEATURES,
    normalize_dataframe,
)
from src.models.ensemble.features import inference_transformer_from_payload  # noqa: E402

from .predictor import _json_safe, compute_feature_coverage  # noqa: E402

CLINICAL_REFERENCE_PATH = REPO_ROOT / "data" / "vybor_from_xlsx.csv"

# Supine clinical extras that Z/Y heads and SideZCalibrator read from base.
# Do not persist *_delta_span_* — those are lateral-derived leakage.
SAFE_QA_CLINICAL_COLUMNS: tuple[str, ...] = (
    "abd_wall_thickness_mm",
    "lumbar_lordosis_deg",
    "s1_plate_tilt_deg",
    "kidney_left_z_span_supine_mm",
    "kidney_right_z_span_supine_mm",
    "kidney_left_y_span_supine_mm",
    "kidney_right_y_span_supine_mm",
)
LEAKAGE_SPAN_COLUMNS: tuple[str, ...] = (
    "kidney_left_z_delta_span_mm",
    "kidney_right_z_delta_span_mm",
    "kidney_left_y_delta_span_mm",
    "kidney_right_y_delta_span_mm",
)

_KIDNEY_POLE_PREFIXES = (
    "kidney_left_upper_",
    "kidney_left_lower_",
    "kidney_left_middle_",
    "kidney_right_upper_",
    "kidney_right_lower_",
    "kidney_right_middle_",
)
_ABSOLUTE_CENTER_KEYS = (
    "kidney_left_center_x",
    "kidney_left_center_y",
    "kidney_left_center_z",
    "kidney_right_center_x",
    "kidney_right_center_y",
    "kidney_right_center_z",
)
_FRAME_KEYS = (
    "patient_position",
    "scan_position",
    "feature_frame",
    "geometry_ood_x",
    "geometry_ood_x_reasons",
    "mid_sagittal_x",
    "mid_sagittal_x_source",
    "solitary_kidney_side",
    *LATERALITY_COLUMNS,
    *MASK_STATUS_COLUMNS,
)
_PASS_THROUGH_FEATURE_KEYS = list(
    dict.fromkeys(
        [
            *BASE_FEATURES,
            *CLINICAL_DEMOGRAPHIC_FEATURES,
            *ANATOMICAL_FEATURES,
            *SAFE_QA_CLINICAL_COLUMNS,
            *_ABSOLUTE_CENTER_KEYS,
            *_FRAME_KEYS,
        ]
    )
)
PATCH_ALLOWED_COLUMNS = (
    set(BASE_FEATURES)
    | set(CLINICAL_DEMOGRAPHIC_FEATURES)
    | set(ANATOMICAL_FEATURES)
    | set(SAFE_QA_CLINICAL_COLUMNS)
    | set(LATERALITY_COLUMNS)
)


def persistable_base_columns(df: pd.DataFrame | None = None) -> List[str]:
    """Columns written to base_features.json (predict reads this file)."""
    cols = list(_PASS_THROUGH_FEATURE_KEYS)
    if df is not None:
        pole_cols = [
            col
            for col in df.columns
            if any(col.startswith(prefix) for prefix in _KIDNEY_POLE_PREFIXES)
        ]
        cols.extend(pole_cols)
    blocked = set(LEAKAGE_SPAN_COLUMNS)
    return [c for c in dict.fromkeys(cols) if c not in blocked]


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
    payload: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, List[str]]:
    """Return base_features, all_features, coverage_pct, missing."""
    # Normalize first (aliases / abs→rel), then overwrite geometry with the
    # clinical Excel/Vybor frame so LPS vertebral-spine distances do not leak in.
    df = normalize_dataframe(pd.DataFrame([base_row]))
    row = attach_laterality(harmonize_ct_to_clinical_frame(df.iloc[0].to_dict()))
    row = sanitize_body_size_for_clinical_model(row)
    df = _apply_ct_enrichment(pd.DataFrame([row]))
    if payload is not None:
        transformer = inference_transformer_from_payload(payload)
    else:
        store = NaTrendStore.from_dict(na_trend_store) if na_trend_store else NaTrendStore.fit(include_kits=False)
        transformer = AdaptiveEnsembleTrainer(
            enrichment_mode=enrichment_mode,
            na_trend_store=store,
        )
        transformer.feature_names = list(feature_names)
    matrix = transformer.build_inference_matrix(df)
    all_features: Dict[str, Any] = {}
    for i, name in enumerate(feature_names):
        all_features[name] = _json_safe(matrix[0, i])
    keep_cols = persistable_base_columns(df)
    base_out = {
        col: _json_safe(df[col].iloc[0]) for col in keep_cols if col in df.columns
    }
    coverage, missing = compute_feature_coverage(all_features, feature_names)
    return base_out, all_features, coverage, missing


def merge_base_features(
    existing: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(existing)
    for key, val in overrides.items():
        if key in LEAKAGE_SPAN_COLUMNS:
            continue
        if key in LATERALITY_COLUMNS:
            merged[key] = normalize_laterality(val)
            continue
        if key in PATCH_ALLOWED_COLUMNS:
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
    for key in (*LATERALITY_COLUMNS, *MASK_STATUS_COLUMNS, "solitary_kidney_side"):
        if key in extracted:
            row[key] = _json_safe(extracted[key])
    return attach_laterality(row)


def rebuild_features_from_extraction(storage: Any, case_id: str, predictor: Any) -> Tuple[Dict[str, Any], Dict[str, Any], float, List[str]]:
    """Rebuild base/features from extraction_raw.json without re-running TotalSegmentator."""
    raw = storage.read_json_artifact(case_id, "extraction_raw.json")
    if not raw:
        raise FileNotFoundError(f"extraction_raw.json missing for case {case_id}")
    base_row = extraction_row_to_base_features(raw)
    base_out, all_features, coverage, missing = build_features_from_base(
        base_row,
        feature_names=list(predictor.payload["feature_names"]),
        enrichment_mode=predictor.enrichment_mode(),
        na_trend_store=predictor.payload.get("na_trend_store"),
        payload=predictor.payload,
    )
    storage.write_json_artifact(case_id, "base_features.json", base_out)
    storage.write_json_artifact(
        case_id,
        "features.json",
        {
            "all_features": all_features,
            "coverage_pct": coverage,
            "missing_features": missing,
        },
    )
    storage.update_meta(
        case_id,
        status="features_ready",
        coverage_pct=coverage,
        message=f"Features rebuilt (coverage {coverage:.1f}%)",
    )
    return base_out, all_features, coverage, missing
