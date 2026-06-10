"""Canonical Phase 1 feature schema and cross-source normalization.

Single source of truth for:
  - base / engineered / cross feature names used by AdaptiveEnsembleTrainer
  - target column names
  - alias mapping from DICOM extractors, Vybor, KiTS19, DICOMS integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

SCHEMA_VERSION = "phase1_v1"

BASE_FEATURES: List[str] = [
    "kidney_left_center_x_rel",
    "kidney_left_center_y_rel",
    "kidney_left_center_z_rel",
    "kidney_right_center_x_rel",
    "kidney_right_center_y_rel",
    "kidney_right_center_z_rel",
    "kidney_left_length_mm",
    "kidney_left_volume_cm3",
    "kidney_right_length_mm",
    "kidney_right_volume_cm3",
    "body_width_mm",
    "body_depth_mm",
    "body_area_mm2",
    "kidney_left_to_spine_distance",
    "kidney_right_to_spine_distance",
    "kidney_left_to_body_center_distance",
    "kidney_right_to_body_center_distance",
    "spine_center_x",
    "spine_center_y",
    "spine_center_z",
    "body_com_x",
    "body_com_y",
    "body_com_z",
]

ENGINEERED_FEATURES: List[str] = [
    "body_ratio",
    "kidney_distance_lr",
    "kidney_left_volume_norm",
    "kidney_right_volume_norm",
    "kidney_left_length_norm",
    "kidney_right_length_norm",
    "volume_asymmetry",
    "length_asymmetry",
    "spine_distance_asymmetry",
    "body_center_asymmetry",
    "kidney_left_to_spine_ratio",
    "kidney_right_to_spine_ratio",
    "patient_position_encoded",
]

CROSS_FEATURES: List[str] = [
    "body_volume_estimated",
    "kidney_left_density_ratio",
    "kidney_right_density_ratio",
    "spine_to_body_ratio_x",
    "spine_to_body_ratio_y",
    "body_com_to_spine_distance",
    "kidney_left_spine_interaction",
    "kidney_right_spine_interaction",
    "body_size_index",
    "kidney_position_index_left",
    "kidney_position_index_right",
    "volume_to_area_ratio_left",
    "volume_to_area_ratio_right",
    "relative_volume_sum",
    "kidney_separation_angle",
]

TARGET_NAMES: List[str] = [
    "kidney_left_delta_x",
    "kidney_left_delta_y",
    "kidney_left_delta_z",
    "kidney_right_delta_x",
    "kidney_right_delta_y",
    "kidney_right_delta_z",
]

# Stored in some datasets but never passed as raw model inputs.
OPTIONAL_METADATA_COLUMNS: List[str] = [
    "scan_position",
    "patient_position",
    "sex",
    "age",
    "bmi",
]

PATIENT_POSITION_ENCODING: Dict[str, int] = {
    "HFS": 1,
    "FFS": 2,
    "HFP": 3,
    "FFP": 4,
    "SUPINE": 1,
    "supine": 1,
    "LATERAL": 5,
    "lateral": 5,
}

# alias -> canonical (applied only when canonical column is missing)
DIRECT_ALIASES: Dict[str, str] = {
    "body_com_x_mm": "body_com_x",
    "body_com_y_mm": "body_com_y",
    "body_com_z_mm": "body_com_z",
    "spine_center_x_mm": "spine_center_x",
    "spine_center_y_mm": "spine_center_y",
    "spine_center_z_mm": "spine_center_z",
    "body_width_mm_median": "body_width_mm",
    "body_depth_mm_median": "body_depth_mm",
    "body_area_mm2_median": "body_area_mm2",
    "kidney_left_vs_spine_x": "kidney_left_center_x_rel",
    "kidney_left_vs_spine_y": "kidney_left_center_y_rel",
    "kidney_left_vs_spine_z": "kidney_left_center_z_rel",
    "kidney_right_vs_spine_x": "kidney_right_center_x_rel",
    "kidney_right_vs_spine_y": "kidney_right_center_y_rel",
    "kidney_right_vs_spine_z": "kidney_right_center_z_rel",
}

_KIDNEY_SIDES = ("left", "right")
_AXES = ("x", "y", "z")


@dataclass
class SchemaValidationResult:
    is_valid: bool
    missing_base: List[str] = field(default_factory=list)
    present_base: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def get_schema_yaml_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "phase1_feature_schema.yaml"


def load_schema_from_yaml(path: Optional[Path] = None) -> dict:
    """Load schema YAML for tooling; Python constants remain the runtime source."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load phase1_feature_schema.yaml")
    schema_path = path or get_schema_yaml_path()
    with open(schema_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def all_model_feature_names() -> List[str]:
    """Base + engineered + cross (pre-training column union)."""
    return list(BASE_FEATURES) + list(ENGINEERED_FEATURES) + list(CROSS_FEATURES)


def encode_patient_position(value: object, default: int = 1) -> int:
    """Map DICOM PatientPosition / scan_position string to numeric code."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    text = str(value).strip().upper()
    if not text:
        return default
    if text in PATIENT_POSITION_ENCODING:
        return PATIENT_POSITION_ENCODING[text]
    # DICOM codes like HFS, FFS, ...
    for code, encoded in PATIENT_POSITION_ENCODING.items():
        if code.isupper() and text == code:
            return encoded
    return default


def _first_series(df: pd.DataFrame, names: Sequence[str]) -> Optional[pd.Series]:
    for name in names:
        if name in df.columns:
            return df[name]
    return None


def _copy_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for alias, canonical in DIRECT_ALIASES.items():
        if canonical in out.columns:
            continue
        if alias in out.columns:
            out[canonical] = out[alias]
    return out


def _fill_center_rel_from_absolute(df: pd.DataFrame) -> pd.DataFrame:
    """Derive *_center_*_rel from absolute center coords and spine center."""
    out = df.copy()
    for side in _KIDNEY_SIDES:
        for axis in _AXES:
            rel_col = f"kidney_{side}_center_{axis}_rel"
            if rel_col in out.columns and out[rel_col].notna().any():
                continue
            abs_candidates = [
                f"kidney_{side}_center_{axis}",
                f"kidney_{side}_middle_{axis}",
            ]
            spine_col = f"spine_center_{axis}"
            abs_series = _first_series(out, abs_candidates)
            if abs_series is None or spine_col not in out.columns:
                continue
            out[rel_col] = abs_series.astype(float) - out[spine_col].astype(float)
    return out


def _fill_distances(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for side in _KIDNEY_SIDES:
        spine_dist_col = f"kidney_{side}_to_spine_distance"
        if spine_dist_col not in out.columns or out[spine_dist_col].isna().all():
            rel_cols = [f"kidney_{side}_center_{a}_rel" for a in _AXES]
            if all(c in out.columns for c in rel_cols):
                rel = out[rel_cols].astype(float)
                out[spine_dist_col] = np.sqrt((rel ** 2).sum(axis=1))

        body_dist_col = f"kidney_{side}_to_body_center_distance"
        if body_dist_col not in out.columns or out[body_dist_col].isna().all():
            center_cols = [f"kidney_{side}_center_{a}_rel" for a in _AXES]
            com_cols = ["body_com_x", "body_com_y", "body_com_z"]
            spine_cols = ["spine_center_x", "spine_center_y", "spine_center_z"]
            if all(c in out.columns for c in center_cols + com_cols + spine_cols):
                dx = (
                    out[center_cols[0]].astype(float)
                    - (out["body_com_x"].astype(float) - out["spine_center_x"].astype(float))
                )
                dy = (
                    out[center_cols[1]].astype(float)
                    - (out["body_com_y"].astype(float) - out["spine_center_y"].astype(float))
                )
                dz = (
                    out[center_cols[2]].astype(float)
                    - (out["body_com_z"].astype(float) - out["spine_center_z"].astype(float))
                )
                out[body_dist_col] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return out


def _fill_length_from_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for side in _KIDNEY_SIDES:
        len_col = f"kidney_{side}_length_mm"
        if len_col in out.columns and out[len_col].notna().any():
            continue
        depth_col = f"kidney_{side}_depth_mm"
        width_col = f"kidney_{side}_width_mm"
        if depth_col in out.columns and width_col in out.columns:
            out[len_col] = np.maximum(out[depth_col].astype(float), out[width_col].astype(float))
    return out


def _fill_patient_position_encoded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "patient_position_encoded" in out.columns and out["patient_position_encoded"].notna().any():
        return out
    pos_series = _first_series(out, ("scan_position", "patient_position"))
    if pos_series is not None:
        out["patient_position_encoded"] = pos_series.map(lambda v: encode_patient_position(v))
    return out


def normalize_dataframe(
    df: pd.DataFrame,
    *,
    inplace: bool = False,
    ensure_base_columns: bool = True,
) -> pd.DataFrame:
    """Rename aliases and derive missing canonical base features.

    Safe to call on train, validation, inference, and extractor outputs.
    Does NOT run engineered/cross feature engineering — that stays in
    ``AdaptiveEnsembleTrainer``.
    """
    out = df if inplace else df.copy()
    out = _copy_alias_columns(out)
    out = _fill_center_rel_from_absolute(out)
    out = _fill_length_from_segmentation(out)
    out = _fill_distances(out)
    out = _fill_patient_position_encoded(out)

    if ensure_base_columns:
        for col in BASE_FEATURES:
            if col not in out.columns:
                out[col] = np.nan
    return out


def normalize_record(record: Mapping[str, object]) -> Dict[str, object]:
    """Normalize a single patient feature dict (API / JSON path)."""
    df = normalize_dataframe(pd.DataFrame([dict(record)]))
    return df.iloc[0].to_dict()


def validate_base_features(
    df: pd.DataFrame,
    *,
    min_present_ratio: float = 0.8,
) -> SchemaValidationResult:
    """Check that enough canonical base columns are populated."""
    present = [c for c in BASE_FEATURES if c in df.columns and df[c].notna().any()]
    missing = [c for c in BASE_FEATURES if c not in present]
    ratio = len(present) / len(BASE_FEATURES) if BASE_FEATURES else 0.0
    warnings: List[str] = []
    if ratio < min_present_ratio:
        warnings.append(
            f"Only {len(present)}/{len(BASE_FEATURES)} base features present "
            f"({ratio:.0%}); minimum {min_present_ratio:.0%} required."
        )
    return SchemaValidationResult(
        is_valid=ratio >= min_present_ratio,
        missing_base=missing,
        present_base=present,
        warnings=warnings,
    )


def align_to_feature_names(
    df: pd.DataFrame,
    feature_names: Iterable[str],
) -> pd.DataFrame:
    """Return DataFrame with exactly ``feature_names`` columns (NaN fill)."""
    names = list(feature_names)
    normalized = normalize_dataframe(df)
    for col in names:
        if col not in normalized.columns:
            normalized[col] = np.nan
    return normalized[names]
