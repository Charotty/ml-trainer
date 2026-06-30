"""Enrich external CT feature tables to match clinical xlsx model inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.xlsx_displacement_parser import enrich_with_boku_volumes
from src.features.displacement_axis_features import CLINICAL_EXTRA_COLUMNS
from src.features.phase1_schema import normalize_dataframe
from src.features.projection_enrichment import (
    add_projection_delta_proxies,
    attach_projection_features,
    load_projection_lookup,
    normalize_name_key,
)

CLINICAL_DRIVER_COLS = [
    "lumbar_lordosis_deg",
    "s1_plate_tilt_deg",
    "abd_wall_thickness_mm",
    "bmi",
    "body_type",
]

SPAN_COLS = [
    "kidney_left_z_span_supine_mm",
    "kidney_right_z_span_supine_mm",
    "kidney_left_y_span_supine_mm",
    "kidney_right_y_span_supine_mm",
]

ANATOMICAL_COLS = [
    "kidney_lr_sep_x",
    "kidney_lr_sep_y",
    "kidney_lr_sep_z",
    "kidney_left_supine_middle_x",
    "kidney_left_supine_middle_y",
    "kidney_left_supine_middle_z",
    "kidney_right_supine_middle_x",
    "kidney_right_supine_middle_y",
    "kidney_right_supine_middle_z",
]

PROJ_SUP_SUFFIXES = [
    "kidney_left_center_x_rel",
    "kidney_left_center_y_rel",
    "kidney_left_center_z_rel",
    "kidney_right_center_x_rel",
    "kidney_right_center_y_rel",
    "kidney_right_center_z_rel",
    "body_width_mm",
    "body_depth_mm",
    "kidney_left_to_spine_distance",
    "kidney_right_to_spine_distance",
]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_spans_from_upper_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Infer supine kidney span (upper-lower) when pole coords exist."""
    out = df.copy()
    for side in ("left", "right"):
        for axis in ("y", "z"):
            col = f"kidney_{side}_{axis}_span_supine_mm"
            if col in out.columns and out[col].notna().any():
                continue
            upper = f"kidney_{side}_upper_{axis}"
            lower = f"kidney_{side}_lower_{axis}"
            if upper not in out.columns or lower not in out.columns:
                continue
            u = _num(out[upper])
            lo = _num(out[lower])
            span = (u - lo).abs()
            out[col] = span.where(u.notna() & lo.notna())
    return out


def recompute_body_com_offset(df: pd.DataFrame) -> pd.DataFrame:
    """Offset body_com from spine using abdominal geometry (clinical convention)."""
    out = df.copy()
    need = [f"spine_center_{a}" for a in "xyz"] + [f"body_com_{a}" for a in "xyz"]
    if not all(c in out.columns for c in need):
        return out

    spine_x = _num(out["spine_center_x"])
    spine_y = _num(out["spine_center_y"])
    spine_z = _num(out["spine_center_z"])
    body_x = _num(out["body_com_x"])
    body_y = _num(out["body_com_y"])
    body_z = _num(out["body_com_z"])
    degenerate = (
        ((spine_x - body_x).abs() < 1e-3)
        & ((spine_y - body_y).abs() < 1e-3)
        & ((spine_z - body_z).abs() < 1e-3)
    )

    depth = _num(out["body_depth_mm"]) if "body_depth_mm" in out.columns else pd.Series(np.nan, index=out.index)
    width = _num(out["body_width_mm"]) if "body_width_mm" in out.columns else pd.Series(np.nan, index=out.index)
    lordosis = _num(out["lumbar_lordosis_deg"]) if "lumbar_lordosis_deg" in out.columns else pd.Series(0.0, index=out.index)
    tilt = _num(out["s1_plate_tilt_deg"]) if "s1_plate_tilt_deg" in out.columns else pd.Series(0.0, index=out.index)

    lordosis_rad = np.deg2rad(lordosis.fillna(0.0))
    tilt_rad = np.deg2rad(tilt.fillna(0.0))
    com_x_off = width * 0.02
    com_y_off = depth * 0.06 * np.cos(lordosis_rad)
    com_z_off = depth * 0.04 * np.sin(tilt_rad)

    if not degenerate.any():
        return out

    idx = out.index[degenerate]
    out.loc[idx, "body_com_x"] = spine_x.loc[idx].values + com_x_off.loc[idx].values
    out.loc[idx, "body_com_y"] = spine_y.loc[idx].values + com_y_off.loc[idx].values
    out.loc[idx, "body_com_z"] = spine_z.loc[idx].values + com_z_off.loc[idx].values
    return out


def compute_anatomical_extras(df: pd.DataFrame) -> pd.DataFrame:
    """Derive lr_sep and supine middle absolute coords from spine + rel."""
    out = df.copy()
    spine_cols = [f"spine_center_{a}" for a in "xyz"]
    if not all(c in out.columns for c in spine_cols):
        return out

    for side in ("left", "right"):
        rel_cols = [f"kidney_{side}_center_{a}_rel" for a in "xyz"]
        if not all(c in out.columns for c in rel_cols):
            continue
        rel = out[rel_cols].astype(float)
        spine = out[spine_cols].astype(float).values
        abs_coords = rel.values + spine
        for j, axis in enumerate("xyz"):
            mid_col = f"kidney_{side}_supine_middle_{axis}"
            if mid_col not in out.columns or out[mid_col].isna().all():
                out[mid_col] = abs_coords[:, j]

    if all(c in out.columns for c in (
        "kidney_left_supine_middle_x", "kidney_right_supine_middle_x",
        "kidney_left_supine_middle_y", "kidney_right_supine_middle_y",
        "kidney_left_supine_middle_z", "kidney_right_supine_middle_z",
    )):
        for axis in "xyz":
            col = f"kidney_lr_sep_{axis}"
            if col not in out.columns or out[col].isna().all():
                l = _num(out[f"kidney_left_supine_middle_{axis}"])
                r = _num(out[f"kidney_right_supine_middle_{axis}"])
                out[col] = r - l
    return out


def apply_self_projection(df: pd.DataFrame) -> pd.DataFrame:
    """Fill proj_sup/proj_lat from row geometry when cohort is single-scan."""
    out = df.copy()
    cohort = out.get("dicom_cohort", pd.Series("", index=out.index)).astype(str).str.lower()
    scan = out.get("scan_position", pd.Series("", index=out.index)).astype(str).str.lower()

    is_lat = cohort.str.contains("boku") | scan.str.contains("lateral")
    is_sup = cohort.str.contains("spine") | scan.str.contains("supine")

    for prefix, mask in (("sup", is_sup), ("lat", is_lat)):
        if not mask.any():
            continue
        for col in PROJ_SUP_SUFFIXES:
            proj_col = f"proj_{prefix}_{col}"
            if col not in out.columns:
                continue
            src = _num(out[col])
            if proj_col not in out.columns:
                out[proj_col] = np.nan
            out.loc[mask, proj_col] = out.loc[mask, proj_col].fillna(src.loc[mask])
    return out


def fill_clinical_drivers_from_reference(
    df: pd.DataFrame,
    clinical_ref: pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing clinical driver columns from matched patient or cohort medians."""
    out = df.copy()
    ref = normalize_dataframe(clinical_ref)
    medians: dict[str, float] = {}
    for col in CLINICAL_DRIVER_COLS + SPAN_COLS:
        if col in ref.columns:
            s = _num(ref[col]).dropna()
            if len(s):
                medians[col] = float(s.median())

    name_lookup: dict[str, dict[str, float]] = {}
    if "full_name" in ref.columns:
        ref_keys = ref.copy()
        ref_keys["_name_key"] = ref_keys["full_name"].map(normalize_name_key)
        for col in CLINICAL_DRIVER_COLS + SPAN_COLS:
            if col not in ref_keys.columns:
                continue
            for key, val in zip(ref_keys["_name_key"], _num(ref_keys[col])):
                if not key or not np.isfinite(val):
                    continue
                name_lookup.setdefault(key, {})[col] = float(val)

    keys = out.get("full_name_key", out.get("full_name", pd.Series("", index=out.index)))
    keys = keys.map(normalize_name_key)

    for col in CLINICAL_DRIVER_COLS + SPAN_COLS:
        if col not in out.columns:
            out[col] = np.nan
        for idx, key in keys.items():
            if out.at[idx, col] is not None and np.isfinite(out.at[idx, col]):
                continue
            if key in name_lookup and col in name_lookup[key]:
                out.at[idx, col] = name_lookup[key][col]
            elif col in medians:
                out.at[idx, col] = medians[col]
    return out


def enrich_external_ct_frame(
    df: pd.DataFrame,
    *,
    clinical_reference: pd.DataFrame | None = None,
    projection_lookup: pd.DataFrame | None = None,
    boku_path: Path | str | None = None,
    source_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply clinical-parity enrichment for external CT / KiTS rows."""
    out = normalize_dataframe(df.copy())
    steps: list[str] = []

    out = compute_spans_from_upper_lower(out)
    steps.append("spans_from_upper_lower")

    out = recompute_body_com_offset(out)
    steps.append("body_com_offset")

    out = compute_anatomical_extras(out)
    steps.append("anatomical_extras")

    if clinical_reference is not None and len(clinical_reference):
        out = fill_clinical_drivers_from_reference(out, clinical_reference)
        steps.append("clinical_driver_fill")

    if boku_path:
        out = enrich_with_boku_volumes(out, boku_path)
        steps.append("boku_volumes")

    lookup = projection_lookup if projection_lookup is not None else load_projection_lookup()
    out = attach_projection_features(out, lookup=lookup)
    steps.append("projection_lookup_join")

    out = apply_self_projection(out)
    steps.append("self_projection")

    out = add_projection_delta_proxies(out)
    steps.append("projection_delta_proxies")

    meta = {
        "source_id": source_id,
        "steps": steps,
        "enriched_columns_added": sorted(
            set(out.columns) - set(df.columns)
        ),
    }
    return out, meta
