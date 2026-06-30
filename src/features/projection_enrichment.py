"""Attach unpaired lateral (na_boku) and supine (na_spine) projection features by patient name."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.phase1_schema import normalize_dataframe

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOKU_PATH = REPO_ROOT / "data" / "harmonized" / "na_boku_full_aligned.csv"
DEFAULT_SPINE_PATH = REPO_ROOT / "data" / "harmonized" / "na_spine_full_aligned.csv"
FALLBACK_BOKU_PATH = REPO_ROOT / "data" / "na_boku_full.bak.csv"
FALLBACK_SPINE_PATH = REPO_ROOT / "data" / "na_spine_full.csv"

PROJECTION_SOURCE_COLUMNS: dict[str, list[str]] = {
    "lat": [
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
    ],
    "sup": [
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
    ],
}


def normalize_name_key(value: object) -> str:
    """Lowercase ASCII key for fuzzy patient matching."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _resolve_path(primary: Path, fallback: Path) -> Path | None:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return None


def _load_projection_table(path: Path, prefix: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = normalize_dataframe(raw)
    if "full_name_key" in df.columns:
        keys = df["full_name_key"].map(normalize_name_key)
    elif "full_name" in df.columns:
        keys = df["full_name"].map(normalize_name_key)
    elif "patient_name" in df.columns:
        keys = df["patient_name"].map(normalize_name_key)
    else:
        keys = df.iloc[:, 0].map(normalize_name_key)

    out = pd.DataFrame({"_name_key": keys})
    for col in PROJECTION_SOURCE_COLUMNS.get(prefix.replace("proj_", ""), []):
        if col not in df.columns:
            continue
        out[f"proj_{prefix}_{col}"] = pd.to_numeric(df[col], errors="coerce")

    out = out[out["_name_key"].astype(str).str.len() > 0]
    return out.drop_duplicates(subset=["_name_key"], keep="first")


def load_projection_lookup(
    *,
    boku_path: Path | None = None,
    spine_path: Path | None = None,
) -> pd.DataFrame:
    """Build merged lookup: one row per name_key with lat_* and sup_* columns."""
    boku_p = _resolve_path(boku_path or DEFAULT_BOKU_PATH, FALLBACK_BOKU_PATH)
    spine_p = _resolve_path(spine_path or DEFAULT_SPINE_PATH, FALLBACK_SPINE_PATH)

    parts: list[pd.DataFrame] = []
    if boku_p is not None:
        parts.append(_load_projection_table(boku_p, "lat"))
    if spine_p is not None:
        parts.append(_load_projection_table(spine_p, "sup"))

    if not parts:
        return pd.DataFrame(columns=["_name_key"])

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="_name_key", how="outer")
    return merged


def projection_feature_names(lookup: pd.DataFrame | None = None) -> list[str]:
    if lookup is None:
        lookup = load_projection_lookup()
    return [c for c in lookup.columns if c.startswith("proj_")]


def attach_projection_features(
    df: pd.DataFrame,
    lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join projection features; unmatched rows keep NaN (for imputer)."""
    out = normalize_dataframe(df.copy())
    if lookup is None:
        lookup = load_projection_lookup()
    if lookup.empty or "_name_key" not in lookup.columns:
        return out

    if "full_name_key" in out.columns:
        keys = out["full_name_key"].map(normalize_name_key)
    elif "full_name" in out.columns:
        keys = out["full_name"].map(normalize_name_key)
    else:
        keys = pd.Series([""] * len(out), index=out.index)

    keyed = out.copy()
    keyed["_name_key"] = keys
    feat_cols = [c for c in lookup.columns if c != "_name_key"]
    merged = keyed.merge(lookup[["_name_key", *feat_cols]], on="_name_key", how="left")
    merged = merged.drop(columns=["_name_key"], errors="ignore")
    return merged


def add_projection_delta_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Optional proxies: lat - sup rel coords (NOT ground-truth δ, aux only)."""
    out = df.copy()
    for side in ("left", "right"):
        for axis in ("x", "y", "z"):
            lat = f"proj_lat_kidney_{side}_center_{axis}_rel"
            sup = f"proj_sup_kidney_{side}_center_{axis}_rel"
            if lat in out.columns and sup in out.columns:
                out[f"proj_diff_{side}_{axis}"] = (
                    pd.to_numeric(out[lat], errors="coerce")
                    - pd.to_numeric(out[sup], errors="coerce")
                )
    return out
