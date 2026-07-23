"""Convert Excel-derived displacement table to Phase 1 canonical schema."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES, normalize_dataframe

DEFAULT_EXCEL_PATH = "data/train_displacement_dataset.csv"


def _normalize_name_key(name: object) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    text = str(name).strip().lower().replace("ё", "е")
    return re.sub(r"\s+", "", text)


def _parse_numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = (
            series.astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace("\u00a0", "", regex=False)
            .str.strip()
        )
        cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _sex_to_vybor_code(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in {"м", "m", "male", "1", "1.0"}:
        return 1.0
    if text in {"ж", "f", "female", "2", "2.0"}:
        return 2.0
    return np.nan


def _body_type_to_code(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    mapping = {
        "норма": 0.0,
        "нормостеническое": 0.0,
        "астеническое": 1.0,
        "астеническое ": 1.0,
        "гиперстеническое": 2.0,
        "гипер": 2.0,
    }
    return mapping.get(text, np.nan)


def convert_excel_displacement_df(
    df: pd.DataFrame,
    *,
    delta_point: str = "middle",
    exclude_names: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Map ``train_displacement_dataset.csv`` rows to canonical labeled schema."""
    exclude_names = exclude_names or set()
    rows = []

    for _, raw in df.iterrows():
        name_key = _normalize_name_key(raw.get("fio"))
        if not name_key or name_key in exclude_names:
            continue

        row_no = raw.get("row_no")
        case_id = f"excel_{int(row_no)}" if pd.notna(row_no) else f"excel_{len(rows) + 1}"

        left_sup_x = _parse_numeric_series(pd.Series([raw.get(f"left_supine_{delta_point}_x")])).iloc[0]
        left_sup_y = _parse_numeric_series(pd.Series([raw.get(f"left_supine_{delta_point}_y")])).iloc[0]
        left_sup_z = _parse_numeric_series(pd.Series([raw.get(f"left_supine_{delta_point}_z")])).iloc[0]
        right_sup_x = _parse_numeric_series(pd.Series([raw.get(f"right_supine_{delta_point}_x")])).iloc[0]
        right_sup_y = _parse_numeric_series(pd.Series([raw.get(f"right_supine_{delta_point}_y")])).iloc[0]
        right_sup_z = _parse_numeric_series(pd.Series([raw.get(f"right_supine_{delta_point}_z")])).iloc[0]

        body_w = _parse_numeric_series(pd.Series([raw.get("abd_width_l3l4_mm")])).iloc[0]
        body_d = _parse_numeric_series(pd.Series([raw.get("abd_depth_l3l4_mm")])).iloc[0]

        # Spine anchor: inter-kidney midpoint in supine (clinical table convention).
        spine_x = np.nanmean([left_sup_x, right_sup_x])
        spine_y = np.nanmean([left_sup_y, right_sup_y])
        spine_z = np.nanmean([left_sup_z, right_sup_z])

        left_rel = np.array([left_sup_x - spine_x, left_sup_y - spine_y, left_sup_z - spine_z], dtype=float)
        right_rel = np.array([right_sup_x - spine_x, right_sup_y - spine_y, right_sup_z - spine_z], dtype=float)

        # Body COM: offset from spine using abdominal geometry (not identical to spine).
        lordosis = _parse_numeric_series(pd.Series([raw.get("lumbar_lordosis_deg")])).iloc[0]
        tilt = _parse_numeric_series(pd.Series([raw.get("s1_plate_tilt_deg")])).iloc[0]
        depth = float(body_d) if pd.notna(body_d) else np.nan
        width = float(body_w) if pd.notna(body_w) else np.nan
        lordosis_rad = np.deg2rad(lordosis) if pd.notna(lordosis) else 0.0
        tilt_rad = np.deg2rad(tilt) if pd.notna(tilt) else 0.0
        com_y_off = depth * 0.06 * np.cos(lordosis_rad) if np.isfinite(depth) else 0.0
        com_z_off = depth * 0.04 * np.sin(tilt_rad) if np.isfinite(depth) else 0.0
        com_x_off = width * 0.02 if np.isfinite(width) else 0.0
        body_com = np.array([
            spine_x + com_x_off,
            spine_y + com_y_off,
            spine_z + com_z_off,
        ], dtype=float)

        left_to_spine_vec = left_rel
        right_to_spine_vec = right_rel
        left_to_body_vec = np.array([
            left_sup_x - body_com[0],
            left_sup_y - body_com[1],
            left_sup_z - body_com[2],
        ], dtype=float)
        right_to_body_vec = np.array([
            right_sup_x - body_com[0],
            right_sup_y - body_com[1],
            right_sup_z - body_com[2],
        ], dtype=float)

        left_to_spine = float(np.linalg.norm(left_to_spine_vec)) if np.isfinite(left_to_spine_vec).all() else np.nan
        right_to_spine = float(np.linalg.norm(right_to_spine_vec)) if np.isfinite(right_to_spine_vec).all() else np.nan
        left_to_body = float(np.linalg.norm(left_to_body_vec)) if np.isfinite(left_to_body_vec).all() else np.nan
        right_to_body = float(np.linalg.norm(right_to_body_vec)) if np.isfinite(right_to_body_vec).all() else np.nan

        record = {
            "case_id": case_id,
            "full_name": raw.get("fio"),
            "sex": _sex_to_vybor_code(raw.get("sex")),
            "age": _parse_numeric_series(pd.Series([raw.get("age")])).iloc[0],
            "bmi": _parse_numeric_series(pd.Series([raw.get("bmi")])).iloc[0],
            "body_type": _body_type_to_code(raw.get("body_type")),
            "has_previous_surgery": _parse_numeric_series(
                pd.Series([raw.get("has_previous_surgery")])
            ).iloc[0],
            "scan_position": "supine",
            "source": "Excel",
            "source_name": "Excel",
            "kidney_left_center_x_rel": left_rel[0],
            "kidney_left_center_y_rel": left_rel[1],
            "kidney_left_center_z_rel": left_rel[2],
            "kidney_right_center_x_rel": right_rel[0],
            "kidney_right_center_y_rel": right_rel[1],
            "kidney_right_center_z_rel": right_rel[2],
            "body_width_mm": body_w,
            "body_depth_mm": body_d,
            "body_area_mm2": body_w * body_d if pd.notna(body_w) and pd.notna(body_d) else np.nan,
            "spine_center_x": spine_x,
            "spine_center_y": spine_y,
            "spine_center_z": spine_z,
            "body_com_x": body_com[0],
            "body_com_y": body_com[1],
            "body_com_z": body_com[2],
            "kidney_left_to_spine_distance": left_to_spine,
            "kidney_right_to_spine_distance": right_to_spine,
            "kidney_left_to_body_center_distance": left_to_body,
            "kidney_right_to_body_center_distance": right_to_body,
            "kidney_lr_sep_x": right_sup_x - left_sup_x if pd.notna(right_sup_x) and pd.notna(left_sup_x) else np.nan,
            "kidney_lr_sep_y": right_sup_y - left_sup_y if pd.notna(right_sup_y) and pd.notna(left_sup_y) else np.nan,
            "kidney_lr_sep_z": right_sup_z - left_sup_z if pd.notna(right_sup_z) and pd.notna(left_sup_z) else np.nan,
            "kidney_left_supine_middle_x": left_sup_x,
            "kidney_left_supine_middle_y": left_sup_y,
            "kidney_left_supine_middle_z": left_sup_z,
            "kidney_right_supine_middle_x": right_sup_x,
            "kidney_right_supine_middle_y": right_sup_y,
            "kidney_right_supine_middle_z": right_sup_z,
            "lumbar_lordosis_deg": lordosis,
            "s1_plate_tilt_deg": tilt,
            "abd_wall_thickness_mm": _parse_numeric_series(pd.Series([raw.get("abd_wall_thickness_mm")])).iloc[0],
        }

        for side, prefix in (("left", "kidney_left"), ("right", "kidney_right")):
            for axis, suffix in zip("xyz", ["x", "y", "z"]):
                col = f"{side}_delta_{delta_point}_{suffix}"
                target = f"{prefix}_delta_{suffix}"
                record[target] = _parse_numeric_series(pd.Series([raw.get(col)])).iloc[0]

        rows.append(record)

    if not rows:
        return pd.DataFrame(columns=list(BASE_FEATURES) + list(TARGET_NAMES))

    out = normalize_dataframe(pd.DataFrame(rows))
    return out


def load_excel_displacement_table(
    path: str = DEFAULT_EXCEL_PATH,
    vybor_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load Excel CSV and skip patients already present in Vybor (by name)."""
    excel_raw = pd.read_csv(path)
    exclude: set[str] = set()
    if vybor_df is not None and "full_name" in vybor_df.columns:
        exclude = {_normalize_name_key(n) for n in vybor_df["full_name"].dropna()}
        exclude.discard("")
    converted = convert_excel_displacement_df(excel_raw, exclude_names=exclude)
    complete = converted.dropna(subset=list(TARGET_NAMES), how="any")
    skipped = len(converted) - len(complete)
    if skipped:
        print(
            f"[excel] Skipped {skipped} rows with incomplete targets "
            f"(kept {len(complete)} unique vs Vybor)"
        )
    return complete
