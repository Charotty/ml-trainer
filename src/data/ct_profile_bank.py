"""CT profile bank: compare matched rows with the sheet, use every row as a cohort.

``na_spine`` is the supine extract, ``na_boku`` the lateral extract. A row is
matched when ``case_id`` and the sheet surname are the same key. Matched rows
are scored against the sheet. Every row, matched or not, supplies the median
and MAD of supine-safe measurements. Lateral kidney coordinates are not copied
onto training rows: they already contain the displacement the model predicts.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from src.data.holdout import normalize_surname_key

SIDES = ("left", "right")
AXES = ("x", "y", "z")
POINTS = ("upper", "middle", "lower")

# Present on a supine scan and on a Workbench case. Safe as predictors.
SUPINE_COLUMNS = (
    "kidney_left_volume_cm3",
    "kidney_right_volume_cm3",
    "kidney_left_middle_x_vert",
    "kidney_left_middle_y_vert",
    "kidney_left_middle_z_vert",
    "kidney_right_middle_x_vert",
    "kidney_right_middle_y_vert",
    "kidney_right_middle_z_vert",
    "kidney_left_medial_to_spine_vert_mm",
    "kidney_right_medial_to_spine_vert_mm",
    "kidney_left_psoas_area_vert_cm2",
    "kidney_right_psoas_area_vert_cm2",
    "body_volume_cm3",
    "fat_volume_cm3",
)

# Lateral position is after the move. Keep size, not coordinates.
LATERAL_COLUMNS = (
    "kidney_left_volume_cm3",
    "kidney_right_volume_cm3",
    "kidney_left_psoas_area_vert_cm2",
    "kidney_right_psoas_area_vert_cm2",
    "body_volume_cm3",
    "fat_volume_cm3",
    "kidney_left_medial_to_spine_vert_mm",
    "kidney_right_medial_to_spine_vert_mm",
)


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _robust(series: pd.Series) -> Dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"median": np.nan, "mad": np.nan, "q10": np.nan, "q90": np.nan, "n": 0}
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad < 1e-9:
        mad = float(values.std()) if float(values.std()) > 1e-9 else 1.0
    return {
        "median": median,
        "mad": mad,
        "q10": float(values.quantile(0.10)),
        "q90": float(values.quantile(0.90)),
        "n": int(len(values)),
    }


def load_profile(path: str | pd.DataFrame) -> pd.DataFrame:
    frame = path if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    out = frame.copy()
    out["surname"] = out["case_id"].map(normalize_surname_key) if "case_id" in out.columns else ""
    out = out[out["surname"].astype(str).str.len() > 0]
    out = out.drop_duplicates("surname", keep=False)
    return out.reset_index(drop=True)


def _sheet_surname(frame: pd.DataFrame) -> pd.Series:
    column = "fio" if "fio" in frame.columns else "full_name"
    return frame[column].map(normalize_surname_key)


def match_counts(sheet: pd.DataFrame, supine: pd.DataFrame, lateral: pd.DataFrame) -> Dict[str, int]:
    sheet_keys = set(_sheet_surname(sheet)) - {""}
    supine_keys = set(supine["surname"]) - {""}
    lateral_keys = set(lateral["surname"]) - {""}
    return {
        "sheet_rows": int(len(sheet)),
        "supine_rows": int(len(supine)),
        "lateral_rows": int(len(lateral)),
        "matched_supine": int(len(sheet_keys & supine_keys)),
        "matched_lateral": int(len(sheet_keys & lateral_keys)),
        "supine_only": int(len(supine_keys - sheet_keys)),
        "lateral_only": int(len(lateral_keys - sheet_keys)),
        "supine_and_lateral": int(len(supine_keys & lateral_keys)),
    }


def _error_summary(ct: pd.Series, excel: pd.Series, metric: str) -> Dict[str, float] | None:
    both = ct.notna() & excel.notna()
    if int(both.sum()) == 0:
        return None
    err = ct[both] - excel[both]
    return {
        "metric": metric,
        "n": int(both.sum()),
        "ct_mean": float(ct[both].mean()),
        "excel_mean": float(excel[both].mean()),
        "bias": float(err.mean()),
        "mae": float(err.abs().mean()),
    }


def agreement_rows(sheet: pd.DataFrame, ct: pd.DataFrame, pose: str) -> List[Dict[str, float]]:
    """CT vertebral middle, volume and abdomen size versus the sheet pose."""
    sheet_keys = sheet.copy()
    sheet_keys["surname"] = _sheet_surname(sheet_keys)
    sheet_keys = sheet_keys[sheet_keys["surname"].astype(str).str.len() > 0]
    sheet_keys = sheet_keys.drop_duplicates("surname", keep=False)
    merged = sheet_keys.merge(ct, on="surname", how="inner", suffixes=("", "_ct"))
    rows: List[Dict[str, float]] = []
    for side in SIDES:
        vol = _error_summary(
            _finite_series(merged, f"kidney_{side}_volume_cm3"),
            _finite_series(merged, f"{side}_kidney_volume_cm3")
            if f"{side}_kidney_volume_cm3" in merged.columns
            else _finite_series(merged, f"excel_{side}_volume_cm3"),
            f"{pose}_{side}_volume_cm3",
        )
        if vol:
            rows.append(vol)
        for point in POINTS:
            for axis in AXES:
                summary = _error_summary(
                    _finite_series(merged, f"kidney_{side}_{point}_{axis}_vert"),
                    _finite_series(merged, f"{side}_{pose}_{point}_{axis}"),
                    f"{pose}_{side}_{point}_{axis}_vert",
                )
                if summary:
                    rows.append(summary)
    for ct_col, excel_col, name in (
        ("body_width_mm", "abd_width_l3l4_mm", f"{pose}_body_width_mm"),
        ("body_depth_mm", "abd_depth_l3l4_mm", f"{pose}_body_depth_mm"),
    ):
        summary = _error_summary(_finite_series(merged, ct_col), _finite_series(merged, excel_col), name)
        if summary:
            rows.append(summary)
    return rows


def vert_middle_bias(agreement: Iterable[Mapping[str, float]], pose: str = "lateral") -> Dict[str, float]:
    """Mean CT-minus-sheet bias of the middle point, left and right averaged."""
    by_axis: Dict[str, List[float]] = {axis: [] for axis in AXES}
    for row in agreement:
        metric = str(row.get("metric") or "")
        for axis in AXES:
            if metric.endswith(f"_middle_{axis}_vert") and metric.startswith(pose):
                by_axis[axis].append(float(row["bias"]))
    return {
        axis: float(np.mean(values)) if values else float("nan")
        for axis, values in by_axis.items()
    }


def fit_cohort(frame: pd.DataFrame, columns: Iterable[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        robust = _robust(frame[column])
        if robust["n"] > 0:
            stats[column] = robust
    return stats


def _z_value(value: float, stats: Mapping[str, float]) -> float:
    if not np.isfinite(value):
        return float("nan")
    median = float(stats.get("median", np.nan))
    mad = float(stats.get("mad", np.nan))
    if not np.isfinite(median) or not np.isfinite(mad) or mad <= 0:
        return float("nan")
    return float((value - median) / mad)


def _attach_one(
    labels: pd.DataFrame,
    source: pd.DataFrame,
    columns: Iterable[str],
    prefix: str,
    stats: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    keep = ["surname", *[column for column in columns if column in source.columns]]
    piece = source[keep].rename(columns={column: f"{prefix}{column}" for column in keep if column != "surname"})
    out = labels.drop(columns=[column for column in piece.columns if column != "surname" and column in labels.columns], errors="ignore")
    out = out.merge(piece, on="surname", how="left")
    for column in columns:
        raw_name = f"{prefix}{column}"
        if raw_name not in out.columns:
            out[raw_name] = np.nan
        column_stats = stats.get(column) or {}
        out[f"{prefix}z_{column}"] = [
            _z_value(float(value) if pd.notna(value) else np.nan, column_stats)
            for value in out[raw_name]
        ]
    return out


def attach_predictors(
    labels: pd.DataFrame,
    supine: pd.DataFrame,
    lateral: pd.DataFrame,
    supine_stats: Mapping[str, Mapping[str, float]],
    lateral_stats: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Add matched CT predictors and a z-score against the full cohort."""
    out = labels.copy()
    out["surname"] = _sheet_surname(out)
    out = _attach_one(out, supine, SUPINE_COLUMNS, "ct_sup_", supine_stats)
    out = _attach_one(out, lateral, LATERAL_COLUMNS, "ct_lat_", lateral_stats)
    return out.drop(columns=["surname"])


def indicators_from_row(
    row: Mapping[str, object],
    supine_stats: Mapping[str, Mapping[str, float]],
) -> Dict[str, float | None]:
    """Where a live supine CT sits in the full spine cohort. Missing stays null."""
    out: Dict[str, float | None] = {}
    for column, stats in supine_stats.items():
        try:
            value = float(row.get(column))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            value = float("nan")
        if not np.isfinite(value):
            out[f"ct_sup_z_{column}"] = None
            continue
        score = _z_value(value, stats)
        out[f"ct_sup_z_{column}"] = None if not np.isfinite(score) else round(score, 3)
    return out


def cohort_indicators_for_case(
    base: Mapping[str, object],
    raw: Mapping[str, object] | None = None,
    bank: Mapping[str, object] | None = None,
) -> Dict[str, float | None]:
    """Supine-cohort z-scores for one case. Empty when the bank file is absent."""
    loaded = bank if bank is not None else load_bank()
    if not loaded:
        return {}
    stats = loaded.get("supine_stats") or {}
    row = dict(base)
    for column in stats:
        if row.get(column) is None and raw is not None and raw.get(column) is not None:
            row[column] = raw[column]
    return indicators_from_row(row, stats)


def load_bank(path: str | None = None) -> Dict[str, object]:
    import json
    from pathlib import Path

    bank_path = Path(path) if path else Path(__file__).resolve().parents[2] / "models" / "ct_profile_bank.json"
    if not bank_path.exists():
        return {}
    return json.loads(bank_path.read_text(encoding="utf-8"))


def bank_from_frames(supine: pd.DataFrame, lateral: pd.DataFrame) -> Dict[str, object]:
    return {
        "supine_rows": int(len(supine)),
        "lateral_rows": int(len(lateral)),
        "supine_stats": fit_cohort(supine, SUPINE_COLUMNS),
        "lateral_stats": fit_cohort(lateral, LATERAL_COLUMNS),
        "supine_columns": list(SUPINE_COLUMNS),
        "lateral_columns": list(LATERAL_COLUMNS),
    }
