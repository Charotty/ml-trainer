"""Labelled training rows whose geometry comes from a CT extract, not the sheet.

Labels (the six ΔXYZ targets) stay on the Excel row. Base geometry columns are
replaced by the same-named columns from the CT extract, joined on surname.
The supine extract (``na_spine``) is the one that matches the Workbench input.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.data.holdout import normalize_surname_key
from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES

GEOMETRY_COLUMNS = list(BASE_FEATURES) + [
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


def _surname(series: pd.Series) -> pd.Series:
    return series.map(normalize_surname_key)


def build_ct_labeled_frame(
    labels: pd.DataFrame,
    ct: pd.DataFrame,
    *,
    label_name_column: str = "full_name",
    ct_name_column: str = "case_id",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (joined frame, counts). Duplicate surnames on either side are dropped."""
    if label_name_column not in labels.columns:
        raise KeyError(label_name_column)
    if ct_name_column not in ct.columns:
        raise KeyError(ct_name_column)

    left = labels.copy()
    left["surname"] = _surname(left[label_name_column])
    left = left[left["surname"].astype(str).str.len() > 0]
    left_dups = int(left["surname"].duplicated(keep=False).sum())
    left = left.drop_duplicates("surname", keep=False)

    right = ct.copy()
    right["surname"] = _surname(right[ct_name_column])
    right = right[right["surname"].astype(str).str.len() > 0]
    right_dups = int(right["surname"].duplicated(keep=False).sum())
    right = right.drop_duplicates("surname", keep=False)

    ct_cols = [col for col in GEOMETRY_COLUMNS if col in right.columns]
    keep = ["surname", *ct_cols]
    merged = left.merge(right[keep], on="surname", how="inner", suffixes=("", "_ct"))
    for col in ct_cols:
        ct_col = f"{col}_ct"
        if ct_col in merged.columns:
            merged[col] = merged[ct_col]
            merged.drop(columns=[ct_col], inplace=True)
    counts = {
        "label_rows": int(len(labels)),
        "ct_rows": int(len(ct)),
        "label_duplicate_surnames": left_dups,
        "ct_duplicate_surnames": right_dups,
        "paired_rows": int(len(merged)),
        "geometry_columns_copied": len(ct_cols),
    }
    return merged.reset_index(drop=True), counts


def missing_targets(frame: pd.DataFrame, targets: Iterable[str] = TARGET_NAMES) -> list[str]:
    return [target for target in targets if target not in frame.columns]
