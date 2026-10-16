#!/usr/bin/env python3
"""Score CT profiles against the displacement sheet and save the cohort bank.

Matched surnames are compared. Every profile row, matched or not, is stored
as a cohort distribution in ``models/ct_profile_bank.json`` (no names).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ct_profile_bank import (  # noqa: E402
    agreement_rows,
    bank_from_frames,
    load_profile,
    match_counts,
    vert_middle_bias,
)
from src.data.holdout import normalize_surname_key  # noqa: E402
from src.data.xlsx_displacement_parser import (  # noqa: E402
    _COL,
    _cell,
    _is_header_like_row,
    _parse_numeric,
    _read_xlsx_rows,
    parse_xlsx_raw_table,
)

DEFAULT_XLSX = ROOT / "Смещение - конечное -14 .xlsx"
DEFAULT_SPINE = ROOT / "na_spine_full_profile.csv"
DEFAULT_BOKU = ROOT / "na_boku_full_profile.csv"
BANK_PATH = ROOT / "models" / "ct_profile_bank.json"
SUMMARY_PATH = ROOT / "results" / "ct_profile_agreement_summary.csv"


def _volumes(path: Path) -> pd.DataFrame:
    records = []
    for sheet_idx, row in enumerate(_read_xlsx_rows(path)):
        if sheet_idx < 5 or _is_header_like_row(row):
            continue
        surname = normalize_surname_key(_cell(row, _COL["fio"]))
        if not surname:
            continue
        records.append(
            {
                "surname": surname,
                "left_kidney_volume_cm3": _parse_numeric(_cell(row, _COL["left_kidney_volume_cm3"])),
                "right_kidney_volume_cm3": _parse_numeric(_cell(row, _COL["right_kidney_volume_cm3"])),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.drop_duplicates("surname", keep=False)


def main() -> int:
    sheet = parse_xlsx_raw_table(DEFAULT_XLSX, exclude_surnames=[])
    sheet = sheet.copy()
    sheet["surname"] = sheet["fio"].map(normalize_surname_key)
    volumes = _volumes(DEFAULT_XLSX)
    if not volumes.empty:
        sheet = sheet.merge(volumes, on="surname", how="left")
    supine = load_profile(DEFAULT_SPINE)
    lateral = load_profile(DEFAULT_BOKU)
    counts = match_counts(sheet, supine, lateral)
    lateral_agreement = agreement_rows(sheet, lateral, "lateral")
    supine_agreement = agreement_rows(sheet, supine, "supine")
    bias = vert_middle_bias(lateral_agreement, "lateral")
    summary = pd.DataFrame(lateral_agreement + supine_agreement)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    bank = bank_from_frames(supine, lateral)
    bank["match_counts"] = counts
    bank["lateral_middle_bias_ct_minus_sheet"] = bias
    BANK_PATH.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"counts": counts, "lateral_middle_bias": bias}, ensure_ascii=False, indent=2))
    if not summary.empty:
        show = summary[summary["metric"].str.contains("middle_|volume_cm3|body_")]
        print(show.to_string(index=False))
    print(f"[ok] {SUMMARY_PATH}")
    print(f"[ok] {BANK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
