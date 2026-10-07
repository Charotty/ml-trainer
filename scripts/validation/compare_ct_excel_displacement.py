#!/usr/bin/env python3
"""Compare CT extracts with «Смещение» in the vertebral frame.

Coordinates are the unsigned X and signed Y/Z written by
``measure_vertebral_frame`` (``kidney_{side}_{third}_{axis}_vert``).
The headline metrics are the displacement, lateral minus supine, against the
same difference in Excel. Absolute position is reported too, without adding
any constant correction.

Patients who share a surname are paired only when the rest of the name or a
date also matches. Otherwise they are listed and left out.

Usage
  py -3 scripts/inference/extract_from_dicom.py --dicom-root <на спине> --anatomy-profile full --canonical --output results/na_spine_full.csv
  py -3 scripts/inference/extract_from_dicom.py --dicom-root <на боку> --anatomy-profile full --canonical --output results/na_boku_full.csv
  py -3 scripts/validation/compare_ct_excel_displacement.py --spine results/na_spine_full.csv --boku results/na_boku_full.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.validate_extractor_anatomy import (  # noqa: E402
    SHARED_COLUMNS,
    SIDE_FIELDS,
    read_manual_table,
    surname_key,
)
from src.data.xlsx_displacement_parser import parse_xlsx_raw_table  # noqa: E402

POINTS = ("upper", "middle", "lower")
AXES = ("x", "y", "z")
SIDES = ("left", "right")

DEFAULT_XLSX = REPO_ROOT / "Смещение - конечное -13 .xlsx"
DEFAULT_OUT = REPO_ROOT / "results" / "ct_vs_excel"


def _extra_tokens(text: object) -> set:
    """Tokens other than the surname: given names, initials, dates."""
    parts = re.findall(r"[0-9a-zа-яё]+", str(text or "").lower().replace("ё", "е"))
    if not parts:
        return set()
    return set(parts[1:])


def _ct_label(row: pd.Series) -> str:
    for key in ("patient_name", "full_name", "case_id", "fio"):
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def _pair_groups(
    excel_rows: Sequence[dict],
    ct_rows: Sequence[dict],
) -> Tuple[List[Tuple[dict, dict]], List[dict]]:
    """Pair unique surnames. Ambiguous groups stay unmatched."""
    if len(excel_rows) == 1 and len(ct_rows) == 1:
        return [(excel_rows[0], ct_rows[0])], []

    used: set = set()
    pairs: List[Tuple[dict, dict]] = []
    skipped: List[dict] = []
    for excel in excel_rows:
        extras = _extra_tokens(excel.get("label", ""))
        scored = []
        for index, ct in enumerate(ct_rows):
            overlap = extras & _extra_tokens(ct.get("label", ""))
            scored.append((len(overlap), index))
        scored.sort(reverse=True)
        if not scored or scored[0][0] == 0 or (len(scored) > 1 and scored[0][0] == scored[1][0]):
            skipped.append(excel)
            continue
        index = scored[0][1]
        if index in used:
            skipped.append(excel)
            continue
        used.add(index)
        pairs.append((excel, ct_rows[index]))
    for index, ct in enumerate(ct_rows):
        if index not in used:
            skipped.append(ct)
    return pairs, skipped


def _groups(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        key = surname_key(row.get("label", ""))
        if not key:
            continue
        grouped.setdefault(key, []).append(row)
    return grouped


def match_tables(
    excel_rows: Sequence[dict],
    ct_rows: Sequence[dict],
) -> Tuple[List[Tuple[dict, dict]], List[dict]]:
    """Return matched pairs and the rows left out because the surname collides."""
    excel_groups = _groups(excel_rows)
    ct_groups = _groups(ct_rows)
    pairs: List[Tuple[dict, dict]] = []
    skipped: List[dict] = []
    for key in sorted(set(excel_groups) | set(ct_groups)):
        excel = excel_groups.get(key, [])
        ct = ct_groups.get(key, [])
        if len(excel) == 0 or len(ct) == 0:
            continue
        if len(excel) == 1 and len(ct) == 1:
            pairs.append((excel[0], ct[0]))
            continue
        matched, left = _pair_groups(excel, ct)
        pairs.extend(matched)
        skipped.extend(left)
    return pairs, skipped


def _value(row: dict, key: str) -> float:
    try:
        number = float(row.get(key))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(number):
        return float("nan")
    return number


def _summary(metric: str, manual: Iterable[float], auto: Iterable[float]) -> dict:
    manual_arr = np.asarray(list(manual), dtype=float)
    auto_arr = np.asarray(list(auto), dtype=float)
    ok = np.isfinite(manual_arr) & np.isfinite(auto_arr)
    diff = auto_arr[ok] - manual_arr[ok]
    if diff.size == 0:
        return {"metric": metric, "n": 0, "mae": np.nan, "bias_ct_minus_excel": np.nan, "median_abs": np.nan}
    return {
        "metric": metric,
        "n": int(diff.size),
        "mae": float(np.mean(np.abs(diff))),
        "bias_ct_minus_excel": float(diff.mean()),
        "median_abs": float(np.median(np.abs(diff))),
    }


def _ct_records(frame: pd.DataFrame, source: str) -> List[dict]:
    records = []
    for _, row in frame.iterrows():
        label = _ct_label(row)
        if not surname_key(label):
            continue
        item = {"label": label, "source": source, "surname": surname_key(label)}
        item.update({key: row[key] for key in frame.columns})
        records.append(item)
    return records


def _excel_coord_records(path: Path) -> List[dict]:
    table = parse_xlsx_raw_table(path)
    records = []
    for _, row in table.iterrows():
        label = str(row.get("fio") or "")
        if not surname_key(label):
            continue
        item = {"label": label, "source": "excel", "surname": surname_key(label)}
        item.update(row.to_dict())
        records.append(item)
    return records


def build_report(
    excel_coords: Sequence[dict],
    excel_anatomy: pd.DataFrame,
    spine_rows: Sequence[dict],
    boku_rows: Sequence[dict],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spine_pairs, spine_skipped = match_tables(excel_coords, spine_rows)
    boku_pairs, boku_skipped = match_tables(excel_coords, boku_rows)
    spine_by_surname = {pair[0]["surname"]: pair for pair in spine_pairs}
    boku_by_surname = {pair[0]["surname"]: pair for pair in boku_pairs}

    anatomy_by_name = {}
    if not excel_anatomy.empty and "name_key" in excel_anatomy.columns:
        counts = excel_anatomy["name_key"].value_counts()
        unique_names = set(counts[counts == 1].index.astype(str))
        for _, row in excel_anatomy.iterrows():
            key = str(row["name_key"])
            if key in unique_names:
                anatomy_by_name[key] = row

    patient_rows: List[dict] = []
    delta_store: Dict[str, List[Tuple[float, float]]] = {}
    position_store: Dict[str, List[Tuple[float, float]]] = {}
    anatomy_store: Dict[str, List[Tuple[float, float]]] = {}

    surnames = sorted(set(spine_by_surname) | set(boku_by_surname))
    for surname in surnames:
        spine_pair = spine_by_surname.get(surname)
        boku_pair = boku_by_surname.get(surname)
        excel = (boku_pair or spine_pair)[0]
        patient: dict = {
            "surname": surname,
            "excel_fio": excel.get("label"),
            "spine_case": None if spine_pair is None else spine_pair[1].get("label"),
            "boku_case": None if boku_pair is None else boku_pair[1].get("label"),
            "match_status": "excel_and_both_ct" if spine_pair and boku_pair else (
                "excel_and_boku" if boku_pair else "excel_and_spine"
            ),
        }
        if spine_pair and boku_pair:
            for side in SIDES:
                for point in POINTS:
                    for axis in AXES:
                        key = f"kidney_{side}_{point}_{axis}_vert"
                        excel_key = f"{side}_delta_{point}_{axis}"
                        manual = _value(excel, excel_key)
                        auto = _value(boku_pair[1], key) - _value(spine_pair[1], key)
                        metric = f"delta/{side}_{point}_{axis}"
                        delta_store.setdefault(metric, []).append((manual, auto))
                        patient[f"excel_{metric}"] = manual
                        patient[f"ct_{metric}"] = auto
                for pose, ct_row, prefix in (
                    ("supine", spine_pair[1], "position_supine"),
                    ("lateral", boku_pair[1], "position_lateral"),
                ):
                    for point in POINTS:
                        for axis in AXES:
                            excel_key = f"{side}_{pose}_{point}_{axis}"
                            ct_key = f"kidney_{side}_{point}_{axis}_vert"
                            manual = _value(excel, excel_key)
                            auto = _value(ct_row, ct_key)
                            metric = f"{prefix}/{side}_{point}_{axis}"
                            position_store.setdefault(metric, []).append((manual, auto))
        supine = spine_pair[1] if spine_pair else None
        anatomy = anatomy_by_name.get(surname)
        if supine is not None and anatomy is not None:
            for feature in list(SHARED_COLUMNS) + [
                f"kidney_{side}_{suffix}" for side in SIDES for suffix in SIDE_FIELDS
            ]:
                manual = _value(anatomy.to_dict(), feature)
                auto = _value(supine, feature)
                anatomy_store.setdefault(f"anatomy/{feature}", []).append((manual, auto))
                patient[f"excel_{feature}"] = manual
                patient[f"ct_{feature}"] = auto
        patient_rows.append(patient)

    duplicates = []
    for row in spine_skipped:
        duplicates.append({"surname": row.get("surname"), "name": row.get("label"), "source": row.get("source", "spine")})
    for row in boku_skipped:
        duplicates.append({"surname": row.get("surname"), "name": row.get("label"), "source": row.get("source", "boku")})

    summary = []
    for store in (delta_store, position_store, anatomy_store):
        for metric, pairs in store.items():
            summary.append(_summary(metric, [item[0] for item in pairs], [item[1] for item in pairs]))
    summary_frame = pd.DataFrame(summary, columns=["metric", "n", "mae", "bias_ct_minus_excel", "median_abs"])
    return summary_frame, pd.DataFrame(patient_rows), pd.DataFrame(duplicates)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare vertebral-frame CT features with the displacement workbook")
    parser.add_argument("--spine", type=Path, required=True, help="Supine extract CSV (--anatomy-profile full)")
    parser.add_argument("--boku", type=Path, required=True, help="Lateral extract CSV (--anatomy-profile full)")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.xlsx.exists():
        print(f"xlsx not found: {args.xlsx}")
        return 2
    for path in (args.spine, args.boku):
        if not path.exists():
            print(f"extract csv not found: {path}")
            return 2

    excel_coords = _excel_coord_records(args.xlsx)
    excel_anatomy = read_manual_table(args.xlsx)
    spine_rows = _ct_records(pd.read_csv(args.spine), "spine")
    boku_rows = _ct_records(pd.read_csv(args.boku), "boku")
    summary, patients, duplicates = build_report(excel_coords, excel_anatomy, spine_rows, boku_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ct_vs_excel_summary.csv"
    patients_path = args.output_dir / "ct_vs_excel_patients.csv"
    duplicates_path = args.output_dir / "ct_vs_excel_duplicate_surnames.csv"
    summary.to_csv(summary_path, index=False)
    patients.to_csv(patients_path, index=False)
    duplicates.to_csv(duplicates_path, index=False)
    paired = int((patients["match_status"] == "excel_and_both_ct").sum()) if not patients.empty else 0
    print(f"excel rows: {len(excel_coords)}  both-position pairs: {paired}  held out: {len(duplicates)}")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
