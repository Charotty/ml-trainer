#!/usr/bin/env python3
"""Merge na_spine / na_boku extract CSVs into canonical DICOM feature table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.phase1_schema import BASE_FEATURES, normalize_dataframe

DEFAULT_SPINE = ROOT / "data" / "na_spine_full.csv"
DEFAULT_BOKU = ROOT / "data" / "na_boku_full.bak.csv"
DEFAULT_OUT = ROOT / "data" / "dicom_medical_features.csv"


def _load_ok(path: Path, cohort: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df = normalize_dataframe(df)
    if "status" in df.columns:
        df = df[df["status"] == "extracted"].copy()
    need = ["spine_center_x", "kidney_left_center_x", "kidney_right_center_x"]
    for col in need:
        if col not in df.columns:
            return pd.DataFrame()
    df = df[
        df["spine_center_x"].notna()
        & df["kidney_left_center_x"].notna()
        & df["kidney_right_center_x"].notna()
    ].copy()
    df["dicom_cohort"] = cohort
    if "scan_position" not in df.columns:
        df["scan_position"] = "supine"
    return df


def build(spine_path: Path, boku_path: Path) -> pd.DataFrame:
    parts = []
    spine = _load_ok(spine_path, "na_spine")
    boku = _load_ok(boku_path, "na_boku")
    if len(spine):
        parts.append(spine)
    if len(boku):
        parts.append(boku)
    if not parts:
        raise ValueError("No usable rows in spine/boku inputs")

    out = pd.concat(parts, ignore_index=True)
    if "case_id" not in out.columns:
        out["case_id"] = [f"dicom_{i:04d}" for i in range(1, len(out) + 1)]
    out["case_id"] = out.apply(
        lambda r: f"{r.get('dicom_cohort', 'dicom')}_{r['case_id']}", axis=1
    )
    if "full_name" not in out.columns and "case_id" in out.columns:
        out["full_name"] = out["case_id"]

    base_present = [c for c in BASE_FEATURES if c in out.columns]
    complete = out[base_present].notna().all(axis=1).sum() if base_present else 0
    print(f"rows={len(out)} spine={len(spine)} boku={len(boku)} base_complete={complete}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spine", type=Path, default=DEFAULT_SPINE)
    parser.add_argument("--boku", type=Path, default=DEFAULT_BOKU)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = build(args.spine, args.boku)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
