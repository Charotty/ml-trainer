#!/usr/bin/env python3
"""Build data/vybor_from_xlsx.csv from the canonical displacement workbook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.holdout import (  # noqa: E402
    DEFAULT_HOLDOUT_PATH,
    assert_no_holdout_leak,
    holdout_surname_keys,
    load_holdout_config,
)
from src.data.xlsx_displacement_parser import (  # noqa: E402
    DEFAULT_OUTPUT_CSV,
    DEFAULT_XLSX_PATH,
    save_vybor_from_xlsx,
)
from src.features.phase1_schema import TARGET_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Vybor clinical CSV from main xlsx")
    p.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument(
        "--boku",
        type=Path,
        default=ROOT / "data" / "na_boku_full.bak.csv",
        help="Optional na_boku CSV for volume/length enrichment",
    )
    p.add_argument("--no-boku", action="store_true")
    p.add_argument(
        "--both-kidneys-only",
        action="store_true",
        help="Drop unilaterally labeled (single-kidney) rows",
    )
    p.add_argument(
        "--holdout",
        type=Path,
        default=DEFAULT_HOLDOUT_PATH,
        help="YAML with holdout surnames that must not appear in the CSV",
    )
    p.add_argument(
        "--include-holdout",
        action="store_true",
        help="Do not exclude holdout patients (for eval_holdout xlsx features only)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    boku = None if args.no_boku else args.boku
    if args.include_holdout:
        exclude: set[str] | list[str] = []
        banned_keys: set[str] = set()
    else:
        cfg = load_holdout_config(args.holdout)
        banned_keys = holdout_surname_keys(cfg)
        exclude = banned_keys
    df = save_vybor_from_xlsx(
        args.out,
        xlsx_path=args.xlsx,
        boku_path=boku,
        require_complete_targets=bool(args.both_kidneys_only),
        require_any_kidney_targets=not bool(args.both_kidneys_only),
        exclude_surnames=exclude,
    )
    if not args.include_holdout:
        assert_no_holdout_leak(df, banned_keys=banned_keys)
    labeled = (
        df["labeled_kidneys"].value_counts(dropna=False).to_dict()
        if "labeled_kidneys" in df.columns
        else {}
    )
    manifest = {
        "source_xlsx": str(args.xlsx),
        "output_csv": str(args.out),
        "rows": int(len(df)),
        "complete_targets": int(df[TARGET_NAMES].notna().all(axis=1).sum()),
        "any_kidney_targets": int(len(df)),
        "labeled_kidneys": {str(k): int(v) for k, v in labeled.items()},
        "boku_enrichment": str(boku) if boku else None,
        "holdout": None if args.include_holdout else str(args.holdout),
        "holdout_excluded": sorted(banned_keys),
        "data_origin": str(df["data_origin"].iloc[0]) if "data_origin" in df.columns and len(df) else None,
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
