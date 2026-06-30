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

from src.data.xlsx_displacement_parser import (  # noqa: E402
    DEFAULT_OUTPUT_CSV,
    DEFAULT_XLSX_PATH,
    build_vybor_from_xlsx,
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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    boku = None if args.no_boku else args.boku
    df = save_vybor_from_xlsx(
        args.out,
        xlsx_path=args.xlsx,
        boku_path=boku,
    )
    manifest = {
        "source_xlsx": str(args.xlsx),
        "output_csv": str(args.out),
        "rows": int(len(df)),
        "complete_targets": int(df[TARGET_NAMES].notna().all(axis=1).sum()),
        "boku_enrichment": str(boku) if boku else None,
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
