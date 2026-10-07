#!/usr/bin/env python3
"""Build data/vybor_from_xlsx.csv without the displacement XLSX.

Labels come from cached clinical Vybor rows in integrated_master_dataset.csv
(87 paired supine/lateral δ). na_spine / na_boku extraction CSVs are unpaired
cohorts and supply na_trend features only (see train_clinical_honest.py).

Usage:
  python scripts/data/build_vybor_from_csv.py
  python scripts/data/build_vybor_from_csv.py --integrated data/integrated_master_dataset.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.xlsx_displacement_parser import DEFAULT_OUTPUT_CSV  # noqa: E402
from src.features.phase1_schema import (  # noqa: E402
    TARGET_NAMES,
    filter_any_kidney_targets,
    labeled_kidneys_series,
    normalize_dataframe,
)

DEFAULT_INTEGRATED = ROOT / "data" / "integrated_master_dataset.csv"


def build_vybor_from_integrated(integrated_path: Path | str) -> pd.DataFrame:
    path = Path(integrated_path)
    if not path.exists():
        raise FileNotFoundError(f"Integrated master not found: {path}")

    raw = pd.read_csv(path)
    if "source" not in raw.columns:
        raise ValueError(f"{path} has no 'source' column — expected integrated master format")

    vybor = raw[raw["source"] == "Vybor"].copy()
    if vybor.empty:
        raise ValueError(f"No Vybor rows in {path}")

    vybor = normalize_dataframe(vybor)
    before = len(vybor)
    vybor = filter_any_kidney_targets(vybor).reset_index(drop=True)
    vybor["labeled_kidneys"] = labeled_kidneys_series(vybor)
    skipped = before - len(vybor)
    if skipped:
        print(
            f"[csv] Skipped {skipped} Vybor rows with no labeled kidney side "
            f"(kept {len(vybor)})"
        )

    vybor["source"] = "Vybor"
    vybor["source_name"] = "Vybor"
    vybor["label_quality"] = "clinical"
    vybor["data_origin"] = str(path)
    return vybor


def save_vybor_from_csv(
    output_path: Path | str = DEFAULT_OUTPUT_CSV,
    *,
    integrated_path: Path | str = DEFAULT_INTEGRATED,
) -> pd.DataFrame:
    df = build_vybor_from_integrated(integrated_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[csv] Saved {len(df)} clinical rows -> {output_path}")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Vybor clinical CSV from integrated master (no XLSX)")
    p.add_argument("--integrated", type=Path, default=DEFAULT_INTEGRATED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_CSV)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = save_vybor_from_csv(args.out, integrated_path=args.integrated)
    labeled = (
        df["labeled_kidneys"].value_counts(dropna=False).to_dict()
        if "labeled_kidneys" in df.columns
        else {}
    )
    manifest = {
        "source_integrated": str(args.integrated),
        "output_csv": str(args.out),
        "rows": int(len(df)),
        "complete_targets": int(df[TARGET_NAMES].notna().all(axis=1).sum()),
        "labeled_kidneys": {str(k): int(v) for k, v in labeled.items()},
        "xlsx_required": False,
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
