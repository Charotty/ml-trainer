#!/usr/bin/env python3
"""Build anatomy reference statistics from KiTS19 (features only, no targets)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES, normalize_dataframe

DEFAULT_KITS = ROOT / "data" / "kits19_medical_grade_features.csv"
OUT_DIR = ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KiTS19 feature reference (no displacement targets)")
    parser.add_argument("--kits-csv", type=Path, default=DEFAULT_KITS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.kits_csv.exists():
        print(f"[FAIL] KiTS19 CSV not found: {args.kits_csv}")
        return 2

    df = normalize_dataframe(pd.read_csv(args.kits_csv))
    feature_df = df[[c for c in BASE_FEATURES if c in df.columns]].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = args.out_dir / "kits19_feature_reference.csv"
    feature_df.to_csv(ref_path, index=False)

    medians = {}
    for col in BASE_FEATURES:
        if col in feature_df.columns:
            medians[col] = float(feature_df[col].median(skipna=True))

    medians_path = args.out_dir / "kits19_feature_medians.json"
    with open(medians_path, "w", encoding="utf-8") as fh:
        json.dump(medians, fh, indent=2, ensure_ascii=False)

    manifest = {
        "source": str(args.kits_csv),
        "rows": int(len(feature_df)),
        "base_features": list(BASE_FEATURES),
        "targets_stripped": list(TARGET_NAMES),
        "note": "Reference anatomy only — do NOT use KiTS delta columns for training labels.",
    }
    manifest_path = args.out_dir / "kits19_feature_reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Reference features: {ref_path} ({len(feature_df)} rows)")
    print(f"[OK] Medians: {medians_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
