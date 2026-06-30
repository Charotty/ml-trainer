#!/usr/bin/env python3
"""Build harmonized CSV copies under data/harmonized/ (originals untouched)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.coordinate_harmonization import (  # noqa: E402
    DEFAULT_HARMONIZED_DIR,
    DEFAULT_REFERENCE_CSV,
    alignment_report,
    build_reference_stats,
    default_harmonization_manifest,
    harmonize_file,
    load_reference_stats,
    save_reference_stats,
)
from src.features.phase1_schema import normalize_dataframe
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harmonize DICOM extracts to Vybor frame")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_HARMONIZED_DIR)
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_CSV)
    p.add_argument("--spine", type=Path, default=ROOT / "data" / "na_spine_full.csv")
    p.add_argument("--boku", type=Path, default=ROOT / "data" / "na_boku_full.bak.csv")
    p.add_argument("--dicom-merged", type=Path, default=ROOT / "data" / "dicom_medical_features.csv")
    p.add_argument("--kits19", type=Path, default=ROOT / "data" / "kits19_medical_grade_features.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_df = pd.read_csv(args.reference)
    ref_norm = normalize_dataframe(ref_df)
    stats = build_reference_stats(ref_norm)
    stats_path = out_dir / "reference_stats.json"
    save_reference_stats(stats, stats_path)

    ref_out = out_dir / "vybor_reference.csv"
    ref_norm.assign(
        harmonization_source="vybor",
        harmonization_applied="identity",
    ).to_csv(ref_out, index=False)

    outputs: dict[str, str] = {"vybor_reference.csv": str(ref_out)}
    reports = []

    jobs = [
        ("na_spine_full_aligned.csv", args.spine, "na_spine"),
        ("na_boku_full_aligned.csv", args.boku, "na_boku"),
        ("dicom_medical_features_aligned.csv", args.dicom_merged, "dicom_lps"),
        ("kits19_medical_grade_features_aligned.csv", args.kits19, "kits19"),
    ]

    for out_name, in_path, kind in jobs:
        if not in_path.exists():
            print(f"[skip] missing {in_path}")
            continue
        out_path = out_dir / out_name
        aligned = harmonize_file(in_path, out_path, stats, source_kind=kind)
        outputs[out_name] = str(out_path)
        rep = alignment_report(ref_norm, aligned)
        rep["dataset"] = out_name
        reports.append(rep)
        print(f"[ok] {in_path.name} -> {out_path} ({len(aligned)} rows)")

    if reports:
        qa = pd.concat(reports, ignore_index=True)
        qa_path = out_dir / "alignment_qa_report.csv"
        qa.to_csv(qa_path, index=False)
        outputs["alignment_qa_report.csv"] = str(qa_path)

    manifest = default_harmonization_manifest(
        reference_path=args.reference,
        outputs=outputs,
        transforms=[
            "normalize_dataframe (rel coords + distances)",
            "Y-axis sign flip (LPS -> Vybor clinical)",
            "robust IQR rescale per BASE feature to Vybor",
            "spine/body_com anchor to Vybor medians",
            "recompute distances",
        ],
    )
    manifest["reference_stats"] = str(stats_path)
    manifest_path = out_dir / "harmonization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
