#!/usr/bin/env python3
"""Batch DICOM extraction with UTF-8 paths (Windows-safe)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inference.enhanced_ct_extractor import (  # noqa: E402
    _add_unified_features,
    _iter_patient_folders,
    _normalize_name,
    extract_features_from_dicom_folder,
    _get_accuracy_params,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dicom_root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "dicom_batch_extract.csv")
    parser.add_argument("--accuracy-mode", default="fast", choices=["high", "balanced", "fast", "minimal"])
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    dicom_root = args.dicom_root
    if not dicom_root.exists():
        print(f"[FAIL] not found: {dicom_root}", flush=True)
        return 2

    params = _get_accuracy_params(args.accuracy_mode)
    folders = list(_iter_patient_folders(dicom_root))
    if args.max_cases:
        folders = folders[: args.max_cases]

    print(f"[scan] root={dicom_root} cases={len(folders)} mode={args.accuracy_mode}", flush=True)

    import pandas as pd

    rows = []
    for i, folder in enumerate(folders, start=1):
        print(f"[{i}/{len(folders)}] {folder.name}", flush=True)
        try:
            feats = extract_features_from_dicom_folder(
                folder,
                downsample=params["downsample"],
                max_slices=params["max_slices"],
                enable_kidney_segmentation=True,
                show_progress=False,
                slice_strategy=params["slice_strategy"],
            )
            feats = _add_unified_features(feats)
            row = {
                "case_id": folder.name,
                "full_name": feats.get("patient_name") or feats.get("full_name"),
                "dicom_folder": folder.name,
                "full_name_key": _normalize_name(str(feats.get("patient_name") or "")),
                "status": "extracted",
                "error": None,
                **feats,
            }
            rows.append(row)
            print(f"  [OK] slices={feats.get('slice_count_used')} vol_L={feats.get('kidney_left_volume_cm3')}", flush=True)
        except Exception as exc:
            print(f"  [WARN] {exc}", flush=True)
            rows.append({
                "case_id": folder.name,
                "dicom_folder": folder.name,
                "status": "error",
                "error": str(exc),
            })

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    ok = sum(1 for r in rows if r.get("status") == "extracted")
    print(f"[DONE] success={ok}/{len(rows)} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
