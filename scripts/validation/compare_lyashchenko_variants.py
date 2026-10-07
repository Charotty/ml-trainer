#!/usr/bin/env python3
"""Score the four Lyashchenko coordinate variants against the displacement sheet.

Variants are the columns written by ``measure_lyashchenko``:

* body / canal — Y origin at the vertebral-body centre or the posterior canal point;
* com / near — centre of mass of the third, or the shortest distance to each axis.

Z is also scored from the centroid's projection onto L1, L2 and L3. The
vertical origin in the sheet is not the vertebra under the kidney, so those
three references are compared separately.

Usage
  py -3 scripts/validation/compare_lyashchenko_variants.py --supine results/na_spine_full.csv --lateral results/na_boku_full.csv --xlsx "F:/Смещение - конечное -13 .xlsx"

If the CSV was extracted before these columns existed, pass the segmentation
work directory. Each row is remeasured from ``seg_<work_slug>`` without
running TotalSegmentator again:

  py -3 scripts/validation/compare_lyashchenko_variants.py --supine results/na_spine_full.csv --work-dir E:/ml/dicom_work --xlsx "F:/Смещение - конечное -13 .xlsx"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.compare_ct_excel_displacement import (  # noqa: E402
    _ct_label,
    _excel_coord_records,
    match_tables,
)
from src.features.ct_anatomy.extract import load_anatomy_volume  # noqa: E402
from src.features.ct_anatomy.vertebral_axes import (  # noqa: E402
    ORIGINS,
    POINT_KINDS,
    measure_lyashchenko,
    score_variant_pairs,
    variant_column,
)

DEFAULT_XLSX = REPO_ROOT / "Смещение - конечное -13 .xlsx"
DEFAULT_OUT = REPO_ROOT / "results" / "ct_vs_excel"
ALIGNED = (
    "perirenal_dorsal_vert_mm",
    "perirenal_ventral_vert_mm",
    "psoas_thickness_vert_mm",
    "psoas_area_vert_cm2",
    "medial_to_spine_vert_mm",
)
EXCEL_ALIGNED = {
    "perirenal_dorsal_vert_mm": "perirenal_dorsal_mm",
    "perirenal_ventral_vert_mm": "perirenal_ventral_mm",
    "psoas_thickness_vert_mm": "psoas_thickness_mm",
    "psoas_area_vert_cm2": "psoas_area_cm2",
    "medial_to_spine_vert_mm": "medial_to_spine_mm",
}


def _windows_path(text: object) -> Optional[Path]:
    raw = str(text or "").strip()
    if not raw or raw.lower() == "nan":
        return None
    if raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5]
        rest = raw[6:].replace("/", "\\")
        return Path(f"{drive.upper()}:{rest}")
    return Path(raw)


def _has_variants(frame: pd.DataFrame) -> bool:
    return variant_column("left", "middle", "x", "body", "com") in frame.columns


def remeasure(frame: pd.DataFrame, work_dir: Path) -> pd.DataFrame:
    """Fill Lyashchenko columns from segmentation folders already on disk."""
    rows: List[dict] = []
    found = 0
    for _, row in frame.iterrows():
        item = row.to_dict()
        slug = str(item.get("work_slug") or "")
        seg_dir = work_dir / f"seg_{slug}"
        if not slug or not seg_dir.is_dir():
            rows.append(item)
            continue
        nifti = _windows_path(item.get("nifti_input_file"))
        hu = nifti if nifti is not None and nifti.exists() else None
        volume = load_anatomy_volume(seg_dir, hu)
        if volume is None:
            rows.append(item)
            continue
        item.update(measure_lyashchenko(volume))
        found += 1
        rows.append(item)
        print(f"  remeasured {found}: {item.get('patient_name') or slug}")
    print(f"remeasured {found} of {len(frame)} rows from {work_dir}")
    return pd.DataFrame(rows)


def _ct_records(frame: pd.DataFrame) -> List[dict]:
    records = []
    for _, row in frame.iterrows():
        label = _ct_label(row)
        item = {"label": label}
        item.update({key: row[key] for key in frame.columns})
        records.append(item)
    return records


def _aligned_scores(excel_anatomy: pd.DataFrame, ct_rows: List[dict], pose: str) -> List[dict]:
    if excel_anatomy.empty or "name_key" not in excel_anatomy.columns:
        return []
    from scripts.validation.validate_extractor_anatomy import surname_key

    by_name: Dict[str, dict] = {}
    for _, row in excel_anatomy.iterrows():
        by_name[str(row["name_key"])] = row.to_dict()
    manual: Dict[str, List[float]] = {name: [] for name in ALIGNED}
    auto: Dict[str, List[float]] = {name: [] for name in ALIGNED}
    for ct in ct_rows:
        excel = by_name.get(surname_key(ct.get("label")))
        if excel is None:
            continue
        for ct_suffix, excel_suffix in EXCEL_ALIGNED.items():
            for side in ("left", "right"):
                manual[ct_suffix].append(_as_float(excel.get(f"kidney_{side}_{excel_suffix}")))
                auto[ct_suffix].append(_as_float(ct.get(f"kidney_{side}_{ct_suffix}")))
    rows = []
    for suffix in ALIGNED:
        scored = _pair_score(manual[suffix], auto[suffix])
        scored["pose"] = pose
        scored["measurement"] = suffix
        rows.append(scored)
    return rows


def _as_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number


def _pair_score(manual: List[float], auto: List[float]) -> dict:
    import numpy as np

    manual_arr = np.asarray(manual, dtype=float)
    auto_arr = np.asarray(auto, dtype=float)
    ok = np.isfinite(manual_arr) & np.isfinite(auto_arr)
    diff = auto_arr[ok] - manual_arr[ok]
    if diff.size == 0:
        return {"n": 0, "mae": float("nan"), "bias_ct_minus_excel": float("nan"), "median_abs": float("nan")}
    return {
        "n": int(diff.size),
        "mae": float(np.mean(np.abs(diff))),
        "bias_ct_minus_excel": float(diff.mean()),
        "median_abs": float(np.median(np.abs(diff))),
    }


def _winner(summary: pd.DataFrame) -> str:
    """Lowest mean absolute error of middle-third X and Y."""
    if summary.empty:
        return "no paired rows"
    middle = summary[(summary["third"] == "middle") & (summary["axis"].isin(["x", "y"]))]
    middle = middle[middle["variant"].isin([f"{origin}_{kind}" for origin in ORIGINS for kind in POINT_KINDS])]
    if middle.empty or middle["n"].sum() == 0:
        return "no paired rows"
    grouped = middle.groupby("variant")["mae"].mean().sort_values()
    if grouped.empty or pd.isna(grouped.iloc[0]):
        return "no paired rows"
    best = str(grouped.index[0])
    z_rows = summary[(summary["axis"] == "z") & (summary["third"] == "middle") & (summary["n"] > 0)]
    z_text = ""
    if not z_rows.empty:
        z_best = z_rows.sort_values("mae").iloc[0]
        z_text = f"; middle Z closest with {z_best['variant']} (MAE {z_best['mae']:.1f} mm)"
    return f"middle X/Y closest with {best} (MAE {grouped.iloc[0]:.1f} mm){z_text}"


def score_pose(
    excel_rows: List[dict],
    ct_frame: pd.DataFrame,
    pose: str,
    excel_anatomy: pd.DataFrame,
) -> tuple:
    pairs, skipped = match_tables(excel_rows, _ct_records(ct_frame))
    coordinate_rows = score_variant_pairs(pairs, pose)
    for row in coordinate_rows:
        row["pose"] = pose
    aligned_rows = _aligned_scores(excel_anatomy, [pair[1] for pair in pairs], pose)
    return coordinate_rows, aligned_rows, len(pairs), len(skipped)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Lyashchenko axis variants with the displacement workbook")
    parser.add_argument("--supine", type=Path, help="Supine extract CSV")
    parser.add_argument("--lateral", type=Path, help="Lateral extract CSV")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--work-dir", type=Path, help="Directory of seg_<work_slug> folders to remeasure")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.supine is None and args.lateral is None:
        print("pass --supine and/or --lateral")
        return 2
    if not args.xlsx.exists():
        print(f"xlsx not found: {args.xlsx}")
        return 2

    excel_rows = _excel_coord_records(args.xlsx)
    from scripts.validation.validate_extractor_anatomy import read_manual_table

    excel_anatomy = read_manual_table(args.xlsx)
    coordinate_rows: List[dict] = []
    aligned_rows: List[dict] = []
    for pose, path in (("supine", args.supine), ("lateral", args.lateral)):
        if path is None:
            continue
        if not path.exists():
            print(f"extract csv not found: {path}")
            return 2
        frame = pd.read_csv(path)
        if args.work_dir is not None:
            if not args.work_dir.is_dir():
                print(f"work dir not found: {args.work_dir}")
                return 2
            frame = remeasure(frame, args.work_dir)
            filled = path.with_name(path.stem + "_lyashchenko.csv")
            frame.to_csv(filled, index=False)
            print(f"wrote {filled}")
        if not _has_variants(frame):
            print(f"{path.name} has no Lyashchenko columns. Pass --work-dir with the segmentation folders.")
            continue
        scored, aligned, paired, skipped = score_pose(excel_rows, frame, pose, excel_anatomy)
        coordinate_rows.extend(scored)
        aligned_rows.extend(aligned)
        print(f"{pose}: paired {paired}, surname collisions held out {skipped}")

    if not coordinate_rows:
        print("nothing scored")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(coordinate_rows)
    summary_path = args.output_dir / "lyashchenko_variant_summary.csv"
    summary.to_csv(summary_path, index=False)
    aligned_path = args.output_dir / "lyashchenko_aligned_summary.csv"
    pd.DataFrame(aligned_rows).to_csv(aligned_path, index=False)
    for pose in ("supine", "lateral"):
        part = summary[summary["pose"] == pose]
        if not part.empty:
            print(f"{pose}: {_winner(part)}")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
