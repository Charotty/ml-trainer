#!/usr/bin/env python3
"""Quick DICOM layout analysis for patient export folders (amImageViewer-style)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pydicom


def is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def iter_dicom_files(folder: Path) -> list[Path]:
    out: list[Path] = []
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".exe", ".inf", ".cds", ".txt", ".html", ".xml"}:
            continue
        if p.suffix.lower() in {"", ".dcm"} and is_dicom(p):
            out.append(p)
    return out


def analyze_patient(patient_folder: Path) -> dict:
    files = [f for f in patient_folder.rglob("*") if f.is_file()]
    dicom_files = iter_dicom_files(patient_folder)
    series: dict[str, list] = defaultdict(list)
    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            suid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
            series[suid].append(ds)
        except Exception:
            continue

    series_rows = []
    for suid, dss in sorted(series.items(), key=lambda x: -len(x[1])):
        ds0 = dss[0]
        inst_nums = [getattr(d, "InstanceNumber", None) for d in dss]
        inst_nums_int = [int(x) for x in inst_nums if x is not None]
        gaps = 0
        if inst_nums_int:
            inst_sorted = sorted(inst_nums_int)
            gaps = sum(
                1 for a, b in zip(inst_sorted, inst_sorted[1:]) if b - a > 1
            )
        series_rows.append(
            {
                "slices": len(dss),
                "modality": str(getattr(ds0, "Modality", "")),
                "description": str(getattr(ds0, "SeriesDescription", ""))[:60],
                "series_number": getattr(ds0, "SeriesNumber", None),
                "instance_gaps": gaps,
                "number_of_frames": getattr(ds0, "NumberOfFrames", None),
                "rows": getattr(ds0, "Rows", None),
                "cols": getattr(ds0, "Columns", None),
            }
        )

    return {
        "patient": patient_folder.name,
        "total_files": len(files),
        "dicom_files": len(dicom_files),
        "extensions": dict(Counter(f.suffix.lower() for f in files).most_common(8)),
        "series_count": len(series_rows),
        "series": series_rows,
        "has_dicomdir": (patient_folder / "DICOMDIR").exists(),
        "has_cds": any(f.suffix.lower() == ".cds" for f in files),
    }


def analyze_root(root: Path, max_patients: int) -> None:
    print(f"\n{'=' * 72}\nROOT: {root}\n{'=' * 72}")
    if not root.exists():
        print("NOT FOUND")
        return

    patients = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"patient_folders: {len(patients)}")
    agg_series_counts: Counter[int] = Counter()
    agg_ct_main_slices: list[int] = []

    for pf in patients[:max_patients]:
        info = analyze_patient(pf)
        print(f"\n-- {info['patient'][:70]}")
        print(
            f"   files={info['total_files']} dicom={info['dicom_files']} "
            f"series={info['series_count']} DICOMDIR={info['has_dicomdir']} CDS={info['has_cds']}"
        )
        print(f"   extensions: {info['extensions']}")
        for i, s in enumerate(info["series"][:8], 1):
            print(
                f"   [{i}] n={s['slices']:4d} mod={s['modality']:<4} "
                f"gaps={s['instance_gaps']:3d} desc={s['description']!r}"
            )
        agg_series_counts[info["series_count"]] += 1
        ct_series = [s for s in info["series"] if s["modality"] == "CT"]
        if ct_series:
            agg_ct_main_slices.append(max(s["slices"] for s in ct_series))

    if patients[max_patients:]:
        print(f"\n... skipped {len(patients) - max_patients} more patients")

    if agg_ct_main_slices:
        print(
            f"\nSummary (first {min(max_patients, len(patients))} patients): "
            f"CT main series slices min/median/max = "
            f"{min(agg_ct_main_slices)}/"
            f"{sorted(agg_ct_main_slices)[len(agg_ct_main_slices)//2]}/"
            f"{max(agg_ct_main_slices)}"
        )
    print(f"series-per-patient histogram (sample): {dict(sorted(agg_series_counts.items()))}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--max-patients", type=int, default=5)
    args = parser.parse_args()
    for root in args.roots:
        analyze_root(root, args.max_patients)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
