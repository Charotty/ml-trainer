#!/usr/bin/env python3
"""Fast layout scan without reading every DICOM header."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def scan_patient(pf: Path) -> None:
    files = [f for f in pf.rglob("*") if f.is_file()]
    dicom = [f for f in files if is_dicom(f)]
    print(f"PAT {pf.name[:65]}")
    print(
        f"  files={len(files)} dicom={len(dicom)} "
        f"DICOMDIR={(pf / 'DICOMDIR').exists()} "
        f"cds={any(f.suffix.lower()=='.cds' for f in files)}"
    )
    print(f"  ext={dict(Counter(f.suffix.lower() for f in files).most_common(6))}")
    per_dir: list[tuple[int, str]] = []
    for sd in pf.rglob("*"):
        if not sd.is_dir():
            continue
        n = sum(1 for f in sd.iterdir() if f.is_file() and is_dicom(f))
        if n:
            per_dir.append((n, str(sd.relative_to(pf))))
    per_dir.sort(reverse=True)
    for n, rel in per_dir[:8]:
        print(f"  dir slices={n:4d}  {rel}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--max-patients", type=int, default=3)
    args = parser.parse_args()
    for root in args.roots:
        print("=" * 60)
        print("ROOT", root, "exists=", root.exists())
        if not root.exists():
            continue
        patients = sorted(p for p in root.iterdir() if p.is_dir())
        print("patients", len(patients))
        for pf in patients[: args.max_patients]:
            scan_patient(pf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
