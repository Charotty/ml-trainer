#!/usr/bin/env python3
"""Internal test runner — avoids Cyrillic paths on shell command line."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_drive_folder(part: str) -> Path:
    base = Path("/mnt/f")
    for p in base.iterdir():
        if part in p.name:
            return p
    raise FileNotFoundError(f"no folder matching {part!r} under {base}")


def main() -> int:
    from scripts.inference.dicom_prep import (
        discover_patient_cases,
        group_dicom_series,
        prepare_case,
        select_main_ct_series,
    )

    boku = find_drive_folder("Боку")
    spine = find_drive_folder("спине")

    boku_cases = discover_patient_cases(boku)
    spine_cases = discover_patient_cases(spine)
    print(f"boku: root={boku.name} cases={len(boku_cases)}")
    print(f"spine: root={spine.name} cases={len(spine_cases)}")

    case = boku_cases[0]
    sm = group_dicom_series(case)
    main = select_main_ct_series(sm)
    print(
        f"  first={case.name[:50]} series={len(sm)} "
        f"main_slices={main.slice_count if main else 0}"
    )

    prep = prepare_case(case, Path("/tmp/ml_dicom_test"), case_id="test_one", case_index=1)
    print("prep ok:", prep.work_slug, prep.nifti_file, "warnings:", prep.warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
