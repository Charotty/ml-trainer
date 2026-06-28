#!/usr/bin/env python3
"""One-case end-to-end test (ASCII temp paths, Cyrillic patient folder)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def find_drive_folder(part: str) -> Path:
    for p in Path("/mnt/f").iterdir():
        if part in p.name:
            return p
    raise FileNotFoundError(part)


def main() -> int:
    boku = find_drive_folder("Боку")
    cmd = [
        sys.executable,
        str(ROOT / "scripts/inference/extract_from_dicom.py"),
        "--dicom-root",
        str(boku),
        "--canonical",
        "--device",
        "auto",
        "--max-cases",
        "1",
        "--output",
        str(ROOT / "results/test_one_ascii_slug.csv"),
        "--temp-dir",
        "/tmp/ml_trainer_dicom",
    ]
    print("RUN:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
