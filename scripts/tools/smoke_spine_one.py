#!/usr/bin/env python3
"""Smoke test: 1 nested 'На спине' case end-to-end + /tmp hygiene check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    out = ROOT / "results/smoke_spine_one.csv"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/inference/extract_from_dicom.py"),
        "--add-job",
        "/mnt/f/На спине",
        str(out),
        "--canonical",
        "--device",
        "auto",
        "--max-cases",
        "1",
        "--temp-dir",
        "/tmp/ml_trainer_dicom",
    ]
    print("RUN:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(ROOT))

    leftovers = list(Path("/tmp/ml_trainer_dicom").glob("*")) if Path("/tmp/ml_trainer_dicom").exists() else []
    print("\n/tmp leftovers after run:", [p.name for p in leftovers])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
