#!/usr/bin/env python3
"""
Production launcher: builds argv as a list (avoids PowerShell/WSL/bash quoting
issues with Cyrillic, space-containing paths) and runs the DICOM pipeline for:

  * F:/На Боку  -> results/na_boku_full.csv   (update-existing: reprocess failed)
  * F:/На спине -> results/na_spine_full.csv  (fresh)

Usage (inside WSL venv):
  python scripts/tools/run_extract_jobs.py            # full run, both jobs
  python scripts/tools/run_extract_jobs.py --prep-only --max-cases 3   # quick check
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BOKU_ROOT = "/mnt/f/На Боку"
BOKU_OUT = str(ROOT / "results/na_boku_full.csv")
SPINE_ROOT = "/mnt/f/На спине"
SPINE_OUT = str(ROOT / "results/na_spine_full.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-update-existing", action="store_true")
    ap.add_argument("--temp-dir", default="/tmp/ml_trainer_dicom")
    args = ap.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts/inference/extract_from_dicom.py"),
        "--dicom-root", BOKU_ROOT,
        "--output", BOKU_OUT,
        "--add-job", SPINE_ROOT, SPINE_OUT,
        "--canonical",
        "--device", args.device,
        "--temp-dir", args.temp_dir,
    ]
    if not args.no_update_existing:
        cmd.append("--update-existing")
    if args.prep_only:
        cmd.append("--prep-only")
    if args.max_cases is not None:
        cmd += ["--max-cases", str(args.max_cases)]

    print("RUN:", " ".join(repr(c) if " " in c else c for c in cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
