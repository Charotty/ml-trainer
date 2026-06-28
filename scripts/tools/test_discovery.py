#!/usr/bin/env python3
"""Discovery + syntax smoke test for both CT roots."""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    for f in ("scripts/inference/dicom_prep.py", "scripts/inference/extract_from_dicom.py"):
        ast.parse((ROOT / f).read_text(encoding="utf-8"))
        print("syntax OK:", f)

    from scripts.inference.dicom_prep import discover_patient_cases

    for label, root in (("boku", "/mnt/f/На Боку"), ("spine", "/mnt/f/На спине")):
        t = time.time()
        cases = discover_patient_cases(Path(root))
        print(f"{label}: cases={len(cases)} in {time.time() - t:.1f}s")
        for c in cases[:5]:
            print("   ", c.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
