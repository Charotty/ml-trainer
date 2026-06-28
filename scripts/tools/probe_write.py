#!/usr/bin/env python3
import os
from pathlib import Path

results = Path(__file__).resolve().parents[2] / "results"
probe = results / "_wtest_probe.csv"
try:
    probe.write_text("x")
    probe.unlink()
    print("results/ writable: YES")
except Exception as e:
    print("results/ writable: NO", repr(e))

for name in ("na_boku_full.csv",):
    f = results / name
    try:
        with open(f, "a"):
            pass
        print(name, "append-open: OK (not locked)")
    except Exception as e:
        print(name, "append-open: FAIL", repr(e))
