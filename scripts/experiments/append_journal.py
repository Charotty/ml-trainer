#!/usr/bin/env python3
"""Append deferred Variant A runs to the journal in a fixed order.

Usage: py -3 scripts/experiments/append_journal.py <run_id> [<run_id> ...]
       py -3 scripts/experiments/append_journal.py --skip "<stage>" "<change>" "<reason>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_variant_a import (  # noqa: E402
    EXPERIMENTS_DIR,
    _git_commit,
    append_journal_row,
    finalize_run,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_ids", nargs="*")
    p.add_argument("--skip", nargs=3, action="append", metavar=("STAGE", "CHANGE", "REASON"))
    args = p.parse_args()
    for run_id in args.run_ids:
        path = EXPERIMENTS_DIR / f"{run_id}_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        out = finalize_run(path)
        print(run_id, out["verdict"], flush=True)
    for stage, change, reason in args.skip or []:
        append_journal_row(
            {
                "stage": stage,
                "change": f"{change} — пропущено: {reason}",
                "commit": _git_commit(),
                "mae_avg": "—",
                "ci95": "—",
                "mae_x": "—",
                "mae_y": "—",
                "mae_z": "—",
                "mae_zl": "—",
                "mae_zr": "—",
                "r2": "—",
                "err_3d": "—",
                "within10": "—",
                "delta_best": "—",
                "holdout": "—",
                "verdict": "пропущено",
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
