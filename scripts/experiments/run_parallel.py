#!/usr/bin/env python3
"""Run several Variant A experiments in parallel with deferred journaling.

Spec file: JSON list of {"run_id": ..., "args": [...]} (args for run_variant_a.py).
Each run writes models/experiments/<run_id>_metrics.json; journal rows are
appended afterwards in spec order by append_journal.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "models" / "experiments"
LOGS = ROOT / "results" / "variant_a_logs"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("spec", type=Path)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--journal", action="store_true", help="Append rows in spec order when done")
    args = p.parse_args()
    specs = json.loads(args.spec.read_text(encoding="utf-8"))
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    pending = list(specs)
    running: list[tuple[dict, subprocess.Popen, object]] = []
    failed: list[str] = []
    while pending or running:
        while pending and len(running) < args.jobs:
            spec = pending.pop(0)
            rid = spec["run_id"]
            if (EXP / f"{rid}_metrics.json").is_file():
                print(f"[skip] {rid} already has metrics", flush=True)
                continue
            log = open(LOGS / f"{rid}.log", "w", encoding="utf-8")
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "experiments" / "run_variant_a.py"),
                "--run-id",
                rid,
                "--defer-journal",
                "--skip-build",
                *spec["args"],
            ]
            print(f"[start] {rid}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((spec, proc, log))
        time.sleep(10)
        still = []
        for spec, proc, log in running:
            if proc.poll() is None:
                still.append((spec, proc, log))
                continue
            log.close()
            status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            if proc.returncode != 0:
                failed.append(spec["run_id"])
            print(f"[done] {spec['run_id']} {status}", flush=True)
        running = still
    if args.journal:
        ids = [s["run_id"] for s in specs if s["run_id"] not in failed]
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "experiments" / "append_journal.py"), *ids],
            cwd=str(ROOT),
            check=True,
            env=env,
        )
    if failed:
        print("FAILED:", failed, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
