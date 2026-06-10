#!/usr/bin/env python3
"""Unified CLI for Phase 1 feature extraction, integration, and training.

Prefer this entry point over ad-hoc script invocations to avoid pipeline drift.

Examples:
  python scripts/run_phase1_pipeline.py info
  python scripts/run_phase1_pipeline.py extract --dicom-root /data/dicom --output results/dicom.csv
  python scripts/run_phase1_pipeline.py integrate
  python scripts/run_phase1_pipeline.py train
  python scripts/run_phase1_pipeline.py validate --run-id my_run_001
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str]) -> int:
    print(f"[run] {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def cmd_info(_args: argparse.Namespace) -> int:
    from src.features.pipeline import print_canonical_flow

    print_canonical_flow()
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    if not args.dicom_root:
        print("error: --dicom-root is required for extract", file=sys.stderr)
        return 2
    out = args.output or str(ROOT / "results" / "dicom_features.csv")
    script = ROOT / "scripts" / "inference" / "enhanced_ct_extractor.py"
    cmd = [sys.executable, str(script), str(args.dicom_root), "--output", out]
    if args.max_slices:
        cmd.extend(["--max-slices", str(args.max_slices)])
    return _run(cmd)


def cmd_integrate(args: argparse.Namespace) -> int:
    script = ROOT / "src" / "models" / "data_integration_fix.py"
    cmd = [sys.executable, str(script), "--mode", args.mode]
    if args.no_kits_fill:
        cmd.append("--no-kits-fill")
    return _run(cmd)


def cmd_train(_args: argparse.Namespace) -> int:
    script = ROOT / "models" / "phase1" / "adaptive_ensemble.py"
    return _run([sys.executable, str(script)])


def cmd_validate(args: argparse.Namespace) -> int:
    dataset = args.dataset or ROOT / "data" / "processed" / "validation.csv"
    model = args.model or ROOT / "models" / "adaptive_ensemble.pkl"
    run_id = args.run_id
    if not run_id:
        print("error: --run-id is required for validate", file=sys.stderr)
        return 2

    rc = _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validation" / "smoke_check.py"),
            "--dataset",
            str(dataset),
            "--model",
            str(model),
        ]
    )
    if rc != 0:
        return rc

    metrics_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "validation" / "evaluate_metrics.py"),
        "--dataset",
        str(dataset),
        "--model",
        str(model),
        "--run-id",
        run_id,
        "--out-dir",
        str(args.out_dir),
        "--seed",
        str(args.seed),
        "--test-size",
        str(args.test_size),
    ]
    if args.source:
        metrics_cmd.extend(["--source", args.source])
    if args.holdout:
        metrics_cmd.append("--holdout")
    rc = _run(metrics_cmd)
    if rc != 0:
        return rc

    if args.visuals:
        return _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "validation" / "run_visual_tests.py"),
                "--dataset",
                str(dataset),
                "--model",
                str(model),
                "--run-id",
                run_id,
                "--out-dir",
                str(args.out_dir),
                "--seed",
                str(args.seed),
                "--test-size",
                str(args.test_size),
                "--num-cases",
                str(args.num_cases),
            ]
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 canonical ML pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Print canonical pipeline steps and legacy warnings")

    p_extract = sub.add_parser("extract", help="DICOM → CSV via enhanced_ct_extractor")
    p_extract.add_argument("--dicom-root", type=Path, required=True)
    p_extract.add_argument("--output", type=Path, default=None)
    p_extract.add_argument("--max-slices", type=int, default=None)

    p_integrate = sub.add_parser("integrate", help="Merge sources -> data/processed/train.csv")
    p_integrate.add_argument(
        "--mode",
        choices=["labeled_only", "all"],
        default="labeled_only",
        help="labeled_only: Vybor+Excel labels only (default)",
    )
    p_integrate.add_argument(
        "--no-kits-fill",
        action="store_true",
        help="Do not impute Excel sparse features from KiTS medians",
    )
    sub.add_parser("train", help="Train adaptive ensemble on data/processed/")

    p_validate = sub.add_parser("validate", help="Smoke + metrics (+ optional visuals)")
    p_validate.add_argument("--run-id", required=True)
    p_validate.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Default: data/processed/validation.csv",
    )
    p_validate.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Default: models/adaptive_ensemble.pkl",
    )
    p_validate.add_argument("--out-dir", type=Path, default=ROOT / "results" / "validation_runs")
    p_validate.add_argument("--seed", type=int, default=42)
    p_validate.add_argument("--test-size", type=float, default=0.5)
    p_validate.add_argument("--visuals", action="store_true", help="Also run visual tests")
    p_validate.add_argument("--num-cases", type=int, default=8)
    p_validate.add_argument(
        "--source",
        default=None,
        help="Filter eval by source column (e.g. Vybor)",
    )
    p_validate.add_argument(
        "--holdout",
        action="store_true",
        help="Evaluate full dataset file without re-split",
    )

    args = parser.parse_args()
    handlers = {
        "info": cmd_info,
        "extract": cmd_extract,
        "integrate": cmd_integrate,
        "train": cmd_train,
        "validate": cmd_validate,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
