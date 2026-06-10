#!/usr/bin/env python3
"""Environment and artifact smoke checks for WSL runs."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


REQUIRED_MODULES = ["numpy", "pandas", "sklearn", "joblib", "matplotlib", "fastapi"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/vybor_unified_features.csv",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--model",
        default="models/adaptive_ensemble.pkl",
        help="Path to model artifact (optional, warning if absent)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok = True

    print("== Module imports ==")
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {module_name}")
        except Exception as exc:  # pragma: no cover - smoke output
            ok = False
            print(f"[FAIL] {module_name}: {exc}")

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)

    print("\n== Artifacts ==")
    if dataset_path.exists():
        print(f"[OK] dataset: {dataset_path}")
    else:
        ok = False
        print(f"[FAIL] dataset missing: {dataset_path}")

    if model_path.exists():
        print(f"[OK] model: {model_path}")
    else:
        print(f"[WARN] model missing: {model_path}")
        print("       Validation will fallback to on-the-fly RandomForest baseline.")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
