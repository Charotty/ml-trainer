#!/usr/bin/env python3
"""Compare holdout metrics across displacement model artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

from common import compute_regression_table, predict_df  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402

DEFAULT_VAL = ROOT / "data" / "processed_full_extended" / "validation.csv"
DEFAULT_MODELS = [
    ("full_extended", ROOT / "models" / "adaptive_ensemble_full_extended.pkl"),
    ("clinical_calibrated", ROOT / "models" / "adaptive_ensemble_clinical_axis_calibrated.pkl"),
    ("phase2", ROOT / "models" / "adaptive_ensemble_phase2.pkl"),
    ("improved_v1", ROOT / "models" / "adaptive_ensemble_improved_v1.pkl"),
]

LEFT_Z = "kidney_left_delta_z"
RIGHT_Z = "kidney_right_delta_z"


def load_bundle(path: Path):
    payload = joblib.load(path)
    return type(
        "Bundle",
        (),
        {
            "mode": "pretrained_adaptive_ensemble",
            "feature_names": payload["feature_names"],
            "target_names": list(payload.get("target_names", payload["models"].keys())),
            "scaler": payload["scaler"],
            "imputer": payload.get("imputer"),
            "models": payload["models"],
            "left_z_calibrator": payload.get("left_z_calibrator"),
            "right_z_calibrator": payload.get("right_z_calibrator"),
            "side_z_models": payload.get("side_z_models"),
            "multitask_model": payload.get("multitask_model"),
            "multitask_blend": payload.get("multitask_blend"),
            "quantile_model": payload.get("quantile_model"),
        },
    )()


def eval_model(name: str, path: Path, val_df: pd.DataFrame) -> dict:
    bundle = load_bundle(path)
    pred = predict_df(bundle, val_df)
    per_target = compute_regression_table(val_df[TARGET_NAMES], pred, list(TARGET_NAMES))
    mae_map = per_target.set_index("target")["mae_mm"].to_dict()
    lz = float(mae_map.get(LEFT_Z, np.nan))
    rz = float(mae_map.get(RIGHT_Z, np.nan))
    z_avg = float((lz + rz) / 2.0) if np.isfinite(lz) and np.isfinite(rz) else np.nan

    q_cov = {}
    quantile = getattr(bundle, "quantile_model", None)
    if quantile is not None and getattr(quantile, "fitted_", False):
        from adaptive_ensemble import AdaptiveEnsembleTrainer
        from src.features.pipeline import apply_model_preprocessing, build_inference_matrix

        trainer = AdaptiveEnsembleTrainer()
        X = build_inference_matrix(trainer, val_df, feature_names=bundle.feature_names)
        X_scaled = apply_model_preprocessing(
            X,
            {"imputer": bundle.imputer, "scaler": bundle.scaler},
        )
        y = val_df[list(TARGET_NAMES)].astype(float).values
        q_cov = quantile.coverage_rate(X_scaled, y, bundle.target_names)

    return {
        "model": name,
        "path": str(path),
        "n": len(val_df),
        "per_target_mae_mm": mae_map,
        "avg_mae_mm": float(per_target["mae_mm"].mean()),
        "left_z_mae_mm": lz,
        "right_z_mae_mm": rz,
        "z_avg_mae_mm": z_avg,
        "quantile_coverage_80": q_cov,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare displacement models on holdout")
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    val_df = normalize_dataframe(pd.read_csv(args.val_csv))
    rows = []
    for name, path in DEFAULT_MODELS:
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        print(f"[eval] {name} ...")
        rows.append(eval_model(name, path, val_df))

    if not rows:
        print("No models evaluated.")
        return 1

    summary = pd.DataFrame(
        [
            {
                "model": r["model"],
                "avg_mae_mm": r["avg_mae_mm"],
                "left_z_mae_mm": r["left_z_mae_mm"],
                "right_z_mae_mm": r["right_z_mae_mm"],
                "z_avg_mae_mm": r["z_avg_mae_mm"],
            }
            for r in rows
        ]
    ).sort_values("z_avg_mae_mm")

    best_z = summary.iloc[0]["model"]
    best_avg = summary.sort_values("avg_mae_mm").iloc[0]["model"]

    report = {
        "val_csv": str(args.val_csv),
        "n_holdout": len(val_df),
        "models": rows,
        "ranking": {
            "best_z_avg": best_z,
            "best_overall_avg": best_avg,
        },
        "summary_table": summary.to_dict(orient="records"),
    }

    out_path = args.out or (
        ROOT / "results" / "validation_runs" / "model_comparison_holdout18.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Holdout comparison (lower is better) ===")
    print(summary.to_string(index=False))
    print(f"\nBest Z avg: {best_z}")
    print(f"Best overall avg: {best_avg}")
    print(f"Report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
