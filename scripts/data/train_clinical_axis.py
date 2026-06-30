#!/usr/bin/env python3
"""Train axis-improved model on clinical xlsx only (honest 80/20 split)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "phase1"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

import pandas as pd
from sklearn.model_selection import train_test_split

from adaptive_ensemble import AdaptiveEnsembleTrainer  # noqa: E402
from src.data.xlsx_displacement_parser import DEFAULT_OUTPUT_CSV  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402

MODEL_PATH = ROOT / "models" / "adaptive_ensemble_clinical_axis.pkl"
RUN_ID = f"clinical_axis_only_{date.today().strftime('%Y%m%d')}"


def main() -> int:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "build_vybor_from_xlsx.py")],
        cwd=str(ROOT),
        check=True,
    )
    vybor = normalize_dataframe(pd.read_csv(DEFAULT_OUTPUT_CSV))
    train_df, val_df = train_test_split(vybor, test_size=0.2, random_state=42)

    trainer = AdaptiveEnsembleTrainer()
    X_train, X_val, y_train, y_val = trainer.prepare_training_data_split(train_df, val_df)
    print(f"[clinical] features={len(trainer.feature_names)}, train={len(train_df)}, val={len(val_df)}")
    trainer.train_and_evaluate_adaptive_ensembles(X_train, X_val, y_train, y_val)
    trainer.save_model(str(MODEL_PATH))

    from src.features.pipeline import apply_model_preprocessing, build_inference_matrix

    Xv = apply_model_preprocessing(
        build_inference_matrix(trainer, val_df, feature_names=trainer.feature_names),
        {"imputer": trainer.imputer, "scaler": trainer.scaler},
    )
    metrics = {}
    for tgt in TARGET_NAMES:
        pred = trainer.trained_models[tgt].predict(Xv)
        err = abs(val_df[tgt].astype(float).values - pred)
        metrics[tgt] = {"mae_mm": float(err.mean()), "within_5mm": float((err <= 5).mean())}
    yz = [metrics[t]["mae_mm"] for t in TARGET_NAMES if t.endswith("_y") or t.endswith("_z")]
    metrics["yz_avg_mae"] = sum(yz) / len(yz)

    out_dir = ROOT / "results" / "validation_runs" / RUN_ID / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "holdout_val_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validation" / "evaluate_metrics.py"),
            "--dataset",
            str(DEFAULT_OUTPUT_CSV),
            "--model",
            str(MODEL_PATH),
            "--run-id",
            RUN_ID,
            "--holdout",
        ],
        cwd=str(ROOT),
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
