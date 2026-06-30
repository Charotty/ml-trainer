#!/usr/bin/env python3
"""Fit left-kidney Z post-hoc calibrator and run axis validation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "phase1"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

from adaptive_ensemble import AdaptiveEnsembleTrainer  # noqa: E402
from common import compute_regression_table, predict_df  # noqa: E402
from src.data.xlsx_displacement_parser import DEFAULT_OUTPUT_CSV  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402
from src.features.pipeline import apply_model_preprocessing, build_inference_matrix  # noqa: E402
from src.models.left_z_calibrator import TARGET as LEFT_Z_TARGET, LeftZCalibrator  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "adaptive_ensemble_clinical_axis.pkl"
OUTPUT_MODEL = ROOT / "models" / "adaptive_ensemble_clinical_axis_calibrated.pkl"
RUN_ID = f"clinical_axis_calibrated_{date.today().strftime('%Y%m%d')}"
SEED = 42


def _raw_predictions(trainer: AdaptiveEnsembleTrainer, df: pd.DataFrame) -> pd.DataFrame:
    df_norm = normalize_dataframe(df)
    X = build_inference_matrix(trainer, df_norm, feature_names=trainer.feature_names)
    X_scaled = apply_model_preprocessing(
        X,
        {"imputer": trainer.imputer, "scaler": trainer.scaler},
    )
    rows = {}
    for tgt in trainer.target_names:
        rows[tgt] = trainer.trained_models[tgt].predict(X_scaled)
    return pd.DataFrame(rows, index=df.index)


def _axis_summary(metrics_df: pd.DataFrame) -> dict:
    axes = {"x": [], "y": [], "z": []}
    for _, row in metrics_df.iterrows():
        axis = row["target"].split("_")[-1]
        if axis in axes:
            axes[axis].append(float(row["mae_mm"]))
    return {
        axis: {
            "mae_mm": float(np.mean(vals)),
            "targets": len(vals),
        }
        for axis, vals in axes.items()
        if vals
    }


def _metrics_block(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    err = np.abs(y_true.astype(float).values - y_pred)
    return {
        "mae_mm": float(err.mean()),
        "rmse_mm": float(np.sqrt(np.mean((y_true.astype(float).values - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "within_5mm": float((err <= 5).mean()),
        "within_10mm": float((err <= 10).mean()),
    }


def main() -> int:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_MODEL

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "build_vybor_from_xlsx.py")],
        cwd=str(ROOT),
        check=True,
    )
    vybor = normalize_dataframe(pd.read_csv(DEFAULT_OUTPUT_CSV))
    train_df, val_df = train_test_split(vybor, test_size=0.2, random_state=SEED)

    payload = joblib.load(model_path)
    trainer = AdaptiveEnsembleTrainer()
    trainer.feature_names = payload["feature_names"]
    trainer.target_names = payload.get("target_names", list(payload["models"].keys()))
    trainer.trained_models = payload["models"]
    trainer.imputer = payload["imputer"]
    trainer.scaler = payload["scaler"]

    train_raw = _raw_predictions(trainer, train_df)
    calibrator = LeftZCalibrator()
    calibrator.fit(
        train_df,
        train_raw[LEFT_Z_TARGET].values,
        train_df[LEFT_Z_TARGET].astype(float).values,
    )

    val_raw = _raw_predictions(trainer, val_df)
    val_cal = val_raw.copy()
    val_cal[LEFT_Z_TARGET] = calibrator.transform(val_df, val_raw[LEFT_Z_TARGET].values)

    before = _metrics_block(val_df[LEFT_Z_TARGET], val_raw[LEFT_Z_TARGET].values)
    after = _metrics_block(val_df[LEFT_Z_TARGET], val_cal[LEFT_Z_TARGET].values)
    print(
        f"[left Z holdout] before MAE={before['mae_mm']:.2f} mm -> "
        f"after MAE={after['mae_mm']:.2f} mm "
        f"(delta {after['mae_mm'] - before['mae_mm']:+.2f} mm)"
    )
    print(f"[left Z calibrator] {json.dumps(calibrator.describe(), ensure_ascii=False)}")

    payload["left_z_calibrator"] = calibrator
    joblib.dump(payload, out_path)
    print(f"[OK] Saved calibrated model: {out_path}")

    bundle = type(
        "Bundle",
        (),
        {
            "mode": "pretrained_adaptive_ensemble",
            "feature_names": trainer.feature_names,
            "target_names": trainer.target_names,
            "scaler": trainer.scaler,
            "imputer": trainer.imputer,
            "models": trainer.trained_models,
            "left_z_calibrator": calibrator,
        },
    )()

    val_pred = predict_df(bundle, val_df)
    per_target = compute_regression_table(
        val_df[TARGET_NAMES],
        val_pred,
        list(TARGET_NAMES),
    )
    axis_summary = _axis_summary(per_target)

    report = {
        "model_path": str(out_path),
        "holdout_n": len(val_df),
        "left_z_calibration": {"before": before, "after": after},
        "per_target": per_target.set_index("target")["mae_mm"].to_dict(),
        "axis_mae_mm": {k: v["mae_mm"] for k, v in axis_summary.items()},
    }
    run_dir = ROOT / "results" / "validation_runs" / RUN_ID / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "axis_holdout_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    per_target.to_csv(run_dir / "metrics_per_target_holdout.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        val_df.to_csv(tmp.name, index=False)
        val_csv = tmp.name

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validation" / "evaluate_metrics.py"),
            "--dataset",
            val_csv,
            "--model",
            str(out_path),
            "--run-id",
            RUN_ID,
            "--holdout",
        ],
        cwd=str(ROOT),
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validation" / "evaluate_metrics.py"),
            "--dataset",
            str(DEFAULT_OUTPUT_CSV),
            "--model",
            str(out_path),
            "--run-id",
            f"{RUN_ID}_full87",
            "--holdout",
        ],
        cwd=str(ROOT),
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
