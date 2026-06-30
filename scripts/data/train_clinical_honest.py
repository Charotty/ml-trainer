#!/usr/bin/env python3
"""Honest clinical training: fixes 2-7 (no leakage, anatomical frame, GKF, OOF calibrators)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

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
from src.models.z_calibrator_oof import SideZCalibrator, fit_calibrator_oof_gated  # noqa: E402

DEFAULT_MODEL_PATH = ROOT / "models" / "adaptive_ensemble_clinical_honest.pkl"
SEED = 42
N_SPLITS = 5
N_BOOTSTRAP = 2000
Z_TARGETS = ["kidney_left_delta_z", "kidney_right_delta_z"]


def _bootstrap_ci(per_patient_avg: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(per_patient_avg)
    means = [per_patient_avg[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _axis_summary(per_target: pd.DataFrame) -> dict[str, float]:
    axes: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    for _, row in per_target.iterrows():
        axis = row["target"].split("_")[-1]
        if axis in axes:
            axes[axis].append(float(row["mae_mm"]))
    return {axis: float(np.mean(vals)) for axis, vals in axes.items() if vals}


def _make_bundle(trainer, left_cal, right_cal):
    return type(
        "Bundle",
        (),
        {
            "mode": "pretrained_adaptive_ensemble",
            "feature_names": trainer.feature_names,
            "target_names": trainer.target_names,
            "scaler": trainer.scaler,
            "imputer": trainer.imputer,
            "models": trainer.trained_models,
            "left_z_calibrator": left_cal,
            "right_z_calibrator": right_cal,
            "z_head": trainer.z_head,
            "z_driver_names": trainer.z_driver_names,
        },
    )()


def evaluate_groupkfold_oof(df: pd.DataFrame, *, z_head: str = "ensemble") -> dict:
    """OOF predictions with patient GroupKFold (honest protocol, step 7)."""
    name_col = "full_name" if "full_name" in df.columns else "case_id"
    groups = df[name_col].astype(str).values
    gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))

    oof = {t: np.full(len(df), np.nan) for t in TARGET_NAMES}
    for train_idx, val_idx in gkf.split(df, df[TARGET_NAMES[0]], groups=groups):
        tr = df.iloc[train_idx].reset_index(drop=True)
        te = df.iloc[val_idx].reset_index(drop=True)
        fold_trainer = AdaptiveEnsembleTrainer(z_head=z_head)
        X_tr, X_te, y_tr, y_te = fold_trainer.prepare_training_data_split(tr, te)
        g_tr = tr[name_col].astype(str).values
        fold_trainer.train_and_evaluate_adaptive_ensembles(
            X_tr, X_te, y_tr, y_te, groups=g_tr, fast_weights=True
        )
        pred = predict_df(_make_bundle(fold_trainer, None, None), te)
        for t in TARGET_NAMES:
            oof[t][val_idx] = pred[t].values

    pred_df = pd.DataFrame(oof, index=df.index)
    per_target = compute_regression_table(df[TARGET_NAMES], pred_df, list(TARGET_NAMES))
    abs_err = pred_df[TARGET_NAMES].subtract(df[TARGET_NAMES]).abs().mean(axis=1).values
    lo, hi = _bootstrap_ci(abs_err)
    return {
        "per_target_mae_mm": per_target.set_index("target")["mae_mm"].to_dict(),
        "axis_mae_mm": _axis_summary(per_target),
        "avg_mae_mm": float(per_target["mae_mm"].mean()),
        "avg_mae_ci95": [lo, hi],
        "z_avg_mae_mm": float(
            per_target.loc[per_target["target"].isin(Z_TARGETS), "mae_mm"].mean()
        ),
    }


def _raw_z_preds(trainer: AdaptiveEnsembleTrainer, frame: pd.DataFrame, target: str) -> np.ndarray:
    X = build_inference_matrix(trainer, frame, feature_names=trainer.feature_names)
    if trainer.z_head == "quantile_v7" and target in Z_TARGETS and trainer.z_driver_names:
        from src.models.z_quantile_v7 import predict_quantile_z

        X_imp = trainer.imputer.transform(X)
        return predict_quantile_z(
            trainer.trained_models[target],
            X_imp,
            trainer.feature_names,
            trainer.z_driver_names,
        )
    X_scaled = apply_model_preprocessing(
        X, {"imputer": trainer.imputer, "scaler": trainer.scaler}
    )
    return trainer.trained_models[target].predict(X_scaled)


def main() -> int:
    parser = argparse.ArgumentParser(description="Honest clinical displacement training")
    parser.add_argument(
        "--z-head",
        choices=("ensemble", "quantile_v7"),
        default="ensemble",
        help="Z-axis head: ensemble (production) or experimental V7 quantile drivers",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Output model path (default depends on --z-head)",
    )
    args = parser.parse_args()
    z_head = args.z_head
    model_path = args.model_path or (
        ROOT / "models" / "adaptive_ensemble_clinical_honest_v7.pkl"
        if z_head == "quantile_v7"
        else DEFAULT_MODEL_PATH
    )
    run_id = f"clinical_honest_{z_head}_{date.today().strftime('%Y%m%d')}"

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "build_vybor_from_xlsx.py")],
        cwd=str(ROOT),
        check=True,
    )
    df = normalize_dataframe(pd.read_csv(DEFAULT_OUTPUT_CSV))
    df = df.dropna(subset=list(TARGET_NAMES), how="any").reset_index(drop=True)
    print(f"[data] clinical patients={len(df)}")

    spine_eq_com = all(
        np.allclose(df[f"spine_center_{a}"], df[f"body_com_{a}"], atol=1e-3)
        for a in "xyz"
        if f"spine_center_{a}" in df.columns and f"body_com_{a}" in df.columns
    )
    print(f"[step3] spine==body_com (expect False): {spine_eq_com}")

    name_col = "full_name" if "full_name" in df.columns else "case_id"
    groups = df[name_col].astype(str).values

    trainer = AdaptiveEnsembleTrainer(z_head=z_head)
    dummy_val = df.iloc[:1].copy()
    X_train, X_val, y_train, y_val = trainer.prepare_training_data_split(df, dummy_val)
    print(f"[z] z_head={z_head}, drivers={len(trainer.z_driver_names)}")
    print(f"[step2] features={len(trainer.feature_names)} (leakage-free)")
    trainer.train_and_evaluate_adaptive_ensembles(
        X_train, X_val, y_train, y_val, groups=groups
    )

    left_cal = fit_calibrator_oof_gated(
        SideZCalibrator(side="left"),
        df,
        _raw_z_preds(trainer, df, Z_TARGETS[0]),
        df[Z_TARGETS[0]].astype(float).values,
        groups,
    )
    right_cal = fit_calibrator_oof_gated(
        SideZCalibrator(side="right"),
        df,
        _raw_z_preds(trainer, df, Z_TARGETS[1]),
        df[Z_TARGETS[1]].astype(float).values,
        groups,
    )
    print(f"[step6] calibrators left={bool(left_cal)} right={bool(right_cal)}")

    payload = {
        "models": trainer.trained_models,
        "scaler": trainer.scaler,
        "imputer": trainer.imputer,
        "feature_names": trainer.feature_names,
        "target_names": trainer.target_names,
        "left_z_calibrator": left_cal,
        "right_z_calibrator": right_cal,
        "z_head": z_head,
        "z_driver_names": trainer.z_driver_names,
        "training_meta": {
            "clinical_only": True,
            "kits_dicom_excluded_from_targets": True,
            "leakage_features_excluded": True,
            "weight_tuning": "GroupKFold",
            "final_fit": "100pct_clinical_train",
            "calibrators": "oof_gated_supine_only",
            "z_head": z_head,
        },
    }
    joblib.dump(payload, model_path)
    print(f"[OK] saved {model_path}")

    oof_metrics = evaluate_groupkfold_oof(df, z_head=z_head)
    report = {
        "run_id": run_id,
        "model_path": str(model_path),
        "n_clinical": len(df),
        "feature_count": len(trainer.feature_names),
        "spine_equals_body_com": spine_eq_com,
        "calibrators": {
            "left": left_cal.describe() if left_cal else None,
            "right": right_cal.describe() if right_cal else None,
        },
        "z_head": z_head,
        "z_driver_names": trainer.z_driver_names,
        "groupkfold_oof_87": oof_metrics,
    }
    run_dir = ROOT / "results" / "validation_runs" / run_id / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "clinical_honest_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
