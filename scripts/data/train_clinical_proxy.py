#!/usr/bin/env python3
"""Proxy-weighted training: Vybor clinical + KiTS19 proxy δ + DICOM pseudo-δ.

Honest evaluation is always GroupKFold OOF on clinical Vybor only (87 patients).
Proxy rows are down-weighted via sample_weight (clinical=1.0, KiTS=0.08, DICOM=0.06).
"""

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
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "phase1"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

from adaptive_ensemble import AdaptiveEnsembleTrainer  # noqa: E402
import src.models.data_integration_fix as integration  # noqa: E402
from common import compute_regression_table, predict_df  # noqa: E402
from src.data.xlsx_displacement_parser import DEFAULT_OUTPUT_CSV  # noqa: E402
from src.features.pseudo_labeling import attach_pseudo_displacement_labels  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402
from src.features.pipeline import apply_model_preprocessing, build_inference_matrix  # noqa: E402
from src.models.data_integration_fix import DataIntegrationFix, LABELED_SOURCES  # noqa: E402
from src.models.z_calibrator_oof import SideZCalibrator, fit_calibrator_oof_gated  # noqa: E402
from train_clinical_honest import (  # noqa: E402
    N_BOOTSTRAP,
    N_SPLITS,
    SEED,
    Z_TARGETS,
    _axis_summary,
    _bootstrap_ci,
    evaluate_groupkfold_oof,
)

HARMONIZED_DIR = ROOT / "data" / "harmonized"
PROCESSED_DIR = ROOT / "data" / "processed_proxy"
DICOM_PSEUDO_PATH = HARMONIZED_DIR / "dicom_medical_features_pseudolabeled_proxy.csv"
DEFAULT_MODEL_PATH = ROOT / "models" / "adaptive_ensemble_clinical_proxy.pkl"
TEACHER_FALLBACKS = [
    ROOT / "models" / "adaptive_ensemble_clinical_honest.pkl",
    ROOT / "models" / "adaptive_ensemble_xlsx_axis_teacher.pkl",
]


def rebuild_vybor() -> Path:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "build_vybor_from_xlsx.py")],
        cwd=str(ROOT),
        check=True,
    )
    return DEFAULT_OUTPUT_CSV


def harmonize(vybor_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "data" / "harmonize_extracted_datasets.py"),
            "--reference",
            str(vybor_path),
        ],
        cwd=str(ROOT),
        check=True,
    )


def _load_teacher(path: Path) -> AdaptiveEnsembleTrainer:
    payload = joblib.load(path)
    trainer = AdaptiveEnsembleTrainer(z_head=payload.get("z_head", "ensemble"))
    trainer.feature_names = payload["feature_names"]
    trainer.target_names = payload.get("target_names", list(payload["models"].keys()))
    trainer.trained_models = payload["models"]
    trainer.imputer = payload["imputer"]
    trainer.scaler = payload["scaler"]
    trainer.z_driver_names = payload.get("z_driver_names")
    return trainer


def load_teacher(vybor_path: Path | None = None) -> tuple[AdaptiveEnsembleTrainer, Path]:
    for candidate in TEACHER_FALLBACKS:
        if candidate.exists():
            print(f"[teacher] loaded {candidate}")
            return _load_teacher(candidate), candidate
    if vybor_path is None:
        raise FileNotFoundError(
            f"No teacher model found. Run train_clinical_honest.py first. Tried: {TEACHER_FALLBACKS}"
        )
    print("[teacher] no checkpoint found — training quick Vybor teacher")
    vybor = normalize_dataframe(pd.read_csv(vybor_path))
    train_df, val_df = train_test_split(vybor, test_size=0.2, random_state=SEED)
    trainer = AdaptiveEnsembleTrainer()
    X_train, X_val, y_train, y_val = trainer.prepare_training_data_split(train_df, val_df)
    trainer.train_and_evaluate_adaptive_ensembles(X_train, X_val, y_train, y_val)
    fallback = ROOT / "models" / "adaptive_ensemble_clinical_proxy_teacher.pkl"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": trainer.trained_models,
            "scaler": trainer.scaler,
            "imputer": trainer.imputer,
            "feature_names": trainer.feature_names,
            "target_names": trainer.target_names,
            "z_head": "ensemble",
            "z_driver_names": trainer.z_driver_names,
        },
        fallback,
    )
    print(f"[teacher] saved {fallback}")
    return trainer, fallback


def pseudo_label_dicom(teacher: AdaptiveEnsembleTrainer, vybor_path: Path) -> pd.DataFrame:
    dicom = pd.read_csv(HARMONIZED_DIR / "dicom_medical_features_aligned.csv")
    bundle = type(
        "Bundle",
        (),
        {
            "mode": "pretrained_adaptive_ensemble",
            "feature_names": teacher.feature_names,
            "target_names": teacher.target_names,
            "scaler": teacher.scaler,
            "imputer": teacher.imputer,
            "models": teacher.trained_models,
            "left_z_calibrator": None,
            "right_z_calibrator": None,
            "z_head": teacher.z_head,
            "z_driver_names": teacher.z_driver_names,
        },
    )()
    labeled = attach_pseudo_displacement_labels(
        dicom,
        bundle,
        predict_fn=predict_df,
        reference_df=pd.read_csv(vybor_path),
    )
    labeled.to_csv(DICOM_PSEUDO_PATH, index=False)
    complete = labeled[TARGET_NAMES].notna().all(axis=1).sum()
    print(f"[pseudo] DICOM {complete}/{len(dicom)} rows -> {DICOM_PSEUDO_PATH}")
    return labeled


def integrate_proxy(vybor_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    integration.PROCESSED_DIR = PROCESSED_DIR
    kits_path = HARMONIZED_DIR / "kits19_medical_grade_features_aligned.csv"
    if not kits_path.exists():
        kits_path = ROOT / "data" / "kits19_medical_grade_features.csv"
    fixer = DataIntegrationFix(
        vybor_path=vybor_path,
        dicom_path=DICOM_PSEUDO_PATH,
        kits19_path=kits_path,
        excel_path=None,
        training_mode="proxy_weighted_extended",
    )
    _master, train_df, val_df, _missing = fixer.run()
    manifest_path = PROCESSED_DIR / "integration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return train_df, val_df, manifest


def _row_group(row: pd.Series) -> str:
    if row.get("source") in LABELED_SOURCES:
        return str(row.get("full_name") or row.get("case_id"))
    return str(row.get("case_id") or row.get("universal_id") or row.name)


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
    parser = argparse.ArgumentParser(description="Proxy-weighted clinical + KiTS/DICOM training")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--skip-harmonize", action="store_true")
    parser.add_argument("--skip-vybor-build", action="store_true")
    args = parser.parse_args()
    run_id = f"clinical_proxy_{date.today().strftime('%Y%m%d')}"

    if args.skip_vybor_build and DEFAULT_OUTPUT_CSV.exists():
        vybor_path = DEFAULT_OUTPUT_CSV
        print(f"[data] using existing {vybor_path}")
    else:
        vybor_path = rebuild_vybor()
    if not args.skip_harmonize:
        harmonize(vybor_path)

    teacher, teacher_path = load_teacher(vybor_path)
    pseudo_label_dicom(teacher, vybor_path)
    train_df, val_df, manifest = integrate_proxy(vybor_path)

    src_counts = train_df["source"].value_counts().to_dict() if "source" in train_df.columns else {}
    w_min = w_max = None
    if "sample_weight" in train_df.columns:
        w = train_df["sample_weight"]
        w_min, w_max = float(w.min()), float(w.max())
    print(
        f"[data] train={len(train_df)} val={len(val_df)} sources={src_counts} "
        f"weights=[{w_min}, {w_max}]"
    )

    groups = train_df.apply(_row_group, axis=1).values
    trainer = AdaptiveEnsembleTrainer()
    X_train, X_val, y_train, y_val = trainer.prepare_training_data_split(train_df, val_df)
    print(f"[train] features={len(trainer.feature_names)} sample_weight={'yes' if trainer.train_sample_weights is not None else 'no'}")
    trainer.train_and_evaluate_adaptive_ensembles(
        X_train, X_val, y_train, y_val, groups=groups
    )

    clinical_all = normalize_dataframe(pd.read_csv(vybor_path))
    clinical_all = clinical_all.dropna(subset=list(TARGET_NAMES), how="any").reset_index(drop=True)
    name_col = "full_name" if "full_name" in clinical_all.columns else "case_id"
    clin_groups = clinical_all[name_col].astype(str).values

    left_cal = fit_calibrator_oof_gated(
        SideZCalibrator(side="left"),
        clinical_all,
        _raw_z_preds(trainer, clinical_all, Z_TARGETS[0]),
        clinical_all[Z_TARGETS[0]].astype(float).values,
        clin_groups,
    )
    right_cal = fit_calibrator_oof_gated(
        SideZCalibrator(side="right"),
        clinical_all,
        _raw_z_preds(trainer, clinical_all, Z_TARGETS[1]),
        clinical_all[Z_TARGETS[1]].astype(float).values,
        clin_groups,
    )

    payload = {
        "models": trainer.trained_models,
        "scaler": trainer.scaler,
        "imputer": trainer.imputer,
        "feature_names": trainer.feature_names,
        "target_names": trainer.target_names,
        "left_z_calibrator": left_cal,
        "right_z_calibrator": right_cal,
        "z_head": "ensemble",
        "z_driver_names": trainer.z_driver_names,
        "training_meta": {
            "mode": "proxy_weighted_extended",
            "teacher": str(teacher_path),
            "honest_eval": "groupkfold_oof_clinical_only",
            "sample_weights": {"clinical": 1.0, "proxy_kits": 0.08, "pseudo_dicom": 0.06},
        },
    }
    joblib.dump(payload, args.model_path)
    print(f"[OK] saved {args.model_path}")

    oof_metrics = evaluate_groupkfold_oof(clinical_all)
    holdout_bundle = type(
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
            "z_head": "ensemble",
            "z_driver_names": trainer.z_driver_names,
        },
    )()
    holdout_pred = predict_df(holdout_bundle, val_df)
    holdout_per_target = compute_regression_table(
        val_df[TARGET_NAMES], holdout_pred, list(TARGET_NAMES)
    )
    holdout_mae = float(holdout_per_target["mae_mm"].mean())

    report = {
        "run_id": run_id,
        "model_path": str(args.model_path),
        "teacher_path": str(teacher_path),
        "integration_manifest": manifest,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_sources": src_counts,
        "n_clinical_eval": len(clinical_all),
        "groupkfold_oof_clinical_87": oof_metrics,
        "clinical_holdout_18_mae_mm": holdout_mae,
        "calibrators": {
            "left": left_cal.describe() if left_cal else None,
            "right": right_cal.describe() if right_cal else None,
        },
    }
    run_dir = ROOT / "results" / "validation_runs" / run_id / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "clinical_proxy_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
