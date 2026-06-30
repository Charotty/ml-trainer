#!/usr/bin/env python3
"""Full extended training: 87 clinical + 210 KiTS19 + 159 DICOM pseudo (~438 train rows)."""

from __future__ import annotations

import json
import subprocess
import sys
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
import src.models.data_integration_fix as integration  # noqa: E402
from common import compute_regression_table, predict_df  # noqa: E402
from src.data.xlsx_displacement_parser import DEFAULT_OUTPUT_CSV  # noqa: E402
from src.features.pseudo_labeling import attach_pseudo_displacement_labels  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402
from src.features.pipeline import apply_model_preprocessing, build_inference_matrix  # noqa: E402
from src.models.data_integration_fix import DataIntegrationFix  # noqa: E402
from src.models.left_z_calibrator import TARGET as LEFT_Z_TARGET, LeftZCalibrator  # noqa: E402

HARMONIZED_DIR = ROOT / "data" / "harmonized"
PROCESSED_DIR = ROOT / "data" / "processed_full_extended"
VYBOR_CSV = DEFAULT_OUTPUT_CSV
MODEL_PATH = ROOT / "models" / "adaptive_ensemble_full_extended.pkl"
TEACHER_PATH = ROOT / "models" / "adaptive_ensemble_full_extended_teacher.pkl"
RUN_ID = f"full_extended_{date.today().strftime('%Y%m%d')}"
DICOM_PSEUDO_PATH = HARMONIZED_DIR / "dicom_medical_features_pseudolabeled_full.csv"
SEED = 42


def rebuild_vybor() -> Path:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "build_vybor_from_xlsx.py")],
        cwd=str(ROOT),
        check=True,
    )
    return VYBOR_CSV


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


TEACHER_FALLBACKS = [
    TEACHER_PATH,
    ROOT / "models" / "adaptive_ensemble_xlsx_axis_teacher.pkl",
    ROOT / "models" / "adaptive_ensemble_clinical_axis.pkl",
]


def _load_trainer_from_pkl(path: Path) -> AdaptiveEnsembleTrainer:
    payload = joblib.load(path)
    trainer = AdaptiveEnsembleTrainer()
    trainer.feature_names = payload["feature_names"]
    trainer.target_names = payload.get("target_names", list(payload["models"].keys()))
    trainer.trained_models = payload["models"]
    trainer.imputer = payload["imputer"]
    trainer.scaler = payload["scaler"]
    return trainer


def train_teacher(vybor_path: Path) -> AdaptiveEnsembleTrainer:
    for candidate in TEACHER_FALLBACKS:
        if candidate.exists():
            print(f"[teacher] loaded {candidate}")
            return _load_trainer_from_pkl(candidate)

    vybor = normalize_dataframe(pd.read_csv(vybor_path))
    train_df, val_df = train_test_split(vybor, test_size=0.2, random_state=SEED)
    trainer = AdaptiveEnsembleTrainer()
    X_train, X_val, y_train, y_val = trainer.prepare_training_data_split(train_df, val_df)
    trainer.train_and_evaluate_adaptive_ensembles(X_train, X_val, y_train, y_val)
    trainer.save_model(str(TEACHER_PATH))
    return trainer


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
        },
    )()
    labeled = attach_pseudo_displacement_labels(
        dicom,
        bundle,
        predict_fn=predict_df,
        reference_df=pd.read_csv(vybor_path),
    )
    labeled.to_csv(DICOM_PSEUDO_PATH, index=False)
    return labeled


def integrate(vybor_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    integration.PROCESSED_DIR = PROCESSED_DIR
    fixer = DataIntegrationFix(
        vybor_path=vybor_path,
        dicom_path=DICOM_PSEUDO_PATH,
        kits19_path=HARMONIZED_DIR / "kits19_medical_grade_features_aligned.csv",
        excel_path=None,
        training_mode="clinical_xlsx_extended",
    )
    return fixer.run()[1:3]


def _raw_left_z(trainer: AdaptiveEnsembleTrainer, df: pd.DataFrame) -> np.ndarray:
    df_norm = normalize_dataframe(df)
    X = build_inference_matrix(trainer, df_norm, feature_names=trainer.feature_names)
    X_scaled = apply_model_preprocessing(
        X,
        {"imputer": trainer.imputer, "scaler": trainer.scaler},
    )
    return trainer.trained_models[LEFT_Z_TARGET].predict(X_scaled)


def fit_left_z_calibrator(
    trainer: AdaptiveEnsembleTrainer,
    train_df: pd.DataFrame,
) -> LeftZCalibrator:
    clinical = train_df[train_df["source"].isin({"Vybor", "Excel"})].copy()
    if len(clinical) == 0:
        clinical = train_df.copy()
    raw = _raw_left_z(trainer, clinical)
    calibrator = LeftZCalibrator()
    calibrator.fit(clinical, raw, clinical[LEFT_Z_TARGET].astype(float).values)
    return calibrator


def _axis_summary(per_target: pd.DataFrame) -> dict[str, float]:
    axes: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    for _, row in per_target.iterrows():
        axis = row["target"].split("_")[-1]
        if axis in axes:
            axes[axis].append(float(row["mae_mm"]))
    return {axis: float(np.mean(vals)) for axis, vals in axes.items() if vals}


def _eval_bundle(bundle, df: pd.DataFrame) -> dict:
    pred = predict_df(bundle, df)
    per_target = compute_regression_table(df[TARGET_NAMES], pred, list(TARGET_NAMES))
    axis = _axis_summary(per_target)
    return {
        "n": len(df),
        "per_target_mae_mm": per_target.set_index("target")["mae_mm"].to_dict(),
        "axis_mae_mm": axis,
        "avg_mae_mm": float(per_target["mae_mm"].mean()),
    }


def save_model_with_calibrator(
    trainer: AdaptiveEnsembleTrainer,
    calibrator: LeftZCalibrator,
    path: Path,
) -> None:
    payload = {
        "models": trainer.trained_models,
        "scaler": trainer.scaler,
        "imputer": trainer.imputer,
        "feature_names": trainer.feature_names,
        "target_names": trainer.target_names,
        "train_data": trainer.X_train,
        "required_features": trainer.required_features,
        "target_columns": trainer.target_columns,
        "adaptive_weights": trainer.adaptive_weights,
        "best_models": trainer.best_models,
        "left_z_calibrator": calibrator,
    }
    joblib.dump(payload, path)


def make_bundle(trainer: AdaptiveEnsembleTrainer, calibrator: LeftZCalibrator):
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
            "left_z_calibrator": calibrator,
        },
    )()


def main() -> int:
    vybor_path = rebuild_vybor()
    harmonize(vybor_path)
    teacher = train_teacher(vybor_path)
    pseudo_label_dicom(teacher, vybor_path)
    train_df, val_df = integrate(vybor_path)

    src_counts = train_df["source"].value_counts().to_dict() if "source" in train_df.columns else {}
    print(
        f"[data] train={len(train_df)} val={len(val_df)} "
        f"sources={src_counts} total_labeled={len(train_df)+len(val_df)}"
    )

    trainer = AdaptiveEnsembleTrainer()
    X_train, X_val, y_train, y_val = trainer.prepare_training_data_split(train_df, val_df)
    print(f"[train] features={len(trainer.feature_names)} sample_weight={'yes' if trainer.train_sample_weights is not None else 'no'}")
    trainer.train_and_evaluate_adaptive_ensembles(X_train, X_val, y_train, y_val)

    calibrator = fit_left_z_calibrator(trainer, train_df)
    save_model_with_calibrator(trainer, calibrator, MODEL_PATH)
    print(f"[OK] model -> {MODEL_PATH}")
    print(f"[left Z calibrator] {json.dumps(calibrator.describe(), ensure_ascii=False)}")

    bundle = make_bundle(trainer, calibrator)
    holdout = _eval_bundle(bundle, val_df)
    vybor_all = normalize_dataframe(pd.read_csv(vybor_path))
    full87 = _eval_bundle(bundle, vybor_all)

    clinical_train = train_df[train_df["source"].isin({"Vybor", "Excel"})]
    raw_lz = _raw_left_z(trainer, val_df)
    cal_lz = calibrator.transform(val_df, raw_lz)
    lz_before = float(mean_absolute_error(val_df[LEFT_Z_TARGET], raw_lz))
    lz_after = float(mean_absolute_error(val_df[LEFT_Z_TARGET], cal_lz))

    report = {
        "run_id": RUN_ID,
        "model_path": str(MODEL_PATH),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_sources": src_counts,
        "total_with_labels": len(train_df) + len(val_df),
        "left_z_holdout": {
            "before_mae_mm": lz_before,
            "after_mae_mm": lz_after,
            "delta_mm": lz_after - lz_before,
        },
        "holdout_18": holdout,
        "vybor_all_87": full87,
        "calibrator": calibrator.describe(),
    }

    run_dir = ROOT / "results" / "validation_runs" / RUN_ID / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "full_extended_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validation" / "evaluate_metrics.py"),
            "--dataset",
            str(vybor_path),
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
