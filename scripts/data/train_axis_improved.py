#!/usr/bin/env python3
"""Rebuild xlsx Vybor with spine/span features and retrain with Z/Y-focused ensemble."""

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
import src.models.data_integration_fix as integration  # noqa: E402
from src.data.xlsx_displacement_parser import DEFAULT_OUTPUT_CSV  # noqa: E402
from src.features.pseudo_labeling import attach_pseudo_displacement_labels  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402
from src.models.data_integration_fix import DataIntegrationFix  # noqa: E402
from common import predict_df  # noqa: E402

HARMONIZED_DIR = ROOT / "data" / "harmonized"
PROCESSED_DIR = ROOT / "data" / "processed_xlsx_axis"
VYBOR_CSV = DEFAULT_OUTPUT_CSV
MODEL_PATH = ROOT / "models" / "adaptive_ensemble_xlsx_axis.pkl"
TEACHER_PATH = ROOT / "models" / "adaptive_ensemble_xlsx_axis_teacher.pkl"
RUN_ID = f"clinical_xlsx_axis_{date.today().strftime('%Y%m%d')}"
DICOM_PSEUDO_PATH = HARMONIZED_DIR / "dicom_medical_features_pseudolabeled_axis.csv"


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


def train_teacher(vybor_path: Path) -> AdaptiveEnsembleTrainer:
    vybor = normalize_dataframe(pd.read_csv(vybor_path))
    train_df, val_df = train_test_split(vybor, test_size=0.2, random_state=42)
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


def axis_metrics(trainer: AdaptiveEnsembleTrainer, val_df: pd.DataFrame) -> dict:
    from src.features.pipeline import apply_model_preprocessing, build_inference_matrix

    val_norm = normalize_dataframe(val_df)
    X = build_inference_matrix(trainer, val_norm, feature_names=trainer.feature_names)
    model_data = {"imputer": trainer.imputer, "scaler": trainer.scaler}
    X_scaled = apply_model_preprocessing(X, model_data)
    out = {}
    for i, tgt in enumerate(trainer.target_names):
        pred = trainer.trained_models[tgt].predict(X_scaled)
        y = val_norm[tgt].astype(float).values
        err = abs(y - pred)
        out[tgt] = {"mae_mm": float(err.mean()), "within_5mm": float((err <= 5).mean())}
    yz = [v for k, v in out.items() if k.endswith("_y") or k.endswith("_z")]
    out["yz_avg_mae"] = float(sum(v["mae_mm"] for v in yz) / len(yz)) if yz else None
    return out


def main() -> int:
    vybor_path = rebuild_vybor()
    harmonize(vybor_path)
    teacher = train_teacher(vybor_path)
    pseudo_label_dicom(teacher, vybor_path)
    train_df, val_df = integrate(vybor_path)

    trainer = AdaptiveEnsembleTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_training_data_split(train_df, val_df)
    print(f"[train] features={len(trainer.feature_names)} (incl. axis extras)")
    trainer.train_and_evaluate_adaptive_ensembles(X_train, X_test, y_train, y_test)
    trainer.save_model(str(MODEL_PATH))

    metrics = axis_metrics(trainer, val_df)
    out = ROOT / "results" / "validation_runs" / RUN_ID / "metrics" / "axis_improvement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

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
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validation" / "plot_xlsx_clinical_analysis.py"),
            "--run-id",
            RUN_ID,
            "--baseline-run-id",
            "clinical_xlsx_extended_20260629",
            "--model",
            str(MODEL_PATH),
        ],
        cwd=str(ROOT),
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
