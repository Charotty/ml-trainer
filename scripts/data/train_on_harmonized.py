#!/usr/bin/env python3
"""Build Vybor from xlsx, harmonize CT features, train on clinical labels only."""

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
from src.features.phase1_schema import normalize_dataframe  # noqa: E402
from src.models.data_integration_fix import DataIntegrationFix  # noqa: E402
from common import predict_df  # noqa: E402

HARMONIZED_DIR = ROOT / "data" / "harmonized"
PROCESSED_DIR = ROOT / "data" / "processed_xlsx"
VYBOR_CSV = DEFAULT_OUTPUT_CSV
MODEL_PATH = ROOT / "models" / "adaptive_ensemble_xlsx_axis.pkl"
TEACHER_PATH = ROOT / "models" / "adaptive_ensemble_xlsx_axis_teacher.pkl"
RUN_ID = f"clinical_xlsx_axis_{date.today().strftime('%Y%m%d')}"
DICOM_PSEUDO_PATH = HARMONIZED_DIR / "dicom_medical_features_pseudolabeled_axis.csv"


def build_vybor_csv() -> Path:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "data" / "build_vybor_from_xlsx.py")],
        cwd=str(ROOT),
        check=True,
    )
    if not VYBOR_CSV.exists():
        raise FileNotFoundError(f"Expected Vybor CSV at {VYBOR_CSV}")
    return VYBOR_CSV


def run_harmonization(vybor_path: Path) -> None:
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


def train_vybor_teacher(vybor_path: Path) -> AdaptiveEnsembleTrainer:
    """Teacher on full clinical xlsx cohort (patient-level split)."""
    vybor = normalize_dataframe(pd.read_csv(vybor_path))
    train_df, val_df = train_test_split(vybor, test_size=0.2, random_state=42)
    trainer = AdaptiveEnsembleTrainer()
    X_train, X_val, y_train, y_val = trainer.prepare_training_data_split(train_df, val_df)
    trainer.train_and_evaluate_adaptive_ensembles(X_train, X_val, y_train, y_val)
    trainer.save_model(str(TEACHER_PATH))
    print(f"[teacher] clinical rows train={len(train_df)} val={len(val_df)} -> {TEACHER_PATH}")
    return trainer


def build_pseudo_labeled_dicom(teacher: AdaptiveEnsembleTrainer, vybor_path: Path) -> pd.DataFrame:
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
    complete = labeled[teacher.target_names].notna().all(axis=1)
    labeled.to_csv(DICOM_PSEUDO_PATH, index=False)
    print(
        f"[pseudo] DICOM labeled {int(complete.sum())}/{len(dicom)} rows -> {DICOM_PSEUDO_PATH}"
    )
    return labeled


def integrate_clinical(vybor_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    integration.PROCESSED_DIR = PROCESSED_DIR
    fixer = DataIntegrationFix(
        vybor_path=vybor_path,
        dicom_path=ROOT / "data" / "dicom_medical_features.csv",
        kits19_path=HARMONIZED_DIR / "kits19_medical_grade_features_aligned.csv",
        excel_path=None,
        training_mode="labeled_only",
    )
    master_df, train_df, val_df, _ = fixer.run()
    master_df.to_csv(ROOT / "data" / "integrated_master_dataset_xlsx.csv", index=False)
    return train_df, val_df


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> AdaptiveEnsembleTrainer:
    trainer = AdaptiveEnsembleTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_training_data_split(train_df, val_df)
    trainer.train_and_evaluate_adaptive_ensembles(X_train, X_test, y_train, y_test)
    trainer.generate_report()
    trainer.save_results(
        str(ROOT / "results" / "adaptive_ensemble_xlsx_extended_results.csv")
    )
    trainer.save_model(str(MODEL_PATH))
    return trainer


def run_validation(vybor_path: Path) -> Path:
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
            "--source",
            "Vybor",
        ],
        cwd=str(ROOT),
        check=True,
    )
    return ROOT / "results" / "validation_runs" / RUN_ID


def summarize_train_sources(train_df: pd.DataFrame) -> dict:
    summary = {
        "run_id": RUN_ID,
        "vybor_source": str(VYBOR_CSV),
        "total_train_rows": int(len(train_df)),
        "by_source": train_df["source"].value_counts().to_dict() if "source" in train_df.columns else {},
        "by_label_quality": (
            train_df["label_quality"].value_counts().to_dict()
            if "label_quality" in train_df.columns
            else {}
        ),
        "sample_weight_stats": (
            train_df["sample_weight"].describe().to_dict()
            if "sample_weight" in train_df.columns
            else {}
        ),
    }
    out = ROOT / "results" / "validation_runs" / RUN_ID / "metrics" / "train_source_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    print("=== 1. Build Vybor CSV from main xlsx (87 clinical rows) ===")
    vybor_path = build_vybor_csv()

    print("\n=== 2. Harmonize DICOM/KiTS19 to xlsx Vybor frame (features only) ===")
    run_harmonization(vybor_path)

    print("\n=== 3. Integrate clinical-only train/val (no KiTS/DICOM regression targets) ===")
    train_df, val_df = integrate_clinical(vybor_path)
    print(f"Train: {len(train_df)}, Val (clinical only): {len(val_df)}")
    train_summary = summarize_train_sources(train_df)
    print(json.dumps(train_summary, indent=2, ensure_ascii=False))

    print("\n=== 4. Train ensemble on clinical labels only ===")
    train_model(train_df, val_df)

    print("\n=== 5. Evaluate on clinical holdout ===")
    run_dir = run_validation(vybor_path)
    summary_path = run_dir / "metrics" / "metrics_summary.csv"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
