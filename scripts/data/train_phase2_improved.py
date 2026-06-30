#!/usr/bin/env python3
"""Phase 2 training: projection features, KiTS impute-only, side Z, multitask, quantile."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "phase1"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

from adaptive_ensemble import AdaptiveEnsembleTrainer  # noqa: E402
import src.models.data_integration_fix as integration  # noqa: E402
from common import compute_regression_table, predict_df  # noqa: E402
from multitask_displacement import MultitaskDisplacementModel  # noqa: E402
from src.data.xlsx_displacement_parser import DEFAULT_OUTPUT_CSV  # noqa: E402
from src.features.pseudo_labeling import attach_pseudo_displacement_labels  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402
from src.models.data_integration_fix import DataIntegrationFix  # noqa: E402
from src.models.left_z_calibrator import LeftZCalibrator, TARGET as LEFT_Z_TARGET  # noqa: E402
from src.models.quantile_displacement import QuantileDisplacementPredictor  # noqa: E402
from src.models.side_z_predictor import RIGHT_Z, SideZModelPair  # noqa: E402

HARMONIZED_DIR = ROOT / "data" / "harmonized"
PROCESSED_DIR = ROOT / "data" / "processed_phase2"
VYBOR_CSV = DEFAULT_OUTPUT_CSV
MODEL_PATH = ROOT / "models" / "adaptive_ensemble_phase2.pkl"
RUN_ID = f"phase2_{date.today().strftime('%Y%m%d')}"
DICOM_PSEUDO_PATH = HARMONIZED_DIR / "dicom_medical_features_pseudolabeled_phase2.csv"
TEACHER_FALLBACKS = [
    ROOT / "models" / "adaptive_ensemble_full_extended.pkl",
    ROOT / "models" / "adaptive_ensemble_xlsx_axis_teacher.pkl",
    ROOT / "models" / "adaptive_ensemble_clinical_axis.pkl",
]
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


def load_teacher_bundle():
    for path in TEACHER_FALLBACKS:
        if path.exists():
            payload = joblib.load(path)
            print(f"[teacher] {path}")
            return payload
    raise FileNotFoundError("No teacher model found for DICOM pseudo-labeling")


def pseudo_label_dicom(vybor_path: Path) -> pd.DataFrame:
    payload = load_teacher_bundle()
    dicom = pd.read_csv(HARMONIZED_DIR / "dicom_medical_features_aligned.csv")
    bundle = type(
        "Bundle",
        (),
        {
            "mode": "pretrained_adaptive_ensemble",
            "feature_names": payload["feature_names"],
            "target_names": list(payload["models"].keys()),
            "scaler": payload["scaler"],
            "imputer": payload["imputer"],
            "models": payload["models"],
            "left_z_calibrator": payload.get("left_z_calibrator"),
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
        dicom_path=ROOT / "data" / "dicom_medical_features.csv",
        kits19_path=HARMONIZED_DIR / "kits19_medical_grade_features_aligned.csv",
        excel_path=None,
        training_mode="labeled_only",
    )
    return fixer.run()[1:3]


def _axis_summary(per_target: pd.DataFrame) -> dict[str, float]:
    axes: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    for _, row in per_target.iterrows():
        axis = row["target"].split("_")[-1]
        if axis in axes:
            axes[axis].append(float(row["mae_mm"]))
    return {axis: float(np.mean(vals)) for axis, vals in axes.items() if vals}


def main() -> int:
    vybor_path = rebuild_vybor()
    harmonize(vybor_path)
    train_df, val_df = integrate(vybor_path)

    print(
        f"[data] train={len(train_df)} val={len(val_df)} "
        f"sources={train_df['source'].value_counts().to_dict()}"
    )

    trainer = AdaptiveEnsembleTrainer()
    X_train, X_val, y_train, y_val = trainer.prepare_training_data_split(train_df, val_df)
    trainer.train_and_evaluate_adaptive_ensembles(X_train, X_val, y_train, y_val)

    sample_w = trainer.train_sample_weights
    side_z = SideZModelPair()
    side_z.fit(X_train, y_train, trainer.target_names, trainer.feature_names, sample_weight=sample_w)

    clinical_train = train_df[train_df["source"].isin({"Vybor", "Excel"})]
    calibrator = LeftZCalibrator()
    if len(clinical_train) > 0:
        from src.features.pipeline import apply_model_preprocessing, build_inference_matrix

        Xc = build_inference_matrix(trainer, clinical_train, feature_names=trainer.feature_names)
        Xc = apply_model_preprocessing(Xc, {"imputer": trainer.imputer, "scaler": trainer.scaler})
        raw_lz = side_z.left.predict(Xc)
        calibrator.fit(clinical_train, raw_lz, clinical_train[LEFT_Z_TARGET].astype(float).values)

    multitask = MultitaskDisplacementModel(n_components=24, alpha=3.0)
    multitask.fit(X_train, y_train, trainer.target_names, sample_weight=sample_w)

    quantile = QuantileDisplacementPredictor()
    quantile.fit(X_train, y_train, trainer.target_names, sample_weight=sample_w)
    q_cov = quantile.coverage_rate(X_val, y_val, trainer.target_names)

    payload = {
        "models": trainer.trained_models,
        "scaler": trainer.scaler,
        "imputer": trainer.imputer,
        "feature_names": trainer.feature_names,
        "target_names": trainer.target_names,
        "adaptive_weights": trainer.adaptive_weights,
        "best_models": trainer.best_models,
        "side_z_models": {"left": side_z.left, "right": side_z.right},
        "multitask_model": multitask,
        "quantile_model": quantile,
        "left_z_calibrator": calibrator if calibrator.fitted_ else None,
        "multitask_blend": {"z": 0.35, "xy": 0.15},
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"[OK] saved {MODEL_PATH}")

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
            "side_z_models": payload["side_z_models"],
            "left_z_calibrator": payload["left_z_calibrator"],
            "multitask_model": multitask,
            "multitask_blend": payload["multitask_blend"],
            "quantile_model": quantile,
        },
    )()
    val_pred = predict_df(bundle, val_df)
    per_target = compute_regression_table(val_df[TARGET_NAMES], val_pred, list(TARGET_NAMES))

    report = {
        "run_id": RUN_ID,
        "model_path": str(MODEL_PATH),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_sources": train_df["source"].value_counts().to_dict(),
        "feature_count": len(trainer.feature_names),
        "holdout": {
            "per_target_mae_mm": per_target.set_index("target")["mae_mm"].to_dict(),
            "axis_mae_mm": _axis_summary(per_target),
            "avg_mae_mm": float(per_target["mae_mm"].mean()),
        },
        "quantile_coverage_80_holdout": q_cov,
        "calibrator": calibrator.describe() if calibrator.fitted_ else None,
    }
    run_dir = ROOT / "results" / "validation_runs" / RUN_ID / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "phase2_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
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
