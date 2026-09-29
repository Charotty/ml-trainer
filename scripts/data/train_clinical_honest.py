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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ensemble import AdaptiveEnsembleTrainer  # noqa: E402
from src.data.xlsx_displacement_parser import DEFAULT_OUTPUT_CSV  # noqa: E402
from src.features.na_trend_features import NaTrendStore  # noqa: E402
from src.features.phase1_schema import (  # noqa: E402
    TARGET_NAMES,
    filter_any_kidney_targets,
    normalize_dataframe,
)
from src.features.pipeline import apply_model_preprocessing, build_inference_matrix  # noqa: E402
from src.models.artifact_manifest import apply_oof_to_manifest, oof_placeholder  # noqa: E402
from src.models.ensemble.serializer import build_runtime_payload  # noqa: E402
from src.models.nested_cv import evaluate_nested_groupkfold_oof  # noqa: E402
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


def _make_bundle(trainer, left_cal, right_cal, na_trend_store=None):
    store = na_trend_store if na_trend_store is not None else getattr(trainer, "na_trend_store", None)
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
            "enrichment_mode": getattr(trainer, "enrichment_mode", "projection"),
            "na_trend_store": store.to_dict() if store is not None else None,
        },
    )()


def evaluate_groupkfold_oof(
    df: pd.DataFrame,
    *,
    z_head: str = "ensemble",
    na_trend_store: NaTrendStore | None = None,
    n_splits: int = N_SPLITS,
    estimator_profile: str = "production",
) -> dict:
    """Honest nested GroupKFold OOF (inner weight tuning). Not a proxy metric."""

    def trainer_factory(**kwargs):
        params = {
            "z_head": z_head,
            "enrichment_mode": "na_trends" if na_trend_store is not None else "none",
            "na_trend_store": na_trend_store,
            "estimator_profile": estimator_profile,
        }
        params.update(kwargs)
        return AdaptiveEnsembleTrainer(**params)

    result = evaluate_nested_groupkfold_oof(
        df,
        trainer_factory=trainer_factory,
        n_splits=n_splits,
        na_trend_store=na_trend_store,
        weight_mode="inner_groupkfold",
    )
    return result.to_report_dict()


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
    parser.add_argument(
        "--with-kits",
        action="store_true",
        help="Include KiTS19 cohort in na_trend features (experimental)",
    )
    parser.add_argument(
        "--skip-vybor-build",
        action="store_true",
        help="Skip build_vybor_from_xlsx.py (use existing or pre-built vybor CSV)",
    )
    parser.add_argument(
        "--vybor-csv",
        type=Path,
        default=None,
        help="Clinical training CSV (default: data/vybor_from_xlsx.csv)",
    )
    parser.add_argument(
        "--spine-csv",
        type=Path,
        default=None,
        help="na_spine cohort CSV for na_trend features",
    )
    parser.add_argument(
        "--boku-csv",
        type=Path,
        default=None,
        help="na_boku cohort CSV for na_trend features",
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=ROOT / "config" / "holdout_patients.yaml",
        help="Holdout YAML; passed to build_vybor_from_xlsx.py",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help="Displacement workbook for rebuild (default: parser DEFAULT_XLSX_PATH)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON with AdaptiveEnsembleTrainer kwargs (Variant A)",
    )
    args = parser.parse_args()
    z_head = args.z_head
    model_path = args.model_path or (
        ROOT / "models" / "adaptive_ensemble_clinical_honest_v7.pkl"
        if z_head == "quantile_v7"
        else DEFAULT_MODEL_PATH
    )
    run_id = f"clinical_honest_{z_head}_{date.today().strftime('%Y%m%d')}"

    vybor_path = args.vybor_csv or DEFAULT_OUTPUT_CSV
    if not args.skip_vybor_build:
        build_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "data" / "build_vybor_from_xlsx.py"),
            "--no-boku",
            "--holdout",
            str(args.holdout),
        ]
        if args.xlsx is not None:
            build_cmd.extend(["--xlsx", str(args.xlsx)])
        if args.vybor_csv is not None:
            build_cmd.extend(["--out", str(args.vybor_csv)])
        subprocess.run(build_cmd, cwd=str(ROOT), check=True)
    elif not vybor_path.exists():
        raise FileNotFoundError(
            f"--skip-vybor-build set but {vybor_path} missing. "
            "Run scripts/data/build_vybor_from_csv.py first."
        )
    else:
        print(f"[data] using existing {vybor_path}")
    df = normalize_dataframe(pd.read_csv(vybor_path))
    before = len(df)
    df = filter_any_kidney_targets(df).reset_index(drop=True)
    both = int(df[list(TARGET_NAMES)].notna().all(axis=1).sum())
    print(
        f"[data] clinical patients={len(df)} "
        f"(both kidneys={both}, one-sided={len(df) - both}; "
        f"dropped {before - len(df)} with no labeled side)"
    )

    from src.data.holdout import assert_no_holdout_leak  # noqa: E402

    assert_no_holdout_leak(df)

    na_trends = NaTrendStore.fit(
        spine_path=args.spine_csv,
        boku_path=args.boku_csv,
        include_kits=args.with_kits,
    )
    print(f"[na_trends] {json.dumps(na_trends.describe(), ensure_ascii=False)}")

    spine_eq_com = all(
        np.allclose(df[f"spine_center_{a}"], df[f"body_com_{a}"], atol=1e-3)
        for a in "xyz"
        if f"spine_center_{a}" in df.columns and f"body_com_{a}" in df.columns
    )
    print(f"[step3] spine==body_com (expect False): {spine_eq_com}")

    name_col = "full_name" if "full_name" in df.columns else "case_id"
    groups = df[name_col].astype(str).values

    trainer_kwargs = {
        "z_head": z_head,
        "enrichment_mode": "na_trends",
        "na_trend_store": na_trends,
    }
    if args.config is not None:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("--config must be a JSON object")
        trainer_kwargs.update({k: v for k, v in cfg.items() if k != "na_trend_store"})
    trainer = AdaptiveEnsembleTrainer(**trainer_kwargs)
    prepared = trainer.prepare_training_data_fit(df)
    if prepared[0] is None:
        raise RuntimeError("prepare_training_data_fit failed")
    X_train, y_train = prepared
    print(f"[z] z_head={z_head}, drivers={len(trainer.z_driver_names)}")
    print(f"[step2] features={len(trainer.feature_names)} (leakage-free)")
    trainer.fit_final(X_train, y_train, groups=groups, weight_mode="inner_groupkfold")

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

    payload = build_runtime_payload(
        trainer,
        extras={
            "left_z_calibrator": left_cal,
            "right_z_calibrator": right_cal,
            "na_trend_store": na_trends.to_dict(),
            "training_meta": {
                "clinical_only": True,
                "kits_dicom_excluded_from_targets": True,
                "na_spine_na_boku": "cohort_trends_only",
                "kits_in_trends": na_trends.include_kits,
                "boku_volume_fill": False,
                "projection_join_by_name": False,
                "leakage_features_excluded": True,
                "weight_tuning": "nested_inner_GroupKFold",
                "final_fit": "100pct_clinical_train_fit_final",
                "oof_protocol": "nested_groupkfold",
                "calibrators": "oof_gated_supine_only",
                "z_head": z_head,
            },
        },
    )
    if payload.get("categorical_encoder") is None:
        raise RuntimeError(
            "categorical_encoder was not fitted; refusing to save a silent-majority artifact"
        )
    joblib.dump(payload, model_path)
    print(f"[OK] saved {model_path}")

    oof_report = evaluate_groupkfold_oof(df, z_head=z_head, na_trend_store=na_trends)
    oof_metrics = {
        k: oof_report[k]
        for k in (
            "per_target_mae_mm",
            "axis_mae_mm",
            "avg_mae_mm",
            "avg_mae_ci95",
            "z_avg_mae_mm",
        )
        if k in oof_report
    }
    report = {
        "run_id": run_id,
        "model_path": str(model_path),
        "n_clinical": len(df),
        "feature_count": len(trainer.feature_names),
        "na_trend_features": len(trainer.na_trend_feature_cols),
        "na_trends": na_trends.describe(),
        "spine_equals_body_com": spine_eq_com,
        "calibrators": {
            "left": left_cal.describe() if left_cal else None,
            "right": right_cal.describe() if right_cal else None,
        },
        "z_head": z_head,
        "z_driver_names": trainer.z_driver_names,
        "groupkfold_oof_87": oof_metrics,
        "nested_groupkfold_oof": oof_report,
    }
    manifest_oof = apply_oof_to_manifest(oof_placeholder(protocol="nested_groupkfold"), oof_report)
    report["artifact_oof"] = {
        "folds": manifest_oof.get("folds"),
        "oof_predictions": manifest_oof.get("oof_predictions"),
        "aggregated_metrics": manifest_oof.get("aggregated_metrics"),
        "oof_protocol": manifest_oof.get("oof_protocol"),
        "oof_status": manifest_oof.get("oof_status"),
        "oof_weight_mode": manifest_oof.get("oof_weight_mode"),
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
