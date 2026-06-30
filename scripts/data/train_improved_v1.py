#!/usr/bin/env python3
"""Improvements v1: clinical fine-tune, R-Z calibrator, Z-only quantile GKF, no Z multitask blend."""

from __future__ import annotations

import json
import sys
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
from common import compute_regression_table, predict_df  # noqa: E402
from multitask_displacement import MultitaskDisplacementModel  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402
from src.features.pipeline import apply_model_preprocessing, build_inference_matrix  # noqa: E402
from src.models.left_z_calibrator import LeftZCalibrator, TARGET as LEFT_Z_TARGET  # noqa: E402
from src.models.quantile_displacement import QuantileDisplacementPredictor, Z_TARGETS  # noqa: E402
from src.models.right_z_calibrator import RightZCalibrator, TARGET as RIGHT_Z_TARGET  # noqa: E402

BASE_MODEL_PATH = ROOT / "models" / "adaptive_ensemble_full_extended.pkl"
PROCESSED_DIR = ROOT / "data" / "processed_full_extended"
MODEL_PATH = ROOT / "models" / "adaptive_ensemble_improved_v1.pkl"
RUN_ID = f"improved_v1_{date.today().strftime('%Y%m%d')}"
CLINICAL_BOOST = 3.0
MULTITASK_BLEND = {"z": 0.0, "xy": 0.12}


def prepare_frozen_preprocess(
    trainer: AdaptiveEnsembleTrainer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    imputer,
    scaler,
    feature_names_expected: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build train/val matrices without refitting imputer or scaler."""
    out_train = trainer._build_feature_matrix(train_df)
    if out_train[0] is None:
        raise RuntimeError("Failed to build train feature matrix")
    X_train_raw, y_train, feature_cols_train, target_cols_train = out_train

    out_val = trainer._build_feature_matrix(val_df)
    if out_val[0] is None:
        raise RuntimeError("Failed to build val feature matrix")
    X_val_raw, y_val, feature_cols_val, _ = out_val

    if feature_cols_train != feature_cols_val:
        common = [c for c in feature_cols_train if c in feature_cols_val]
        train_indices = [feature_cols_train.index(c) for c in common]
        val_indices = [feature_cols_val.index(c) for c in common]
        X_train_raw = X_train_raw[:, train_indices]
        X_val_raw = X_val_raw[:, val_indices]
        feature_cols = common
    else:
        feature_cols = feature_cols_train

    if feature_names_expected:
        missing = [c for c in feature_names_expected if c not in feature_cols]
        if missing:
            raise ValueError(f"Missing expected features after engineering: {missing[:5]}")
        indices = [feature_cols.index(c) for c in feature_names_expected]
        X_train_raw = X_train_raw[:, indices]
        X_val_raw = X_val_raw[:, indices]
        feature_cols = list(feature_names_expected)

    trainer.imputer = imputer
    trainer.scaler = scaler
    trainer.feature_names = feature_cols
    trainer.target_names = target_cols_train

    X_train_imp = imputer.transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_train_scaled = scaler.transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    trainer.X_train = X_train_scaled
    return X_train_scaled, X_val_scaled, y_train, y_val


def clinical_mask(df: pd.DataFrame) -> np.ndarray:
    if "source" not in df.columns:
        return np.ones(len(df), dtype=bool)
    return df["source"].isin({"Vybor", "Excel"}).values


def finetune_sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    base = (
        pd.to_numeric(train_df.get("sample_weight", 1.0), errors="coerce")
        .fillna(1.0)
        .astype(float)
        .values
    )
    boost = np.where(clinical_mask(train_df), CLINICAL_BOOST, 1.0)
    return base * boost


def _raw_target_preds(
    trainer: AdaptiveEnsembleTrainer,
    df: pd.DataFrame,
    target: str,
) -> np.ndarray:
    df_norm = normalize_dataframe(df)
    X = build_inference_matrix(trainer, df_norm, feature_names=trainer.feature_names)
    X_scaled = apply_model_preprocessing(
        X,
        {"imputer": trainer.imputer, "scaler": trainer.scaler},
    )
    return trainer.trained_models[target].predict(X_scaled)


def _axis_summary(per_target: pd.DataFrame) -> dict[str, float]:
    axes: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    for _, row in per_target.iterrows():
        axis = row["target"].split("_")[-1]
        if axis in axes:
            axes[axis].append(float(row["mae_mm"]))
    return {axis: float(np.mean(vals)) for axis, vals in axes.items() if vals}


def make_bundle(payload: dict):
    return type(
        "Bundle",
        (),
        {
            "mode": "pretrained_adaptive_ensemble",
            "feature_names": payload["feature_names"],
            "target_names": payload.get("target_names", list(payload["models"].keys())),
            "scaler": payload["scaler"],
            "imputer": payload["imputer"],
            "models": payload["models"],
            "left_z_calibrator": payload.get("left_z_calibrator"),
            "right_z_calibrator": payload.get("right_z_calibrator"),
            "multitask_model": payload.get("multitask_model"),
            "multitask_blend": payload.get("multitask_blend"),
            "quantile_model": payload.get("quantile_model"),
        },
    )()


def main() -> int:
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL_PATH}")

    train_df = normalize_dataframe(pd.read_csv(PROCESSED_DIR / "train.csv"))
    val_df = normalize_dataframe(pd.read_csv(PROCESSED_DIR / "validation.csv"))
    print(
        f"[data] train={len(train_df)} val={len(val_df)} "
        f"sources={train_df['source'].value_counts().to_dict()}"
    )

    base_payload = joblib.load(BASE_MODEL_PATH)
    trainer = AdaptiveEnsembleTrainer()
    X_train, X_val, y_train, y_val = prepare_frozen_preprocess(
        trainer,
        train_df,
        val_df,
        imputer=base_payload["imputer"],
        scaler=base_payload["scaler"],
        feature_names_expected=list(base_payload["feature_names"]),
    )

    finetune_w = finetune_sample_weights(train_df)
    trainer.train_sample_weights = finetune_w
    print(
        f"[1] fine-tune on frozen preprocess, clinical_boost={CLINICAL_BOOST}x "
        f"(weight mean={finetune_w.mean():.3f})"
    )
    trainer.train_and_evaluate_adaptive_ensembles(
        X_train,
        X_val,
        y_train,
        y_val,
        sample_weight=finetune_w,
    )

    clinical_train = train_df[clinical_mask(train_df)].copy()
    print(f"[2] L/R Z calibrators on {len(clinical_train)} clinical rows")

    left_cal = LeftZCalibrator()
    raw_lz = _raw_target_preds(trainer, clinical_train, LEFT_Z_TARGET)
    left_cal.fit(clinical_train, raw_lz, clinical_train[LEFT_Z_TARGET].astype(float).values)

    right_cal = RightZCalibrator()
    raw_rz = _raw_target_preds(trainer, clinical_train, RIGHT_Z_TARGET)
    right_cal.fit(clinical_train, raw_rz, clinical_train[RIGHT_Z_TARGET].astype(float).values)

    print(f"[3] quantile Z-only + GroupKFold tuning on clinical groups")
    quantile = QuantileDisplacementPredictor(targets=list(Z_TARGETS))
    group_col = "full_name_key" if "full_name_key" in clinical_train.columns else "case_id"
    clinical_idx = np.where(clinical_mask(train_df))[0]
    X_clinical = X_train[clinical_idx]
    y_clinical = y_train[clinical_idx]
    groups = (
        train_df.loc[clinical_mask(train_df), group_col]
        .astype(str)
        .fillna("unknown")
        .values
    )
    quantile.fit_z_with_groupkfold(
        X_clinical,
        y_clinical,
        trainer.target_names,
        groups,
        sample_weight=finetune_w[clinical_idx],
        n_splits=5,
    )
    q_cov_val = quantile.coverage_rate(X_val, y_val, trainer.target_names)
    print(f"     n_estimators={quantile.n_estimators_by_target}")
    print(f"     holdout Z coverage: {q_cov_val}")

    print(f"[4] multitask blend z=0, xy={MULTITASK_BLEND['xy']}")
    multitask = MultitaskDisplacementModel(n_components=24, alpha=3.0)
    multitask.fit(X_train, y_train, trainer.target_names, sample_weight=finetune_w)

    payload = {
        "models": trainer.trained_models,
        "scaler": trainer.scaler,
        "imputer": trainer.imputer,
        "feature_names": trainer.feature_names,
        "target_names": trainer.target_names,
        "adaptive_weights": trainer.adaptive_weights,
        "best_models": trainer.best_models,
        "left_z_calibrator": left_cal,
        "right_z_calibrator": right_cal,
        "multitask_model": multitask,
        "multitask_blend": dict(MULTITASK_BLEND),
        "quantile_model": quantile,
        "improvement_meta": {
            "base_model": str(BASE_MODEL_PATH),
            "clinical_boost": CLINICAL_BOOST,
            "quantile_targets": list(Z_TARGETS),
            "quantile_n_estimators": quantile.n_estimators_by_target,
        },
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"[OK] saved {MODEL_PATH}")

    bundle = make_bundle(payload)
    val_pred = predict_df(bundle, val_df)
    per_target = compute_regression_table(val_df[TARGET_NAMES], val_pred, list(TARGET_NAMES))

    lz_mae = float(per_target.loc[per_target["target"] == LEFT_Z_TARGET, "mae_mm"].iloc[0])
    rz_mae = float(per_target.loc[per_target["target"] == RIGHT_Z_TARGET, "mae_mm"].iloc[0])

    report = {
        "run_id": RUN_ID,
        "model_path": str(MODEL_PATH),
        "base_model": str(BASE_MODEL_PATH),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "improvements": {
            "1_finetune_clinical_boost": CLINICAL_BOOST,
            "2_right_z_calibrator": right_cal.describe(),
            "3_quantile_z_gkf": {
                "n_estimators": quantile.n_estimators_by_target,
                "holdout_coverage_80": q_cov_val,
            },
            "4_multitask_blend": MULTITASK_BLEND,
            "left_z_calibrator": left_cal.describe(),
        },
        "holdout_18": {
            "per_target_mae_mm": per_target.set_index("target")["mae_mm"].to_dict(),
            "axis_mae_mm": _axis_summary(per_target),
            "avg_mae_mm": float(per_target["mae_mm"].mean()),
            "z_avg_mae_mm": float((lz_mae + rz_mae) / 2.0),
        },
    }

    run_dir = ROOT / "results" / "validation_runs" / RUN_ID / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "improved_v1_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
