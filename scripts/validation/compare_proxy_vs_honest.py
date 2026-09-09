#!/usr/bin/env python3
"""Compare nested GroupKFold OOF: honest vs proxy on the same clinical folds.

Never scores proxy by retraining the honest-only ensemble. Cached nested
reports are preferred; pass --recompute to run the nested protocol (slow).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

from src.models.ensemble import AdaptiveEnsembleTrainer  # noqa: E402
from common import compute_regression_table, predict_df  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402
from src.models.nested_cv import evaluate_nested_proxy_vs_honest  # noqa: E402
from train_clinical_honest import N_SPLITS  # noqa: E402
from train_clinical_proxy import (  # noqa: E402
    HARMONIZED_DIR,
    build_proxy_train_for_fold,
)

HONEST_REPORTS = [
    ROOT / "results" / "validation_runs" / "clinical_honest_20260630" / "metrics" / "clinical_honest_report.json",
]
PROXY_REPORTS = [
    ROOT / "results" / "validation_runs" / "clinical_proxy_20260630" / "metrics" / "clinical_proxy_report.json",
]


def load_bundle(path: Path):
    import joblib

    p = joblib.load(path)
    return type(
        "Bundle",
        (),
        {
            "mode": "pretrained_adaptive_ensemble",
            "feature_names": p["feature_names"],
            "target_names": p.get("target_names", list(p["models"].keys())),
            "scaler": p["scaler"],
            "imputer": p["imputer"],
            "models": p["models"],
            "left_z_calibrator": p.get("left_z_calibrator"),
            "right_z_calibrator": p.get("right_z_calibrator"),
            "z_head": p.get("z_head", "ensemble"),
            "z_driver_names": p.get("z_driver_names"),
        },
    )()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _honest_from_report(saved: dict) -> dict | None:
    nested = saved.get("nested_groupkfold_oof") or saved.get("artifact_oof")
    if isinstance(nested, dict) and nested.get("per_target_mae_mm"):
        return nested
    cached = saved.get("groupkfold_oof_87") or saved.get("groupkfold_oof_clinical_87")
    return cached if isinstance(cached, dict) else None


def _proxy_from_report(saved: dict) -> dict | None:
    nested = saved.get("nested_proxy_vs_honest")
    if isinstance(nested, dict) and nested.get("clinical_proxy"):
        return nested
    return None


def recompute_nested_proxy_vs_honest(clinical: pd.DataFrame) -> dict:
    kits_path = HARMONIZED_DIR / "kits19_medical_grade_features_aligned.csv"
    if not kits_path.exists():
        kits_path = ROOT / "data" / "kits19_medical_grade_features.csv"
    kits_df = pd.read_csv(kits_path) if kits_path.exists() else None
    dicom_path = HARMONIZED_DIR / "dicom_medical_features_aligned.csv"
    dicom_df = pd.read_csv(dicom_path) if dicom_path.exists() else None

    def trainer_factory(**kwargs):
        params = {"enrichment_mode": "none"}
        params.update(kwargs)
        return AdaptiveEnsembleTrainer(**params)

    def proxy_train_builder(train_clinical, fold_teacher):
        return build_proxy_train_for_fold(
            train_clinical,
            fold_teacher,
            kits_df=kits_df,
            dicom_df=dicom_df,
            reference_df=train_clinical,
        )

    return evaluate_nested_proxy_vs_honest(
        clinical,
        trainer_factory=trainer_factory,
        proxy_train_builder=proxy_train_builder,
        n_splits=N_SPLITS,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Nested honest vs proxy OOF comparison")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Run nested proxy-vs-honest OOF (slow). Never uses honest-only OOF as a proxy metric.",
    )
    args = parser.parse_args()

    vybor = normalize_dataframe(pd.read_csv(ROOT / "data" / "vybor_from_xlsx.csv"))
    clinical = vybor.dropna(subset=list(TARGET_NAMES), how="any").reset_index(drop=True)

    honest = None
    for path in HONEST_REPORTS:
        saved = _load_json(path)
        if saved:
            honest = _honest_from_report(saved)
            if honest:
                print(f"[eval] clinical_honest: loaded nested/cached OOF from {path.name}")
                break

    nested = None
    for path in PROXY_REPORTS:
        saved = _load_json(path)
        if saved:
            nested = _proxy_from_report(saved)
            if nested:
                print(f"[eval] clinical_proxy: loaded nested comparison from {path.name}")
                break

    if args.recompute or nested is None:
        if args.recompute:
            print(f"[eval] recomputing nested proxy vs honest on {len(clinical)} patients...")
            nested = recompute_nested_proxy_vs_honest(clinical)
        else:
            print(
                "[skip] no nested_proxy_vs_honest report found. "
                "Run scripts/data/train_clinical_proxy.py or pass --recompute. "
                "Refusing to score proxy via honest-only OOF."
            )

    proxy = nested.get("clinical_proxy") if nested else None
    if nested and nested.get("clinical_honest") and honest is None:
        honest = nested["clinical_honest"]

    if honest and proxy:
        print("\n=== nested GKF OOF on clinical (mm) ===")
        header = f"{'target':28s} {'honest':>8s} {'proxy':>8s} {'delta':>8s}"
        print(header)
        for t in TARGET_NAMES:
            hv = honest["per_target_mae_mm"][t]
            pv = proxy["per_target_mae_mm"][t]
            print(f"{t:28s} {hv:8.3f} {pv:8.3f} {pv - hv:+8.3f}")
        for k in ["avg_mae_mm", "z_avg_mae_mm"]:
            hv, pv = honest[k], proxy[k]
            print(f"{k:28s} {hv:8.3f} {pv:8.3f} {pv - hv:+8.3f}")

    proxy_path = ROOT / "models" / "adaptive_ensemble_clinical_proxy.pkl"
    val_path = ROOT / "data" / "processed_proxy" / "validation.csv"
    holdout = None
    if proxy_path.exists() and val_path.exists():
        val = normalize_dataframe(pd.read_csv(val_path))
        bundle = load_bundle(proxy_path)
        pred = predict_df(bundle, val)
        ht = compute_regression_table(val[TARGET_NAMES], pred, list(TARGET_NAMES))
        holdout = {
            "holdout_18_mae": float(ht["mae_mm"].mean()),
            "holdout_per_target_mae": dict(zip(ht["target"], ht["mae_mm"])),
            "note": "integration holdout; not nested OOF",
        }
        print(f"\nproxy holdout (integration val, NOT nested OOF): {holdout['holdout_18_mae']:.3f} mm avg")

    out_dir = ROOT / "results" / "validation_runs" / "clinical_proxy_20260630" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    delta = None
    if nested and nested.get("delta_proxy_minus_honest"):
        delta = nested["delta_proxy_minus_honest"]
    elif honest and proxy:
        delta = {
            "avg_mae_mm": proxy["avg_mae_mm"] - honest["avg_mae_mm"],
            "z_avg_mae_mm": proxy["z_avg_mae_mm"] - honest["z_avg_mae_mm"],
            "per_target_mae_mm": {
                t: proxy["per_target_mae_mm"][t] - honest["per_target_mae_mm"][t]
                for t in TARGET_NAMES
            },
        }
    report = {
        "protocol": "nested_proxy_vs_honest",
        "clinical_honest_oof": honest,
        "clinical_proxy_oof": proxy,
        "nested_proxy_vs_honest": nested,
        "delta_proxy_minus_honest": delta,
        "proxy_holdout": holdout,
    }
    out_path = out_dir / "clinical_proxy_comparison.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
