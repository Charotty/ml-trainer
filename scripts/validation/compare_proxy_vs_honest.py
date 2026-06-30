#!/usr/bin/env python3
"""Compare GKF-5 OOF on 87 clinical: honest vs proxy-weighted model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "phase1"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

from common import compute_regression_table, predict_df  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402
from train_clinical_honest import evaluate_groupkfold_oof  # noqa: E402


def load_bundle(path: Path):
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


def main() -> int:
    vybor = normalize_dataframe(pd.read_csv(ROOT / "data" / "vybor_from_xlsx.csv"))
    clinical = vybor.dropna(subset=list(TARGET_NAMES), how="any").reset_index(drop=True)

    models = {
        "clinical_honest": ROOT / "models" / "adaptive_ensemble_clinical_honest.pkl",
        "clinical_proxy": ROOT / "models" / "adaptive_ensemble_clinical_proxy.pkl",
    }
    results: dict = {}
    honest_report = ROOT / "results" / "validation_runs" / "clinical_honest_20260630" / "metrics" / "clinical_honest_report.json"
    if honest_report.exists():
        saved = json.loads(honest_report.read_text(encoding="utf-8"))
        results["clinical_honest"] = saved.get("groupkfold_oof_87") or saved.get("groupkfold_oof_clinical_87")
        print(f"[eval] clinical_honest: loaded cached OOF from {honest_report.name}")

    proxy_path = models["clinical_proxy"]
    if proxy_path.exists():
        print(f"[eval] clinical_proxy GKF-5 OOF on {len(clinical)} patients...")
        results["clinical_proxy"] = evaluate_groupkfold_oof(clinical)
    else:
        print(f"[skip] clinical_proxy: missing {proxy_path}")

    honest = results.get("clinical_honest")
    proxy = results.get("clinical_proxy")
    if honest and proxy:
        print("\n=== GKF-5 OOF on 87 clinical (mm) ===")
        header = f"{'target':28s} {'honest':>8s} {'proxy':>8s} {'delta':>8s}"
        print(header)
        for t in TARGET_NAMES:
            hv = honest["per_target_mae_mm"][t]
            pv = proxy["per_target_mae_mm"][t]
            print(f"{t:28s} {hv:8.3f} {pv:8.3f} {pv - hv:+8.3f}")
        for k in ["avg_mae_mm", "z_avg_mae_mm"]:
            hv, pv = honest[k], proxy[k]
            print(f"{k:28s} {hv:8.3f} {pv:8.3f} {pv - hv:+8.3f}")

    val_path = ROOT / "data" / "processed_proxy" / "validation.csv"
    if proxy and val_path.exists():
        val = normalize_dataframe(pd.read_csv(val_path))
        bundle = load_bundle(models["clinical_proxy"])
        pred = predict_df(bundle, val)
        ht = compute_regression_table(val[TARGET_NAMES], pred, list(TARGET_NAMES))
        holdout_mae = float(ht["mae_mm"].mean())
        results["clinical_proxy"]["holdout_18_mae"] = holdout_mae
        results["clinical_proxy"]["holdout_per_target_mae"] = dict(
            zip(ht["target"], ht["mae_mm"])
        )
        print(f"\nproxy holdout (18 Vybor, NOT honest): {holdout_mae:.3f} mm avg")

    out_dir = ROOT / "results" / "validation_runs" / "clinical_proxy_20260630" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "clinical_honest_oof": honest,
        "clinical_proxy_oof": proxy,
        "delta_proxy_minus_honest": {
            "avg_mae_mm": proxy["avg_mae_mm"] - honest["avg_mae_mm"],
            "z_avg_mae_mm": proxy["z_avg_mae_mm"] - honest["z_avg_mae_mm"],
            "per_target_mae_mm": {
                t: proxy["per_target_mae_mm"][t] - honest["per_target_mae_mm"][t]
                for t in TARGET_NAMES
            },
        }
        if honest and proxy
        else None,
    }
    out_path = out_dir / "clinical_proxy_comparison.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
