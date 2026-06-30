#!/usr/bin/env python3
"""Out-of-sample CT audit: clinical labels vs KiTS/DICOM feature extraction.

Workflow (matches ``results/validation_runs/`` layout):

  * **Clinical (87 xlsx)** — supervised GroupKFold OOF metrics (ground truth).
  * **KiTS19 / DICOM / na_spine / na_boku** — external OOS:
      - feature coverage vs production model inputs;
      - harmonization alignment vs clinical reference;
      - displacement **predictions** (no trusted MAE — proxy KiTS δ are flagged);
      - plausibility vs clinical prediction bounds;
      - utility ranking for imputation / trend extraction (V1/V2 experiment matrix).

KiTS/DICOM are never scored as regression targets. KiTS proxy deltas are compared
only as an **anomaly check** (how far fake labels drift from clinical-trend preds).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "phase1"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

from adaptive_ensemble import AdaptiveEnsembleTrainer  # noqa: E402
from common import (  # noqa: E402
    TARGET_COLUMNS,
    compute_regression_table,
    ensure_run_dirs,
    load_ct_features,
    load_model_bundle,
    predict_df,
    save_manifest,
)
from src.data.xlsx_displacement_parser import build_vybor_from_xlsx  # noqa: E402
from src.features.coordinate_harmonization import alignment_report  # noqa: E402
from src.features.ct_external_enrichment import enrich_external_ct_frame  # noqa: E402
from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES, normalize_dataframe  # noqa: E402
from src.features.projection_enrichment import load_projection_lookup  # noqa: E402
from src.features.pseudo_labeling import vybor_reference_bounds  # noqa: E402

SEED = 42
N_SPLITS = 5
HARMONIZED = ROOT / "data" / "harmonized"
DEFAULT_MODEL = ROOT / "models" / "adaptive_ensemble_clinical_honest.pkl"

FALLBACK_MODELS = [
    DEFAULT_MODEL,
    ROOT / "models" / "adaptive_ensemble_improved_v1.pkl",
    ROOT / "models" / "adaptive_ensemble.pkl",
]


@dataclass(frozen=True)
class CTSource:
    source_id: str
    label: str
    raw_path: Path
    harmonized_name: str | None
    role: str  # supervised | external_oos
    has_proxy_targets: bool = False


CT_SOURCES: list[CTSource] = [
    CTSource(
        "clinical_vybor",
        "Clinical xlsx (paired supine+lateral)",
        ROOT / "data" / "vybor_from_xlsx.csv",
        None,
        "supervised",
    ),
    CTSource(
        "kits19",
        "KiTS19 (external, proxy delta only for anomaly)",
        ROOT / "data" / "kits19_medical_grade_features.csv",
        "kits19_medical_grade_features_aligned.csv",
        "external_oos",
        has_proxy_targets=True,
    ),
    CTSource(
        "dicom",
        "DICOM batch extract",
        ROOT / "data" / "dicom_medical_features.csv",
        "dicom_medical_features_aligned.csv",
        "external_oos",
    ),
    CTSource(
        "na_spine",
        "NA spine CT (supine-only)",
        ROOT / "data" / "na_spine_full.csv",
        "na_spine_full_aligned.csv",
        "external_oos",
    ),
    CTSource(
        "na_boku",
        "NA lateral CT (unpaired)",
        ROOT / "data" / "na_boku_full.bak.csv",
        "na_boku_full_aligned.csv",
        "external_oos",
    ),
]


def _resolve_path(src: CTSource, *, use_harmonized: bool) -> Path | None:
    if use_harmonized and src.harmonized_name:
        hp = HARMONIZED / src.harmonized_name
        if hp.exists():
            return hp
    return src.raw_path if src.raw_path.exists() else None


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Full train-time feature pipeline (matches adaptive_ensemble inference)."""
    trainer = AdaptiveEnsembleTrainer()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        out = normalize_dataframe(df.copy())
        out = trainer._create_engineered_features(out)
        out = trainer._create_cross_features(out)
        from src.features.displacement_axis_features import add_displacement_axis_features
        from src.features.projection_enrichment import (
            add_projection_delta_proxies,
            attach_projection_features,
        )

        out = add_displacement_axis_features(out)
        out = attach_projection_features(out)
        out = add_projection_delta_proxies(out)
    return out


def _feature_coverage(df: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
    trainer = AdaptiveEnsembleTrainer()
    trainer.feature_names = feature_names
    from src.features.pipeline import build_inference_matrix

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        X = build_inference_matrix(trainer, df, feature_names=feature_names)
    row_cov = np.isfinite(X).mean(axis=1)
    col_cov = np.isfinite(X).mean(axis=0)
    base_cols = [c for c in BASE_FEATURES if c in df.columns]
    base_cov = (
        float(
            df[base_cols]
            .apply(lambda c: pd.to_numeric(c, errors="coerce"))
            .notna()
            .mean()
            .mean()
            * 100.0
        )
        if base_cols
        else 0.0
    )
    return {
        "coverage_pct": float(row_cov.mean() * 100.0),
        "coverage_median_pct": float(np.median(row_cov) * 100.0),
        "present_features": int((col_cov > 0).sum()),
        "required_features": len(feature_names),
        "fully_present_features": int((col_cov >= 0.99).sum()),
        "worst_columns": [feature_names[i] for i in np.argsort(col_cov)[:8]],
        "base_feature_coverage_pct": base_cov,
        "inference_reliable": bool(row_cov.mean() >= 0.80),
    }


def _alignment_score(clinical: pd.DataFrame, external: pd.DataFrame) -> dict[str, float]:
    rep = alignment_report(clinical, external, features=BASE_FEATURES)
    if rep.empty:
        return {"mean_abs_median_delta": float("nan"), "n_features": 0}
    deltas = rep["delta_median"].abs()
    ref_iqr = rep["ref_iqr"].replace(0, np.nan)
    z = (deltas / ref_iqr).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "mean_abs_median_delta": float(deltas.mean()),
        "mean_abs_z_vs_iqr": float(z.mean()) if len(z) else float("nan"),
        "n_features": int(len(rep)),
    }


def _prediction_stats(
    pred: pd.DataFrame,
    bounds: tuple[dict, dict, dict],
    *,
    reliable: bool,
) -> dict[str, Any]:
    if not reliable:
        return {
            "per_target": [],
            "within_clinical_p5_p95_pct": 0.0,
            "avg_vector_norm_mm": float("nan"),
            "note": "Skipped: model feature coverage <80% — imputed inputs unreliable for displacement.",
        }
    _, low, high = bounds
    rows = []
    within = []
    for col in TARGET_NAMES:
        if col not in pred.columns:
            continue
        s = pd.to_numeric(pred[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        lo = low.get(col, s.quantile(0.05))
        hi = high.get(col, s.quantile(0.95))
        rows.append(
            {
                "target": col,
                "median_mm": float(s.median()),
                "mean_mm": float(s.mean()),
                "std_mm": float(s.std()),
                "p05_mm": float(s.quantile(0.05)),
                "p95_mm": float(s.quantile(0.95)),
            }
        )
        within.append(float(((s >= lo) & (s <= hi)).mean()))
    return {
        "per_target": rows,
        "within_clinical_p5_p95_pct": float(np.mean(within) * 100.0) if within else 0.0,
        "avg_vector_norm_mm": float(
            np.sqrt(
                pred[[f"kidney_left_delta_{a}" for a in "xyz"]].astype(float).pow(2).sum(axis=1)
            ).mean()
            + np.sqrt(
                pred[[f"kidney_right_delta_{a}" for a in "xyz"]].astype(float).pow(2).sum(axis=1)
            ).mean()
        )
        if all(c in pred.columns for c in TARGET_NAMES)
        else float("nan"),
    }


def _kits_proxy_anomaly(df: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float] | None:
    """Compare KiTS proxy δ to model preds — anomaly diagnostic, not MAE."""
    if not all(c in df.columns for c in TARGET_NAMES):
        return None
    proxy = df[TARGET_NAMES].apply(pd.to_numeric, errors="coerce")
    if proxy.notna().sum().sum() < len(TARGET_NAMES) * 3:
        return None
    complete = proxy.notna().all(axis=1)
    if complete.sum() < 5:
        return None
    gap = (pred.loc[complete, TARGET_NAMES].astype(float) - proxy.loc[complete]).abs()
    return {
        "n_with_proxy": int(complete.sum()),
        "mean_abs_gap_mm": float(gap.mean().mean()),
        "z_axis_gap_mm": float(
            gap[["kidney_left_delta_z", "kidney_right_delta_z"]].mean().mean()
        ),
        "note": "High gap => KiTS proxy δ is anomalous vs clinical-trend model (expected).",
    }


def _utility_score(
    coverage_pct: float,
    alignment: dict[str, float],
    plausibility_pct: float,
    *,
    inference_reliable: bool,
) -> float:
    align_penalty = alignment.get("mean_abs_z_vs_iqr", 5.0)
    if not np.isfinite(align_penalty):
        align_penalty = 5.0
    align_component = max(0.0, 1.0 - min(align_penalty, 5.0) / 5.0)
    plaus = (plausibility_pct / 100.0) if inference_reliable else 0.0
    return float(0.50 * (coverage_pct / 100.0) + 0.35 * align_component + 0.15 * plaus)


def evaluate_clinical_gkf(clinical: pd.DataFrame, bundle) -> dict[str, Any]:
    clinical = clinical.dropna(subset=TARGET_NAMES, how="any").reset_index(drop=True)
    name_col = "full_name" if "full_name" in clinical.columns else "case_id"
    groups = clinical[name_col].astype(str).values
    gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))

    oof = {t: np.full(len(clinical), np.nan) for t in TARGET_NAMES}
    for train_idx, val_idx in gkf.split(clinical, clinical[TARGET_NAMES[0]], groups=groups):
        tr = clinical.iloc[train_idx].reset_index(drop=True)
        te = clinical.iloc[val_idx].reset_index(drop=True)
        fold_trainer = AdaptiveEnsembleTrainer()
        X_tr, X_te, y_tr, y_te = fold_trainer.prepare_training_data_split(tr, te)
        g_tr = tr[name_col].astype(str).values
        fold_trainer.train_and_evaluate_adaptive_ensembles(
            X_tr, X_te, y_tr, y_te, groups=g_tr, fast_weights=True
        )
        fold_bundle = type(
            "B",
            (),
            {
                "mode": "pretrained_adaptive_ensemble",
                "feature_names": fold_trainer.feature_names,
                "target_names": fold_trainer.target_names,
                "scaler": fold_trainer.scaler,
                "imputer": fold_trainer.imputer,
                "models": fold_trainer.trained_models,
                "left_z_calibrator": None,
                "right_z_calibrator": None,
            },
        )()
        pred = predict_df(fold_bundle, te)
        for t in TARGET_NAMES:
            oof[t][val_idx] = pred[t].values

    pred_df = pd.DataFrame(oof, index=clinical.index)
    per_target = compute_regression_table(clinical[TARGET_NAMES], pred_df, list(TARGET_NAMES))
    z_targets = [t for t in TARGET_NAMES if t.endswith("_z")]
    return {
        "n_patients": len(clinical),
        "protocol": f"GroupKFold({min(N_SPLITS, len(np.unique(groups)))}) OOF",
        "avg_mae_mm": float(per_target["mae_mm"].mean()),
        "z_avg_mae_mm": float(per_target.loc[per_target["target"].isin(z_targets), "mae_mm"].mean()),
        "per_target_mae_mm": per_target.set_index("target")["mae_mm"].to_dict(),
    }


def audit_external(
    src: CTSource,
    df: pd.DataFrame,
    clinical_ref: pd.DataFrame,
    bundle,
    bounds: tuple[dict, dict, dict],
) -> dict[str, Any]:
    coverage = _feature_coverage(df, bundle.feature_names)
    alignment = _alignment_score(clinical_ref, df)
    reliable = bool(coverage.get("inference_reliable")) or src.role == "supervised"
    pred = predict_df(bundle, df) if reliable else pd.DataFrame(
        {t: np.full(len(df), np.nan) for t in TARGET_NAMES}, index=df.index
    )
    pred_stats = _prediction_stats(pred, bounds, reliable=reliable)
    out: dict[str, Any] = {
        "source_id": src.source_id,
        "label": src.label,
        "role": src.role,
        "n_rows": len(df),
        "feature_coverage": coverage,
        "alignment_vs_clinical": alignment,
        "predictions": pred_stats,
        "utility_score": _utility_score(
            coverage["coverage_pct"],
            alignment,
            pred_stats["within_clinical_p5_p95_pct"],
            inference_reliable=reliable,
        ),
        "scoring_note": (
            "Supervised MAE only for clinical_vybor. "
            "External: extraction QA + optional clipped preds when coverage>=80%."
        ),
    }
    if src.has_proxy_targets and reliable:
        out["kits_proxy_anomaly"] = _kits_proxy_anomaly(df, pred)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clinical + external CT out-of-sample audit")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "validation_runs")
    p.add_argument("--harmonize", action="store_true", help="Run harmonize_extracted_datasets.py first")
    p.add_argument("--skip-gkf", action="store_true", help="Skip clinical GroupKFold (faster)")
    p.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip clinical-parity enrichment (projection, spans, anatomical extras)",
    )
    return p.parse_args()


def _pick_model(explicit: Path | None) -> Path:
    if explicit and explicit.exists():
        return explicit
    for p in FALLBACK_MODELS:
        if p.exists():
            return p
    raise FileNotFoundError("No displacement model artifact found")


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"external_ct_audit_{date.today().strftime('%Y%m%d')}"
    run_dir = ensure_run_dirs(args.out_dir, run_id)
    model_path = _pick_model(args.model)

    if args.harmonize:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "data" / "harmonize_extracted_datasets.py"),
                "--reference",
                str(ROOT / "data" / "vybor_from_xlsx.csv"),
            ],
            cwd=str(ROOT),
            check=True,
        )

    xlsx_files = list(ROOT.glob("data/*.xlsx"))
    if not (ROOT / "data" / "vybor_from_xlsx.csv").exists() and xlsx_files:
        build_vybor_from_xlsx(
            str(xlsx_files[0]),
            boku_path=str(ROOT / "data" / "na_boku_full.bak.csv"),
        )

    bundle = load_model_bundle(model_path)
    clinical_path = ROOT / "data" / "vybor_from_xlsx.csv"
    clinical = load_ct_features(clinical_path)
    clinical = clinical.dropna(subset=TARGET_NAMES, how="any").reset_index(drop=True)
    bounds = vybor_reference_bounds(clinical)

    report: dict[str, Any] = {
        "run_id": run_id,
        "model_path": str(model_path),
        "clinical_reference": {
            "path": str(clinical_path),
            "n_complete": len(clinical),
            "target_medians": bounds[0],
        },
        "protocol": {
            "supervised": "Clinical xlsx GroupKFold OOF (trusted labels)",
            "external": "KiTS/DICOM/NA — feature audit + prediction plausibility (no MAE)",
            "enrichment": not args.no_enrich,
        },
        "sources": [],
    }

    if not args.skip_gkf:
        print("[clinical] GroupKFold OOF ...")
        report["clinical_gkf_oof"] = evaluate_clinical_gkf(clinical, bundle)
        print(f"  avg MAE={report['clinical_gkf_oof']['avg_mae_mm']:.3f} mm")
    else:
        honest_report = (
            ROOT / "results" / "validation_runs" / "clinical_honest_20260630" / "metrics"
            / "clinical_honest_report.json"
        )
        if honest_report.exists():
            report["clinical_gkf_oof"] = json.loads(
                honest_report.read_text(encoding="utf-8")
            ).get("groupkfold_oof_87")
            report["clinical_gkf_oof_source"] = str(honest_report)

    use_harmonized = HARMONIZED.exists()
    external_ranking: list[dict[str, Any]] = []
    projection_lookup = load_projection_lookup() if not args.no_enrich else None
    boku_path = ROOT / "data" / "na_boku_full.bak.csv"

    for src in CT_SOURCES:
        path = _resolve_path(src, use_harmonized=use_harmonized)
        if path is None:
            print(f"[skip] {src.source_id}: file missing")
            continue
        print(f"[audit] {src.source_id} ({path.name}) ...")
        raw_df = load_ct_features(path)
        if src.role == "supervised":
            raw_df = raw_df.dropna(subset=TARGET_NAMES, how="any")

        coverage_before = _feature_coverage(raw_df, bundle.feature_names)
        enrich_meta: dict[str, Any] | None = None
        if args.no_enrich:
            df = raw_df
        else:
            df, enrich_meta = enrich_external_ct_frame(
                raw_df,
                clinical_reference=clinical,
                projection_lookup=projection_lookup,
                boku_path=boku_path if boku_path.exists() else None,
                source_id=src.source_id,
            )
            gain = (
                _feature_coverage(df, bundle.feature_names)["coverage_pct"]
                - coverage_before["coverage_pct"]
            )
            print(
                f"  enrich: {coverage_before['coverage_pct']:.1f}% -> "
                f"{_feature_coverage(df, bundle.feature_names)['coverage_pct']:.1f}% "
                f"(+{gain:.1f} pp)"
            )

        entry = audit_external(src, df, clinical, bundle, bounds)
        entry["feature_coverage_before"] = coverage_before
        if enrich_meta:
            entry["enrichment"] = enrich_meta
            entry["coverage_gain_pp"] = float(
                entry["feature_coverage"]["coverage_pct"] - coverage_before["coverage_pct"]
            )
        report["sources"].append(entry)

        pred_path = run_dir / "predictions" / f"{src.source_id}_predictions.csv"
        if entry["feature_coverage"].get("inference_reliable"):
            pred_df = predict_df(bundle, df)
        else:
            pred_df = pd.DataFrame(
                {t: np.nan for t in TARGET_NAMES}, index=range(len(df))
            )
        meta = df[[c for c in ("case_id", "full_name", "source", "universal_id") if c in df.columns]]
        out_pred = pd.concat([meta.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
        out_pred.to_csv(pred_path, index=False)

        if src.role == "external_oos":
            external_ranking.append(
                {
                    "source_id": src.source_id,
                    "label": src.label,
                    "n_rows": entry["n_rows"],
                    "utility_score": entry["utility_score"],
                    "coverage_pct": entry["feature_coverage"]["coverage_pct"],
                    "coverage_before_pct": entry.get("feature_coverage_before", {}).get(
                        "coverage_pct"
                    ),
                    "coverage_gain_pp": entry.get("coverage_gain_pp"),
                    "fully_present_features": entry["feature_coverage"].get(
                        "fully_present_features", 0
                    ),
                    "inference_reliable": entry["feature_coverage"].get(
                        "inference_reliable", False
                    ),
                    "alignment_z": entry["alignment_vs_clinical"].get("mean_abs_z_vs_iqr"),
                    "plausibility_pct": entry["predictions"]["within_clinical_p5_p95_pct"],
                    "kits_proxy_gap_mm": (
                        entry.get("kits_proxy_anomaly", {}) or {}
                    ).get("mean_abs_gap_mm"),
                }
            )

    external_ranking.sort(key=lambda r: r["utility_score"], reverse=True)
    report["external_utility_ranking"] = external_ranking
    report["recommendation"] = {
        "best_for_imputation_pca": external_ranking[0]["source_id"] if external_ranking else None,
        "kits19_note": (
            "KiTS19 has proxy δ with wrong clinical sign; use for CT feature stats only, "
            "not as displacement labels. High proxy_gap confirms anomaly."
        ),
        "dicom_note": "DICOM rows have no labels — compare prediction distributions only.",
        "training_rule": "Train only on clinical xlsx; external sets inform extraction QA.",
        "enrichment": (
            "Projection join + spans + anatomical frame + clinical driver medians "
            "(see src/features/ct_external_enrichment.py)"
        ),
    }

    metrics_path = run_dir / "metrics" / "external_ct_audit.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    ranking_df = pd.DataFrame(external_ranking)
    if len(ranking_df):
        ranking_df.to_csv(run_dir / "metrics" / "external_utility_ranking.csv", index=False)

    save_manifest(
        run_dir,
        run_id=run_id,
        dataset_path=clinical_path,
        model_path=model_path,
        predictor_mode=bundle.mode,
        train_count=0,
        eval_count=len(clinical),
        seed=SEED,
        holdout_eval=False,
    )

    print("\n=== External CT utility ranking (higher = better for imputation/trend) ===")
    if len(ranking_df):
        print(ranking_df.to_string(index=False))
        print(f"\nRecommended CT aux source: {report['recommendation']['best_for_imputation_pca']}")
    print(f"\nReport -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
