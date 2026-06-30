#!/usr/bin/env python3
"""Honest experiment matrix for kidney displacement.

Variants (all evaluated with patient GroupKFold(5) + bootstrap CI):
  V0  clinical-only baseline (87 xlsx patients, leakage-free features)
  V1  clinical-only + missing-input imputation from CT sources (na_spine/boku/kits)
  V2  clinical-only + unsupervised CT representation (PCA on CT features as extra inputs)
  V6  clinical-only + dedicated Z-specialist on clinical drivers (X/Y as V0)

Key correctness rules enforced here:
  * the ONLY real paired labels are the 87 xlsx patients;
  * features containing lateral-scan info (``*_delta_span_mm``, ``*lateral*``) are
    DROPPED -- they are target leakage and unavailable for a new supine-only patient;
  * imputer/scaler/PCA are fit on TRAIN folds only (no leakage);
  * CT sources are NEVER used as regression targets.
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "phase1"))

from adaptive_ensemble import AdaptiveEnsembleTrainer  # noqa: E402
from src.data.xlsx_displacement_parser import build_vybor_from_xlsx  # noqa: E402
from src.features.ct_external_enrichment import enrich_external_ct_frame  # noqa: E402
from src.features.phase1_schema import TARGET_NAMES, normalize_dataframe  # noqa: E402
from src.features.projection_enrichment import (  # noqa: E402
    add_projection_delta_proxies,
    attach_projection_features,
)

SEED = 42
N_SPLITS = 5
N_BOOTSTRAP = 2000
PCA_COMPONENTS = 10
TARGETS = list(TARGET_NAMES)
Z_TARGETS = ["kidney_left_delta_z", "kidney_right_delta_z"]
HARMONIZED = ROOT / "data" / "harmonized"
BOKU_PATH = ROOT / "data" / "na_boku_full.bak.csv"

# Harmonized paths preferred (audit: DICOM best for CT aux after enrichment).
CT_SOURCE_PATHS = [
    HARMONIZED / "na_spine_full_aligned.csv",
    HARMONIZED / "na_boku_full_aligned.csv",
    HARMONIZED / "kits19_medical_grade_features_aligned.csv",
    HARMONIZED / "dicom_medical_features_aligned.csv",
    ROOT / "data" / "na_spine_full.csv",
    ROOT / "data" / "na_boku_full.bak.csv",
    ROOT / "data" / "kits19_medical_grade_features.csv",
    ROOT / "data" / "dicom_medical_features.csv",
]

LEAKY_SUBSTRINGS = ("delta_span", "lateral")
ID_LIKE = {
    "case_id", "full_name", "full_name_key", "fio", "patient_name", "source",
    "source_name", "source_id", "universal_id", "label_quality", "data_origin",
    "scan_position", "patient_position", "contrast_phase", "dicom_cohort",
    "study_date", "harmonization_source", "harmonization_applied",
}

CLINICAL_DRIVERS = [
    "bmi", "body_type", "age", "sex",
    "lumbar_lordosis_deg", "s1_plate_tilt_deg", "abd_wall_thickness_mm",
    "body_depth_mm", "body_width_mm", "body_sagittal_index",
    "lordosis_x_depth", "abd_wall_over_depth",
    "kidney_left_z_span_supine_mm", "kidney_right_z_span_supine_mm",
    "kidney_left_center_z_rel", "kidney_right_center_z_rel",
    "kidney_z_asymmetry_rel",
    "kidney_left_z_over_depth", "kidney_right_z_over_depth",
]

# Supine-only projection coords (no lateral leakage) for Z specialist variants.
Z_SAFE_EXTRA = [
    "proj_sup_kidney_left_center_z_rel",
    "proj_sup_kidney_right_center_z_rel",
    "proj_sup_body_depth_mm",
    "kidney_lr_sep_z",
]

CT_FILES = []  # legacy name kept for report metadata


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Full leakage-safe feature pipeline (matches production inference)."""
    trainer = AdaptiveEnsembleTrainer()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        out = normalize_dataframe(df.copy())
        out = trainer._create_engineered_features(out)
        out = trainer._create_cross_features(out)
        from src.features.displacement_axis_features import add_displacement_axis_features

        out = add_displacement_axis_features(out)
        out = attach_projection_features(out)
        out = add_projection_delta_proxies(out)
    return out


def _numeric_feature_columns(engineered: pd.DataFrame) -> list[str]:
    cols = []
    for c in engineered.columns:
        if c in TARGETS or c in ID_LIKE:
            continue
        if any(sub in c for sub in LEAKY_SUBSTRINGS):
            continue
        series = pd.to_numeric(engineered[c], errors="coerce")
        if series.notna().sum() == 0:
            continue
        cols.append(c)
    return sorted(cols)


def _build_target_model(target: str):
    axis = target.split("_")[-1]
    if axis == "z":
        return GradientBoostingRegressor(
            loss="huber", alpha=0.9, n_estimators=500, learning_rate=0.04,
            max_depth=3, subsample=0.85, random_state=SEED,
        )
    if axis == "y":
        return GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            subsample=0.85, random_state=SEED,
        )
    return RandomForestRegressor(
        n_estimators=400, max_depth=14, min_samples_leaf=2,
        max_features="sqrt", random_state=SEED, n_jobs=-1,
    )


@dataclass
class FoldContext:
    feature_cols: list[str]
    train_medians: pd.Series
    ct_medians: pd.Series | None = None
    pca: PCA | None = None
    pca_scaler_mean: np.ndarray | None = None
    pca_scaler_std: np.ndarray | None = None


def _impute(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return X.apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(medians).fillna(0.0)


def _ct_reference_matrix(feature_cols: list[str], clinical_ref: pd.DataFrame) -> pd.DataFrame:
    """Build enriched CT aux matrix for V1/V2 (never used as regression targets)."""
    seen: set[str] = set()
    frames: list[pd.DataFrame] = []
    for path in CT_SOURCE_PATHS:
        key = path.name
        if not path.exists() or key in seen:
            continue
        seen.add(key)
        frames.append(pd.read_csv(path, low_memory=False))

    if not frames:
        return pd.DataFrame(columns=feature_cols)

    ct = pd.concat(frames, ignore_index=True)
    ct, enrich_meta = enrich_external_ct_frame(
        ct,
        clinical_reference=clinical_ref,
        boku_path=BOKU_PATH if BOKU_PATH.exists() else None,
        source_id="experiment_matrix_ct_aux",
    )
    print(
        f"[ct] enriched aux rows={len(ct)} steps={enrich_meta.get('steps', [])}"
    )
    eng = _engineer(ct)
    for c in feature_cols:
        if c not in eng.columns:
            eng[c] = np.nan
    return eng[feature_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))


# --- variant fit/predict functions: (train_X, train_y_df, test_X, target) -> pred ---

def variant_v0(train_X, train_y, test_X, ct_matrix):
    medians = train_X.median(numeric_only=True)
    Xtr = _impute(train_X, medians)
    Xte = _impute(test_X, medians)
    preds = {}
    for t in TARGETS:
        m = _build_target_model(t)
        m.fit(Xtr.values, train_y[t].values)
        preds[t] = m.predict(Xte.values)
    return preds


def variant_v1(train_X, train_y, test_X, ct_matrix):
    train_med = train_X.median(numeric_only=True)
    ct_med = ct_matrix.median(numeric_only=True) if len(ct_matrix) else pd.Series(dtype=float)
    # prefer CT medians for inputs missing in this fold, else train median
    fill = train_med.copy()
    for c in fill.index:
        if (pd.isna(fill[c]) or train_X[c].isna().mean() > 0.5) and c in ct_med and pd.notna(ct_med[c]):
            fill[c] = ct_med[c]
    Xtr = _impute(train_X, fill)
    Xte = _impute(test_X, fill)
    preds = {}
    for t in TARGETS:
        m = _build_target_model(t)
        m.fit(Xtr.values, train_y[t].values)
        preds[t] = m.predict(Xte.values)
    return preds


def variant_v2(train_X, train_y, test_X, ct_matrix):
    medians = train_X.median(numeric_only=True)
    Xtr = _impute(train_X, medians)
    Xte = _impute(test_X, medians)

    comps = 0
    if len(ct_matrix) >= PCA_COMPONENTS + 2:
        ct_med = ct_matrix.median(numeric_only=True)
        ct_imp = ct_matrix.fillna(ct_med).fillna(0.0)
        mu = ct_imp.mean().values
        sd = ct_imp.std().replace(0, 1.0).values
        ct_std = (ct_imp.values - mu) / sd
        comps = min(PCA_COMPONENTS, ct_std.shape[1], ct_std.shape[0] - 1)
        pca = PCA(n_components=comps, random_state=SEED).fit(ct_std)
        tr_p = pca.transform((Xtr.values - mu) / sd)
        te_p = pca.transform((Xte.values - mu) / sd)
        for i in range(comps):
            Xtr[f"_ctpca_{i}"] = tr_p[:, i]
            Xte[f"_ctpca_{i}"] = te_p[:, i]
    preds = {}
    for t in TARGETS:
        m = _build_target_model(t)
        m.fit(Xtr.values, train_y[t].values)
        preds[t] = m.predict(Xte.values)
    return preds


def variant_v6(train_X, train_y, test_X, ct_matrix):
    medians = train_X.median(numeric_only=True)
    Xtr = _impute(train_X, medians)
    Xte = _impute(test_X, medians)
    drivers = [c for c in CLINICAL_DRIVERS if c in train_X.columns]
    Ztr = Xtr[drivers]
    Zte = Xte[drivers]
    preds = {}
    for t in TARGETS:
        if t in Z_TARGETS:
            m = _build_target_model(t)
            m.fit(Ztr.values, train_y[t].values)
            preds[t] = m.predict(Zte.values)
        else:
            m = _build_target_model(t)
            m.fit(Xtr.values, train_y[t].values)
            preds[t] = m.predict(Xte.values)
    return preds


def variant_v7(train_X, train_y, test_X, ct_matrix):
    """Z: median quantile regression on clinical drivers (literature: robust small-n)."""
    medians = train_X.median(numeric_only=True)
    Xtr = _impute(train_X, medians)
    Xte = _impute(test_X, medians)
    z_drivers = [c for c in CLINICAL_DRIVERS + Z_SAFE_EXTRA if c in train_X.columns]
    preds = {}
    for t in TARGETS:
        if t in Z_TARGETS and z_drivers:
            m = QuantileRegressor(
                quantile=0.5,
                alpha=0.1,
                solver="highs",
            )
            m.fit(Xtr[z_drivers].values, train_y[t].values)
            preds[t] = m.predict(Xte[z_drivers].values)
        else:
            m = _build_target_model(t)
            m.fit(Xtr.values, train_y[t].values)
            preds[t] = m.predict(Xte.values)
    return preds


VARIANTS: dict[str, Callable] = {
    "V0_clinical_baseline": variant_v0,
    "V1_ct_imputation": variant_v1,
    "V2_ct_pca_pretrain": variant_v2,
    "V6_z_specialist": variant_v6,
    "V7_z_quantile_drivers": variant_v7,
}


def _bootstrap_ci(per_patient_avg: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(per_patient_avg)
    means = [per_patient_avg[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate_variant(name, fn, engineered, feature_cols, y_df, groups, ct_matrix):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = {t: np.full(len(engineered), np.nan) for t in TARGETS}
    X_all = engineered[feature_cols]
    for train_idx, test_idx in gkf.split(X_all, y_df[TARGETS[0]], groups=groups):
        tr_X = X_all.iloc[train_idx].reset_index(drop=True)
        te_X = X_all.iloc[test_idx].reset_index(drop=True)
        tr_y = y_df.iloc[train_idx].reset_index(drop=True)
        preds = fn(tr_X.copy(), tr_y, te_X.copy(), ct_matrix)
        for t in TARGETS:
            oof[t][test_idx] = preds[t]

    abs_err = np.column_stack([np.abs(oof[t] - y_df[t].values) for t in TARGETS])
    per_patient_avg = abs_err.mean(axis=1)
    per_target = {t: float(np.mean(np.abs(oof[t] - y_df[t].values))) for t in TARGETS}
    axis = {
        "x": float(np.mean([per_target[t] for t in TARGETS if t.endswith("_x")])),
        "y": float(np.mean([per_target[t] for t in TARGETS if t.endswith("_y")])),
        "z": float(np.mean([per_target[t] for t in TARGETS if t.endswith("_z")])),
    }
    lo, hi = _bootstrap_ci(per_patient_avg)
    return {
        "variant": name,
        "avg_mae_mm": float(per_patient_avg.mean()),
        "avg_mae_ci95": [lo, hi],
        "z_avg_mae_mm": axis["z"],
        "axis_mae_mm": axis,
        "per_target_mae_mm": per_target,
        "n_patients": int(len(per_patient_avg)),
    }


def main() -> int:
    xlsx = glob.glob(str(ROOT / "data" / "*.xlsx"))
    if not xlsx:
        raise FileNotFoundError("No xlsx in data/")
    print(f"[data] clinical source: {xlsx[0]}")
    clinical = build_vybor_from_xlsx(
        xlsx[0],
        boku_path=str(ROOT / "data" / "na_boku_full.bak.csv"),
    )
    clinical = clinical.dropna(subset=TARGETS, how="any").reset_index(drop=True)
    print(f"[data] clinical paired patients: {len(clinical)}")

    engineered = _engineer(clinical).reset_index(drop=True)
    feature_cols = _numeric_feature_columns(engineered)
    dropped_leaky = sorted(c for c in engineered.columns
                           if any(s in c for s in LEAKY_SUBSTRINGS) and c not in TARGETS)
    print(f"[features] used={len(feature_cols)} | dropped_leaky={dropped_leaky}")

    y_df = clinical[TARGETS].astype(float).reset_index(drop=True)
    name_col = "full_name" if "full_name" in clinical.columns else "case_id"
    groups = clinical[name_col].astype(str).fillna("na").values

    ct_matrix = _ct_reference_matrix(feature_cols, clinical)
    print(f"[ct] auxiliary feature rows: {len(ct_matrix)}")

    results = []
    for name, fn in VARIANTS.items():
        print(f"[run] {name} ...")
        res = evaluate_variant(name, fn, engineered, feature_cols, y_df, groups, ct_matrix)
        results.append(res)
        ci = res["avg_mae_ci95"]
        print(f"      avg MAE={res['avg_mae_mm']:.3f} (95% CI {ci[0]:.2f}-{ci[1]:.2f}) | "
              f"Z={res['z_avg_mae_mm']:.3f} | X={res['axis_mae_mm']['x']:.2f} Y={res['axis_mae_mm']['y']:.2f}")

    results.sort(key=lambda r: r["avg_mae_mm"])
    report = {
        "run_id": f"experiment_matrix_{date.today().strftime('%Y%m%d')}",
        "eval": f"GroupKFold({N_SPLITS}) by patient + bootstrap CI (n={N_BOOTSTRAP})",
        "clinical_patients": len(clinical),
        "features_used": len(feature_cols),
        "dropped_leaky_features": dropped_leaky,
        "ct_aux_rows": len(ct_matrix),
        "ct_enrichment": True,
        "z_axis_notes": (
            "V6: GBT on clinical drivers; V7: QuantileRegressor median on drivers+proj_sup Z "
            "(robust small-n, conformal/uncertainty literature)."
        ),
        "results": results,
        "ranking": {
            "best_avg": results[0]["variant"],
            "best_z": min(results, key=lambda r: r["z_avg_mae_mm"])["variant"],
        },
    }
    out_dir = ROOT / "results" / "validation_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_matrix_gkf5.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Experiment matrix (GroupKFold-5, honest, lower=better) ===")
    print(f"{'variant':28s} {'avgMAE':>8s} {'95%CI':>16s} {'Z':>7s} {'Y':>6s} {'X':>6s}")
    for r in results:
        ci = r["avg_mae_ci95"]
        print(f"{r['variant']:28s} {r['avg_mae_mm']:8.3f} "
              f"{ci[0]:6.2f}-{ci[1]:5.2f}  {r['z_avg_mae_mm']:7.2f} "
              f"{r['axis_mae_mm']['y']:6.2f} {r['axis_mae_mm']['x']:6.2f}")
    print(f"\nBest avg: {report['ranking']['best_avg']} | Best Z: {report['ranking']['best_z']}")
    print(f"Report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
