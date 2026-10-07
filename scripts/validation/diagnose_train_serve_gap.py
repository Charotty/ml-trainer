#!/usr/bin/env python3
"""Diagnose why train-time accuracy does not transfer to uploaded patients.

Compares the persisted artifact's feature contract against the training table
(``data/vybor_from_xlsx.csv``) and against real Workbench cases:

  * dead features (near-constant in train -> zeroed by the inference guard)
  * imputed features (NaN at serve time -> model sees a train constant)
  * out-of-distribution features (outside train min/max)
  * per-target reliance on features that are dead/imputed/OOD at serve time
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.pipeline import apply_model_preprocessing  # noqa: E402
from src.models.runtime import default_model_path  # noqa: E402

TRAIN_CSV = REPO_ROOT / "data" / "vybor_from_xlsx.csv"
NEAR_CONSTANT_SCALE = 1e-6


def _case_matrix(case_id: str, feature_names: List[str]) -> np.ndarray | None:
    path = REPO_ROOT / "data" / "cases" / case_id / "artifacts" / "features.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    all_features = doc.get("all_features") or {}
    row = [all_features.get(name) for name in feature_names]
    return np.array([[np.nan if v is None else float(v) for v in row]], dtype=float)


def _train_matrix(payload: Mapping[str, Any], feature_names: List[str]) -> np.ndarray:
    from src.models.ensemble.features import inference_transformer_from_payload

    df = pd.read_csv(TRAIN_CSV)
    transformer = inference_transformer_from_payload(payload)
    return transformer.build_inference_matrix(df)


def _laterality_audit(payload: Mapping[str, Any], feature_names: List[str]) -> None:
    """How one-kidney patients are encoded in train vs what serve can produce."""
    from src.features.phase1_schema import (
        has_complete_left_targets,
        has_complete_right_targets,
    )

    df = pd.read_csv(TRAIN_CSV)
    left = has_complete_left_targets(df)
    right = has_complete_right_targets(df)
    one_sided = df.loc[left ^ right]

    print("\n-- laterality contract --")
    print(f"  train rows            : {len(df)}")
    print(f"  both sides labelled   : {int((left & right).sum())}")
    print(f"  one-sided             : {len(one_sided)} "
          f"(left-only {int((left & ~right).sum())}, right-only {int((right & ~left).sum())})")

    present_x_rel = []
    for _, row in one_sided.iterrows():
        side = "left" if pd.notna(row.get("kidney_left_delta_x")) else "right"
        val = row.get(f"kidney_{side}_center_x_rel")
        if pd.notna(val):
            present_x_rel.append(float(val))
    zeros = sum(1 for v in present_x_rel if abs(v) < 1e-9)
    both_x_rel = df.loc[left & right, "kidney_left_center_x_rel"].dropna()
    both_zeros = int((both_x_rel.abs() < 1e-9).sum())
    print(f"  one-sided rows encoding the present kidney at x_rel == 0.0 exactly: "
          f"{zeros}/{len(present_x_rel)}")
    print(f"  two-kidney rows with x_rel == 0.0                                 : "
          f"{both_zeros}/{len(both_x_rel)}")
    print("  -> x_rel == 0.0 is a de-facto solitary-kidney marker in train.")

    stats = np.asarray(payload["imputer"].statistics_, dtype=float)
    print("\n  serve cannot reproduce it; a missing kidney is imputed with:")
    for col in (
        "kidney_left_center_x_rel", "kidney_right_center_x_rel",
        "kidney_left_volume_cm3", "kidney_distance_lr", "volume_asymmetry",
    ):
        if col in feature_names:
            print(f"    {col:34} -> {stats[feature_names.index(col)]:.4f}")


def _importances(model: Any, n_features: int) -> np.ndarray | None:
    for attr in ("feature_importances_", "coef_"):
        value = getattr(model, attr, None)
        if value is None:
            continue
        arr = np.abs(np.asarray(value, dtype=float)).reshape(-1)
        if arr.shape[0] == n_features:
            return arr / arr.sum() if arr.sum() > 0 else arr
    estimators = getattr(model, "estimators_", None) or getattr(model, "named_estimators_", None)
    if isinstance(estimators, dict):
        estimators = list(estimators.values())
    if estimators:
        parts = [_importances(est, n_features) for est in estimators]
        parts = [p for p in parts if p is not None]
        if parts:
            return np.mean(parts, axis=0)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--case", action="append", default=[], metavar="LABEL=CASE_ID")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    model_path = args.model or default_model_path()
    payload = joblib.load(model_path)
    feature_names: List[str] = list(payload["feature_names"])
    scaler = payload["scaler"]
    imputer = payload.get("imputer")
    n = len(feature_names)

    print(f"model      : {model_path.name}")
    print(f"features   : {n}")
    print(f"targets    : {list(payload['models'].keys())}")

    print("\n-- artifact contract --")
    print(f"  categorical_encoder persisted : {'categorical_encoder' in payload}")
    print(f"  z_head                        : {payload.get('z_head')}")
    print(f"  z_driver_names                : {payload.get('z_driver_names')}")
    print(f"  left_z_calibrator             : {payload.get('left_z_calibrator')}")
    print(f"  right_z_calibrator            : {payload.get('right_z_calibrator')}")
    mirrors = [c for c in feature_names if c.startswith(("na_sup_z_", "na_sup_pct_"))]
    sources = {c.split("_", 3)[-1] for c in mirrors}
    print(f"  na_sup_* affine mirrors       : {len(mirrors)} of {len(sources)} core columns")
    print(f"  na_pop_shift_* cohort consts  : "
          f"{sum(1 for c in feature_names if c.startswith('na_pop_shift_'))}")

    _laterality_audit(payload, feature_names)

    scale = np.asarray(getattr(scaler, "scale_", np.ones(n)), dtype=float)
    dead_mask = ~np.isfinite(scale) | (np.abs(scale) < NEAR_CONSTANT_SCALE)
    print(f"\ndead (near-constant in train, forced to 0 at serve): {int(dead_mask.sum())}/{n}")
    for name in np.array(feature_names)[dead_mask][: args.top]:
        print(f"  {name}")

    X_train = _train_matrix(payload, feature_names)
    train_nan = np.isnan(X_train)
    print(f"\ntrain rows : {X_train.shape[0]}  (n features {X_train.shape[1]})")
    print(f"train NaN cells: {int(train_nan.sum())} "
          f"({100.0 * train_nan.mean():.1f}%), fully-NaN columns: "
          f"{int(train_nan.all(axis=0).sum())}")

    lo = np.nanmin(X_train, axis=0)
    hi = np.nanmax(X_train, axis=0)
    train_present = ~train_nan.all(axis=0)

    imp_by_target: Dict[str, np.ndarray] = {}
    for target, model in payload["models"].items():
        imp = _importances(model, n)
        if imp is not None:
            imp_by_target[target] = imp

    for spec in args.case:
        label, _, case_id = spec.partition("=")
        X_case = _case_matrix(case_id, feature_names)
        if X_case is None:
            print(f"\n[{label}] features.json not found for case {case_id}")
            continue
        nan_mask = np.isnan(X_case[0])
        with np.errstate(invalid="ignore"):
            ood_mask = train_present & ~nan_mask & ((X_case[0] < lo) | (X_case[0] > hi))
        usable = ~(dead_mask | nan_mask | ood_mask)

        print(f"\n=== {label} (case {case_id}) ===")
        print(f"  NaN at serve (imputed with a train constant): {int(nan_mask.sum())}/{n}")
        print(f"  outside train min/max (extrapolation)       : {int(ood_mask.sum())}/{n}")
        print(f"  dead                                        : {int(dead_mask.sum())}/{n}")
        print(f"  actually informative                        : {int(usable.sum())}/{n}"
              f"  ({100.0 * usable.mean():.0f}%)")

        X_scaled = apply_model_preprocessing(
            X_case.copy(), {"imputer": imputer, "scaler": scaler}
        )
        clipped = int(np.sum(np.abs(X_scaled[0]) >= 12.0 - 1e-9))
        print(f"  clipped at |z|=12 (hard extrapolation guard) : {clipped}")

        for target, imp in imp_by_target.items():
            share_nan = float(imp[nan_mask].sum())
            share_ood = float(imp[ood_mask].sum())
            share_dead = float(imp[dead_mask].sum())
            print(
                f"    {target:24} importance on  NaN {share_nan:5.1%} | "
                f"OOD {share_ood:5.1%} | dead {share_dead:5.1%} | "
                f"real {1 - share_nan - share_ood - share_dead:5.1%}"
            )

        top_idx = np.argsort(-np.nan_to_num(np.mean(list(imp_by_target.values()), axis=0)))[: args.top]
        print(f"  top-{args.top} features by mean importance:")
        for i in top_idx:
            flag = (
                "DEAD" if dead_mask[i] else
                "NaN " if nan_mask[i] else
                "OOD " if ood_mask[i] else
                "ok  "
            )
            train_range = f"[{lo[i]:.2f}, {hi[i]:.2f}]" if train_present[i] else "[all-NaN]"
            value = X_case[0, i]
            print(f"    {flag} {feature_names[i]:38} serve={value!s:>12.12} train={train_range}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
