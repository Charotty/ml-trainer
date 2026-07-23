#!/usr/bin/env python3
"""Compute validation metrics and export reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from common import (
    DEFAULT_MODEL_PATH_STR,
    TARGET_COLUMNS,
    build_or_load_predictor,
    compute_regression_table,
    ensure_run_dirs,
    load_dataset,
    predict_df,
    save_manifest,
    vector_norm,
)

ArrayLike = Union[np.ndarray, pd.DataFrame, Mapping[str, Sequence[float]]]


def compute_clinical_within_ratios(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    target_columns: Sequence[str] | None = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Clinical within_* ratios from per-patient vector_error_mean_mm.

    ``within_5mm_ratio`` / ``within_10mm_ratio`` are the fraction of patients
    whose mean left/right L2-norm error is <= 5 / 10 mm.

    Pointwise (axis-wise) rates are retained under distinct names for diagnostics.
    """
    cols = list(target_columns) if target_columns is not None else list(TARGET_COLUMNS)
    true_df = _as_target_frame(y_true, cols)
    pred_df = _as_target_frame(y_pred, cols)

    left_true, right_true = vector_norm(
        true_df[["kidney_left_delta_x", "kidney_left_delta_y", "kidney_left_delta_z"]].to_numpy(),
        true_df[["kidney_right_delta_x", "kidney_right_delta_y", "kidney_right_delta_z"]].to_numpy(),
    )
    left_pred, right_pred = vector_norm(
        pred_df[["kidney_left_delta_x", "kidney_left_delta_y", "kidney_left_delta_z"]].to_numpy(),
        pred_df[["kidney_right_delta_x", "kidney_right_delta_y", "kidney_right_delta_z"]].to_numpy(),
    )
    vector_error_left = np.abs(left_true - left_pred)
    vector_error_right = np.abs(right_true - right_pred)
    vector_error_mean = (vector_error_left + vector_error_right) / 2.0

    pointwise_abs = np.abs(true_df[cols].to_numpy() - pred_df[cols].to_numpy())

    summary = {
        "vector_error_left_mae_mm": float(vector_error_left.mean()),
        "vector_error_right_mae_mm": float(vector_error_right.mean()),
        "vector_error_mean_mae_mm": float(vector_error_mean.mean()),
        "within_5mm_ratio": float((vector_error_mean <= 5.0).mean()),
        "within_10mm_ratio": float((vector_error_mean <= 10.0).mean()),
        "within_5mm_pointwise_ratio": float((pointwise_abs <= 5.0).mean()),
        "within_10mm_pointwise_ratio": float((pointwise_abs <= 10.0).mean()),
        "sample_count": float(len(true_df)),
    }
    per_patient = {
        "vector_error_left_mm": vector_error_left,
        "vector_error_right_mm": vector_error_right,
        "vector_error_mean_mm": vector_error_mean,
    }
    return summary, per_patient


def _as_target_frame(data: ArrayLike, cols: Sequence[str]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.loc[:, list(cols)].copy()
    if isinstance(data, Mapping):
        return pd.DataFrame({c: data[c] for c in cols})
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(cols):
        raise ValueError(f"Expected array shape (n, {len(cols)}), got {arr.shape}")
    return pd.DataFrame(arr, columns=list(cols))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/vybor_unified_features.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH_STR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="results/validation_runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--source",
        default=None,
        help="Comma-separated source filter (e.g. Vybor or Vybor,Excel)",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Evaluate entire dataset without re-splitting (honest holdout CSV)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = ensure_run_dirs(Path(args.out_dir), args.run_id)
    dataset_path = Path(args.dataset)
    model_path = Path(args.model) if args.model else None

    df = load_dataset(dataset_path, source_filter=args.source)
    bundle, train_df, eval_df = build_or_load_predictor(
        df=df,
        model_path=model_path,
        test_size=args.test_size,
        seed=args.seed,
        holdout_eval=args.holdout,
    )
    pred_df = predict_df(bundle, eval_df)
    y_true = eval_df[TARGET_COLUMNS].copy()

    per_target = compute_regression_table(y_true, pred_df, TARGET_COLUMNS)
    per_target.to_csv(run_dir / "metrics" / "metrics_per_target.csv", index=False)

    clinical, per_patient = compute_clinical_within_ratios(y_true, pred_df, TARGET_COLUMNS)

    summary = pd.DataFrame(
        [
            {"metric": "mae_avg_mm", "value": float(per_target["mae_mm"].mean())},
            {"metric": "rmse_avg_mm", "value": float(per_target["rmse_mm"].mean())},
            {"metric": "r2_avg", "value": float(per_target["r2"].mean())},
            {"metric": "vector_error_left_mae_mm", "value": clinical["vector_error_left_mae_mm"]},
            {"metric": "vector_error_right_mae_mm", "value": clinical["vector_error_right_mae_mm"]},
            {"metric": "vector_error_mean_mae_mm", "value": clinical["vector_error_mean_mae_mm"]},
            {"metric": "within_5mm_ratio", "value": clinical["within_5mm_ratio"]},
            {"metric": "within_10mm_ratio", "value": clinical["within_10mm_ratio"]},
            {
                "metric": "within_5mm_pointwise_ratio",
                "value": clinical["within_5mm_pointwise_ratio"],
            },
            {
                "metric": "within_10mm_pointwise_ratio",
                "value": clinical["within_10mm_pointwise_ratio"],
            },
            {"metric": "sample_count", "value": clinical["sample_count"]},
        ]
    )
    summary.to_csv(run_dir / "metrics" / "metrics_summary.csv", index=False)

    worst = eval_df[["case_id"]].copy() if "case_id" in eval_df.columns else pd.DataFrame(index=eval_df.index)
    worst["vector_error_left_mm"] = per_patient["vector_error_left_mm"]
    worst["vector_error_right_mm"] = per_patient["vector_error_right_mm"]
    worst["vector_error_mean_mm"] = per_patient["vector_error_mean_mm"]
    worst = worst.sort_values("vector_error_mean_mm", ascending=False).head(args.top_n)
    worst.to_csv(run_dir / "metrics" / "worst_cases.csv", index=True)

    pred_export = pred_df.copy()
    pred_export.columns = [f"pred_{c}" for c in pred_export.columns]
    joined = pd.concat([eval_df.reset_index(drop=True), pred_export.reset_index(drop=True)], axis=1)
    joined.to_csv(run_dir / "predictions" / "evaluation_predictions.csv", index=False)

    save_manifest(
        run_dir,
        run_id=args.run_id,
        dataset_path=dataset_path,
        model_path=model_path,
        predictor_mode=bundle.mode,
        train_count=len(train_df),
        eval_count=len(eval_df),
        seed=args.seed,
        source_filter=args.source,
        holdout_eval=args.holdout,
    )

    print(f"[OK] Metrics written to: {run_dir / 'metrics'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
