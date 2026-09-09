#!/usr/bin/env python3
"""Shared utilities for WSL validation workflows.

Prediction goes through ``src.models.runtime``. This module keeps dataset
helpers and the optional RF baseline used only when ``model_path is None``.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.phase1_schema import (  # noqa: E402
    BASE_FEATURES,
    TARGET_NAMES,
    normalize_dataframe,
)
from src.models.runtime import (  # noqa: E402
    DEFAULT_MODEL_PATH_STR,
    LEGACY_MODEL_NAME,
    PredictorBundle,
    bundle_from_payload,
    load_model_bundle,
    predict_df,
    warn_if_legacy_model,
)

TARGET_COLUMNS: List[str] = list(TARGET_NAMES)
DEFAULT_MODEL_PATH = ROOT / DEFAULT_MODEL_PATH_STR


def ensure_run_dirs(base_output_dir: Path, run_id: str) -> Path:
    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    return run_dir


def load_dataset(
    dataset_path: Path,
    *,
    source_filter: Optional[str] = None,
) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    df = normalize_dataframe(pd.read_csv(dataset_path))
    if source_filter and "source" in df.columns:
        sources = {s.strip() for s in source_filter.split(",") if s.strip()}
        df = df[df["source"].isin(sources)].copy()
        if len(df) == 0:
            raise ValueError(f"No rows left after source filter {sorted(sources)}")
    missing = [c for c in BASE_FEATURES + TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns for validation: "
            + ", ".join(missing)
        )
    return df


def load_ct_features(path: Path) -> pd.DataFrame:
    """Load a CT feature table for inference-only workflows (targets optional)."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return normalize_dataframe(pd.read_csv(path, low_memory=False))


def build_or_load_predictor(
    df: pd.DataFrame,
    model_path: Path | None,
    test_size: float,
    seed: int,
    *,
    holdout_eval: bool = False,
) -> Tuple[PredictorBundle, pd.DataFrame, pd.DataFrame]:
    if holdout_eval:
        train_df = df.iloc[0:0].copy()
        eval_df = df.copy()
    else:
        train_df, eval_df = train_test_split(df, test_size=test_size, random_state=seed)

    if model_path is not None:
        bundle = load_model_bundle(Path(model_path))
        return bundle, train_df, eval_df

    feature_names = [c for c in BASE_FEATURES if c in df.columns]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_names].values)
    y_train = train_df[TARGET_COLUMNS]
    models: Dict[str, object] = {}
    for target in TARGET_COLUMNS:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train[target].values)
        models[target] = model
    bundle = PredictorBundle(
        mode="fallback_random_forest",
        feature_names=feature_names,
        target_names=TARGET_COLUMNS,
        scaler=scaler,
        models=models,
    )
    return bundle, train_df, eval_df


def vector_norm(left_xyz: np.ndarray, right_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    left = np.linalg.norm(left_xyz, axis=1)
    right = np.linalg.norm(right_xyz, axis=1)
    return left, right


def compute_regression_table(
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    targets: List[str],
) -> pd.DataFrame:
    metrics_rows = []
    for target in targets:
        y_true = truth[target].to_numpy()
        y_pred = pred[target].to_numpy()
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics_rows.append(
            {
                "target": target,
                "mae_mm": float(mean_absolute_error(y_true, y_pred)),
                "rmse_mm": rmse,
                "r2": float(r2_score(y_true, y_pred)),
                "count": len(y_true),
            }
        )
    return pd.DataFrame(metrics_rows)


def save_manifest(
    run_dir: Path,
    *,
    run_id: str,
    dataset_path: Path,
    model_path: Path | None,
    predictor_mode: str,
    train_count: int,
    eval_count: int,
    seed: int,
    source_filter: Optional[str] = None,
    holdout_eval: bool = False,
) -> None:
    manifest = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "model_path": str(model_path) if model_path else None,
        "predictor_mode": predictor_mode,
        "train_count": train_count,
        "eval_count": eval_count,
        "seed": seed,
        "source_filter": source_filter,
        "holdout_eval": holdout_eval,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_MODEL_PATH_STR",
    "LEGACY_MODEL_NAME",
    "PredictorBundle",
    "TARGET_COLUMNS",
    "build_or_load_predictor",
    "bundle_from_payload",
    "compute_regression_table",
    "ensure_run_dirs",
    "load_ct_features",
    "load_dataset",
    "load_model_bundle",
    "predict_df",
    "save_manifest",
    "vector_norm",
    "warn_if_legacy_model",
]
