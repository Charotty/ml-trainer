"""V7 Z-head: median quantile regression on clinical drivers (experiment matrix winner)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import QuantileRegressor

Z_TARGETS = ("kidney_left_delta_z", "kidney_right_delta_z")

CLINICAL_DRIVERS = [
    "bmi",
    "body_type",
    "age",
    "sex",
    "lumbar_lordosis_deg",
    "s1_plate_tilt_deg",
    "abd_wall_thickness_mm",
    "body_depth_mm",
    "body_width_mm",
    "body_sagittal_index",
    "lordosis_x_depth",
    "abd_wall_over_depth",
    "kidney_left_z_span_supine_mm",
    "kidney_right_z_span_supine_mm",
    "kidney_left_center_z_rel",
    "kidney_right_center_z_rel",
    "kidney_z_asymmetry_rel",
    "kidney_left_z_over_depth",
    "kidney_right_z_over_depth",
]

Z_SAFE_EXTRA = [
    "proj_sup_kidney_left_center_z_rel",
    "proj_sup_kidney_right_center_z_rel",
    "proj_sup_body_depth_mm",
    "kidney_lr_sep_z",
]


def resolve_z_driver_names(feature_names: Sequence[str]) -> list[str]:
    """Pick available V7 driver columns in stable order."""
    candidates = CLINICAL_DRIVERS + Z_SAFE_EXTRA
    name_set = set(feature_names)
    return [c for c in candidates if c in name_set]


def driver_indices(feature_names: Sequence[str], driver_names: Sequence[str]) -> list[int]:
    idx = {n: i for i, n in enumerate(feature_names)}
    return [idx[n] for n in driver_names if n in idx]


def extract_driver_matrix(
    X: np.ndarray,
    feature_names: Sequence[str],
    driver_names: Sequence[str],
) -> np.ndarray:
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature matrix width {X.shape[1]} != len(feature_names) {len(feature_names)}"
        )
    cols = driver_indices(feature_names, driver_names)
    if not cols:
        raise ValueError("No V7 Z driver columns found in feature matrix")
    return X[:, cols]


def build_quantile_z_model() -> QuantileRegressor:
    return QuantileRegressor(quantile=0.5, alpha=0.1, solver="highs")


def fit_quantile_z(
    X_imputed: np.ndarray,
    feature_names: Sequence[str],
    y: np.ndarray,
    driver_names: Sequence[str] | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[QuantileRegressor, list[str]]:
    drivers = list(driver_names) if driver_names else resolve_z_driver_names(feature_names)
    if not drivers:
        raise ValueError("V7 Z drivers unavailable in feature set")
    model = build_quantile_z_model()
    X_z = extract_driver_matrix(X_imputed, feature_names, drivers)
    if sample_weight is not None:
        model.fit(X_z, y, sample_weight=sample_weight)
    else:
        model.fit(X_z, y)
    return model, drivers


def predict_quantile_z(
    model: QuantileRegressor,
    X_imputed: np.ndarray,
    feature_names: Sequence[str],
    driver_names: Sequence[str],
) -> np.ndarray:
    X_z = extract_driver_matrix(X_imputed, feature_names, driver_names)
    return model.predict(X_z)
