"""Numerical-safety tests for persisted model preprocessing."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.features.pipeline import apply_model_preprocessing


def test_preprocessing_neutralizes_near_constant_scaler_columns() -> None:
    """A real CT value must not become a 1e16 linear-model input."""
    scaler = StandardScaler().fit(np.array([[0.0, 100.0], [0.0, 102.0]]))
    # Mirrors legacy model artifacts where a mathematically constant feature
    # retained floating-point noise instead of an exact unit scale.
    scaler.scale_[0] = 7.4e-15

    scaled = apply_model_preprocessing(
        np.array([[-79.59, 104.0]]),
        {"scaler": scaler},
    )

    assert scaled.shape == (1, 2)
    assert scaled[0, 0] == 0.0
    assert np.isfinite(scaled).all()


def test_preprocessing_caps_extreme_standardized_values() -> None:
    scaler = StandardScaler().fit(np.array([[0.0], [2.0]]))

    scaled = apply_model_preprocessing(
        np.array([[1000.0]]),
        {"scaler": scaler},
    )

    assert scaled[0, 0] == 12.0
