"""Tests for V7 quantile Z-head."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.z_quantile_v7 import (
    Z_SAFE_EXTRA,
    build_quantile_z_model,
    fit_quantile_z,
    predict_quantile_z,
    resolve_z_driver_names,
)


def test_resolve_z_driver_names_order():
    names = ["noise", *Z_SAFE_EXTRA, "bmi", "age"]
    drivers = resolve_z_driver_names(names)
    assert "bmi" in drivers
    assert drivers.index("bmi") < drivers.index("age")
    assert all(d in names for d in drivers)


def test_fit_predict_quantile_z():
    feature_names = ["bmi", "body_depth_mm", "kidney_left_center_z_rel", "extra"]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, len(feature_names)))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.5, size=20)
    model, drivers = fit_quantile_z(X, feature_names, y)
    pred = predict_quantile_z(model, X[:3], feature_names, drivers)
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))


def test_build_quantile_z_model():
    m = build_quantile_z_model()
    assert m.quantile == 0.5
