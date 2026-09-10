"""Sanity checks for displacement prediction magnitudes."""

import pytest

from src.api.cases.predictor import assess_prediction_sanity, imputed_importance_share


def test_assess_prediction_sanity_flags_explosion():
    ok, warnings = assess_prediction_sanity(
        {
            "kidney_left_delta_x": 1e13,
            "kidney_left_delta_y": 2.0,
            "kidney_left_delta_z": 3.0,
            "kidney_right_delta_x": 4.0,
            "kidney_right_delta_y": 5.0,
            "kidney_right_delta_z": 6.0,
        }
    )
    assert ok is False
    assert any("kidney_left_delta_x" in w for w in warnings)


def test_assess_prediction_sanity_accepts_clinical_range():
    ok, warnings = assess_prediction_sanity(
        {
            "kidney_left_delta_x": -8.5,
            "kidney_left_delta_y": 31.9,
            "kidney_left_delta_z": -7.6,
            "kidney_right_delta_x": -4.7,
            "kidney_right_delta_y": 4.6,
            "kidney_right_delta_z": 29.5,
        }
    )
    assert ok is True
    assert warnings == []


def test_assess_prediction_sanity_skips_withheld_none():
    ok, warnings = assess_prediction_sanity(
        {
            "kidney_left_delta_x": None,
            "kidney_left_delta_y": None,
            "kidney_left_delta_z": None,
            "kidney_right_delta_x": -4.7,
            "kidney_right_delta_y": 4.6,
            "kidney_right_delta_z": 7.0,
        }
    )
    assert ok is True
    assert warnings == []


class _ImpModel:
    feature_importances_ = None

    def __init__(self, imp):
        self.feature_importances_ = imp


def test_imputed_importance_share_flags_high_nan_load():
    import numpy as np

    names = ["a", "b", "c"]
    model = _ImpModel(np.array([0.5, 0.4, 0.1]))
    payload = {"feature_names": names, "models": {"kidney_left_delta_x": model}}
    all_features = {"a": None, "b": None, "c": 1.0}
    share = imputed_importance_share(
        payload=payload, all_features=all_features, target="kidney_left_delta_x"
    )
    assert share == pytest.approx(0.9)
    ok, warnings = assess_prediction_sanity(
        {"kidney_left_delta_x": 3.0},
        all_features=all_features,
        payload=payload,
        imputed_share_threshold=0.40,
    )
    assert ok is False
    assert any("импьютированы" in w for w in warnings)
