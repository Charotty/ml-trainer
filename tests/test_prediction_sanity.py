"""Sanity checks for displacement prediction magnitudes."""

from src.api.cases.predictor import assess_prediction_sanity


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
