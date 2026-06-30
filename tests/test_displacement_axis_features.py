"""Tests for Y/Z displacement axis feature engineering."""

import pandas as pd

from src.features.displacement_axis_features import (
    DISPLACEMENT_AXIS_FEATURES,
    add_displacement_axis_features,
)


def test_adds_z_y_ratio_features():
    df = pd.DataFrame(
        [{
            "kidney_left_center_z_rel": 10.0,
            "kidney_right_center_z_rel": 5.0,
            "kidney_left_center_y_rel": 3.0,
            "kidney_right_center_y_rel": 1.0,
            "body_depth_mm": 100.0,
            "body_width_mm": 200.0,
            "lumbar_lordosis_deg": 40.0,
            "abd_wall_thickness_mm": 20.0,
        }]
    )
    out = add_displacement_axis_features(df)
    assert out["kidney_z_asymmetry_rel"].iloc[0] == 5.0
    assert out["kidney_left_z_over_depth"].iloc[0] == 0.1
    assert out["body_sagittal_index"].iloc[0] == 0.5
    for col in DISPLACEMENT_AXIS_FEATURES:
        if col in (
            "kidney_left_z_span_supine_mm",
            "kidney_right_z_span_supine_mm",
            "kidney_left_y_span_supine_mm",
            "kidney_right_y_span_supine_mm",
            "kidney_left_z_delta_span_mm",
            "kidney_right_z_delta_span_mm",
            "kidney_left_y_delta_span_mm",
            "kidney_right_y_delta_span_mm",
        ):
            continue
        assert col in out.columns
