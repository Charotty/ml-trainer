"""CT geometry replaces sheet geometry; Excel deltas stay."""

from __future__ import annotations

import pandas as pd

from src.data.ct_feature_table import build_ct_labeled_frame


def test_join_copies_ct_geometry_and_keeps_deltas():
    labels = pd.DataFrame(
        [
            {
                "full_name": "Иванов",
                "kidney_left_center_x_rel": 1.0,
                "kidney_left_supine_middle_z": -50.0,
                "kidney_left_delta_z": -8.0,
            },
            {
                "full_name": "Петров",
                "kidney_left_center_x_rel": 2.0,
                "kidney_left_supine_middle_z": -40.0,
                "kidney_left_delta_z": 3.0,
            },
            {
                "full_name": "Дубль",
                "kidney_left_center_x_rel": 0.0,
                "kidney_left_delta_z": 1.0,
            },
            {
                "full_name": "Дубль",
                "kidney_left_center_x_rel": 0.0,
                "kidney_left_delta_z": 1.0,
            },
        ]
    )
    ct = pd.DataFrame(
        [
            {"case_id": "Иванов", "kidney_left_center_x_rel": -70.0, "kidney_left_supine_middle_z": 12.0},
            {"case_id": "Сидоров", "kidney_left_center_x_rel": 5.0},
        ]
    )
    frame, counts = build_ct_labeled_frame(labels, ct)
    assert counts["paired_rows"] == 1
    assert counts["label_duplicate_surnames"] == 2
    assert frame.iloc[0]["kidney_left_center_x_rel"] == -70.0
    assert frame.iloc[0]["kidney_left_supine_middle_z"] == 12.0
    assert frame.iloc[0]["kidney_left_delta_z"] == -8.0
