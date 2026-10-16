"""Matched CT rows are scored; every row still feeds the cohort."""

from __future__ import annotations

import pandas as pd

from src.data.ct_profile_bank import (
    LATERAL_COLUMNS,
    agreement_rows,
    attach_predictors,
    bank_from_frames,
    indicators_from_row,
    load_profile,
    match_counts,
)


def _sheet() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fio": "Иванов",
                "left_lateral_middle_x": 60.0,
                "left_lateral_middle_y": 40.0,
                "left_lateral_middle_z": -60.0,
                "right_lateral_middle_x": 60.0,
                "right_lateral_middle_y": 40.0,
                "right_lateral_middle_z": -60.0,
                "left_kidney_volume_cm3": 170.0,
                "right_kidney_volume_cm3": 160.0,
                "abd_width_l3l4_mm": 340.0,
                "abd_depth_l3l4_mm": 230.0,
            },
            {"fio": "Петров", "left_lateral_middle_z": -10.0, "left_kidney_volume_cm3": 150.0},
        ]
    )


def _profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    supine = load_profile(
        pd.DataFrame(
            [
                {"case_id": "Иванов", "kidney_left_volume_cm3": 140.0, "kidney_left_middle_z_vert": -20.0},
                {"case_id": "Сидоров", "kidney_left_volume_cm3": 180.0, "kidney_left_middle_z_vert": -10.0},
                {"case_id": "Орлов", "kidney_left_volume_cm3": 200.0, "kidney_left_middle_z_vert": 0.0},
            ]
        )
    )
    lateral = load_profile(
        pd.DataFrame(
            [
                {
                    "case_id": "Иванов",
                    "kidney_left_volume_cm3": 150.0,
                    "kidney_left_middle_x_vert": 70.0,
                    "kidney_left_middle_y_vert": 5.0,
                    "kidney_left_middle_z_vert": 0.0,
                    "kidney_right_middle_x_vert": 70.0,
                    "kidney_right_middle_y_vert": 5.0,
                    "kidney_right_middle_z_vert": 0.0,
                    "body_width_mm": 100.0,
                    "body_depth_mm": 120.0,
                },
                {"case_id": "Козлов", "kidney_left_volume_cm3": 190.0, "kidney_left_middle_z_vert": 5.0},
            ]
        )
    )
    return supine, lateral


def test_match_counts_and_agreement():
    supine, lateral = _profiles()
    counts = match_counts(_sheet(), supine, lateral)
    assert counts["matched_supine"] == 1
    assert counts["matched_lateral"] == 1
    assert counts["supine_only"] == 2
    rows = agreement_rows(_sheet(), lateral, "lateral")
    volume = next(row for row in rows if row["metric"] == "lateral_left_volume_cm3")
    assert volume["n"] == 1
    assert volume["bias"] == -20.0
    middle_z = next(row for row in rows if row["metric"] == "lateral_left_middle_z_vert")
    assert middle_z["bias"] == 60.0


def test_unmatched_rows_still_set_the_cohort_and_lateral_xyz_is_not_a_predictor():
    supine, lateral = _profiles()
    bank = bank_from_frames(supine, lateral)
    assert bank["supine_stats"]["kidney_left_volume_cm3"]["n"] == 3
    assert "kidney_left_middle_z_vert" not in LATERAL_COLUMNS
    attached = attach_predictors(
        _sheet(),
        supine,
        lateral,
        bank["supine_stats"],
        bank["lateral_stats"],
    )
    ivanov = attached.iloc[0]
    petrov = attached.iloc[1]
    assert ivanov["ct_sup_kidney_left_volume_cm3"] == 140.0
    assert pd.isna(petrov["ct_sup_kidney_left_volume_cm3"])
    assert "ct_lat_kidney_left_middle_z_vert" not in attached.columns
    assert ivanov["ct_lat_kidney_left_volume_cm3"] == 150.0
    live = indicators_from_row({"kidney_left_volume_cm3": 180.0}, bank["supine_stats"])
    assert live["ct_sup_z_kidney_left_volume_cm3"] == 0.0
