"""Tests for na_spine / na_boku cohort trend features."""

from __future__ import annotations

import pandas as pd

from src.features.na_trend_features import NaTrendStore, attach_na_trend_features
from src.features.phase1_schema import TARGET_NAMES


def test_na_trend_store_fit_and_attach():
    store = NaTrendStore.fit(include_kits=False)
    assert store.spine_rows > 0
    assert store.boku_rows > 0
    assert not store.include_kits
    assert len(store.population_shift) == 6
    assert all(k.startswith("na_pop_shift_") for k in store.population_shift)

    clinical = pd.DataFrame(
        [
            {
                "full_name": "test",
                "kidney_left_center_x_rel": 10.0,
                "kidney_left_center_y_rel": 20.0,
                "kidney_left_center_z_rel": 30.0,
                "kidney_right_center_x_rel": 11.0,
                "kidney_right_center_y_rel": 21.0,
                "kidney_right_center_z_rel": 31.0,
                **{t: 1.0 for t in TARGET_NAMES},
            }
        ]
    )
    out = attach_na_trend_features(clinical, store)
    assert "na_pop_shift_left_x" in out.columns
    assert "na_sup_z_kidney_left_center_x_rel" in out.columns
    assert out["na_pop_shift_left_x"].nunique() == 1


def test_na_trend_kits_optional():
    store = NaTrendStore.fit(include_kits=True)
    assert store.kits_rows > 0
    assert any(k.startswith("kits_cohort_median_") for k in store.kits_delta_medians)


def test_na_trend_no_per_patient_lateral_join():
    store = NaTrendStore.fit()
    names = store.trend_feature_names()
    assert not any(n.startswith("proj_lat_") for n in names)
    assert not any(n.startswith("proj_sup_") for n in names)
