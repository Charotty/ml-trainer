"""Variant A trainer profile / nested seed smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES
from src.models.ensemble import AdaptiveEnsembleTrainer
from src.models.ensemble.estimators import make_base_models
from src.models.nested_cv import (
    evaluate_nested_groupkfold_oof,
    evaluate_repeated_nested_cv,
    remap_groups_for_seed,
)


def _synthetic(n: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {c: float(100 + rng.normal(0, 2)) for c in BASE_FEATURES}
        row["full_name"] = f"P{i:02d}"
        for t in TARGET_NAMES:
            row[t] = float(rng.normal(0, 3))
        rows.append(row)
    return pd.DataFrame(rows)


def _tiny_factory(**kwargs):
    params = {
        "enrichment_mode": "none",
        "estimator_profile": "tiny",
        "inner_n_splits": 2,
        "yz_target_boost": False,
    }
    params.update(kwargs)
    return AdaptiveEnsembleTrainer(**params)


def test_remap_groups_for_seed_changes_order():
    groups = np.array(["A", "B", "C", "A", "B", "C"])
    g0 = remap_groups_for_seed(groups, 0)
    g1 = remap_groups_for_seed(groups, 1)
    assert list(g0) != list(g1)
    # Same patient still same remapped label within a seed.
    assert g0[0] == g0[3]


def test_nested_cv_group_seed_changes_folds():
    df = _synthetic()
    r0 = evaluate_nested_groupkfold_oof(
        df, trainer_factory=_tiny_factory, n_splits=2, group_seed=0,
        targets=list(TARGET_NAMES[:2]),
    )
    r1 = evaluate_nested_groupkfold_oof(
        df, trainer_factory=_tiny_factory, n_splits=2, group_seed=1,
        targets=list(TARGET_NAMES[:2]),
    )
    folds0 = [tuple(f["val_groups"]) for f in r0.folds]
    folds1 = [tuple(f["val_groups"]) for f in r1.folds]
    assert folds0 != folds1


def test_repeated_nested_cv_three_seeds():
    df = _synthetic()
    out = evaluate_repeated_nested_cv(
        df,
        trainer_factory=_tiny_factory,
        seeds=(0, 1),
        n_splits=2,
        targets=list(TARGET_NAMES[:2]),
    )
    assert out["seeds"] == [0, 1]
    assert "mean" in out["aggregated"]["avg_mae_mm"]
    assert len(out["per_seed"]) == 2


def test_small_n_profile_exists():
    models = make_base_models(estimator_profile="small_n", target_name="kidney_left_delta_z")
    assert "RandomForest" in models
    assert "GradientBoosting" in models
    gbt = models["GradientBoosting"]
    assert gbt.max_depth <= 3
    assert gbt.n_estimators <= 300
