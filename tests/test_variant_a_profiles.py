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


@pytest.mark.parametrize(
    "profile",
    ["absolute_error", "quantile_0.5", "huber_0.5", "huber_0.7", "huber_0.9", "huber_all_axes"],
)
def test_loss_profiles_configure_members(profile):
    models = make_base_models(
        estimator_profile="tiny", target_name="kidney_left_delta_x", loss_profile=profile
    )
    gbt = models["GradientBoosting"]
    if profile == "absolute_error":
        assert gbt.loss == "absolute_error"
        assert models["RandomForest"].criterion == "absolute_error"
    elif profile == "quantile_0.5":
        assert gbt.loss == "quantile" and gbt.alpha == 0.5
    else:
        assert gbt.loss == "huber"


@pytest.mark.parametrize("mode", ["default", "off", "soft", "y_only"])
def test_yz_boost_modes(mode):
    trainer = AdaptiveEnsembleTrainer(enrichment_mode="none", estimator_profile="tiny", yz_boost_mode=mode)
    y = np.array([0.0, 15.0, 30.0, 60.0])
    wz = trainer._per_target_sample_weights("kidney_left_delta_z", y)
    wy = trainer._per_target_sample_weights("kidney_left_delta_y", y)
    if mode == "off":
        assert wz is None and wy is None
    elif mode == "soft":
        assert wz.max() <= 2.0 + 1e-9
    elif mode == "y_only":
        assert np.allclose(wz, 1.0)
        assert wy.max() > 1.0
    else:
        assert wz.max() == pytest.approx(3.5)


@pytest.mark.parametrize("ewm", ["equal", "fixed_prior", "shrink", "stacking", "best_per_axis"])
def test_ensemble_weight_modes_run(ewm):
    df = _synthetic(n=10)
    r = evaluate_nested_groupkfold_oof(
        df,
        trainer_factory=lambda **kw: _tiny_factory(ensemble_weight_mode=ewm, **kw),
        n_splits=2,
        group_seed=0,
        targets=list(TARGET_NAMES[:2]),
    )
    for fold in r.folds:
        for weights in fold["inner_ensemble_weights"].values():
            assert sum(weights.values()) == pytest.approx(1.0)
            if ewm == "best_per_axis":
                assert sorted(weights.values())[-1] == pytest.approx(1.0)


def test_z_postprocess_median_shrink_bounds():
    from sklearn.linear_model import LinearRegression

    from src.models.ensemble.estimators import ZPostprocessWrapper

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40) * 5
    base = LinearRegression().fit(X, y)
    w = ZPostprocessWrapper(estimator=base, mode="median_shrink").fit(X, y)
    assert w.k_ in ZPostprocessWrapper.K_GRID
    clip = ZPostprocessWrapper(estimator=base, mode="clip_q05_q95").fit(X, y)
    pred = clip.predict(X * 100)
    assert pred.min() >= clip.q05_ - 1e-9 and pred.max() <= clip.q95_ + 1e-9


def test_keep_feature_names_limits_matrix():
    df = _synthetic(n=8)
    keep = list(BASE_FEATURES[:4])
    trainer = AdaptiveEnsembleTrainer(
        enrichment_mode="none",
        estimator_profile="tiny",
        keep_feature_names=keep,
    )
    prepared = trainer.prepare_training_data_fit(df)
    assert prepared[0] is not None
    base_cols = [c for c in trainer.feature_names if not c.endswith("_was_missing")]
    assert set(base_cols) <= set(keep) | {c for c in trainer.feature_names if c.startswith(("sex_", "body_type_"))}


def test_plugin_registry_roundtrip():
    from sklearn.linear_model import LinearRegression

    from src.models.baselines import normalize_model_kind
    from src.models.ensemble import model_kinds

    @model_kinds.register("unit_test_linear", aliases=("utl",))
    def _factory(target_name, **ctx):
        return LinearRegression()

    assert normalize_model_kind("utl") == "unit_test_linear"
    df = _synthetic(n=8)
    r = evaluate_nested_groupkfold_oof(
        df,
        trainer_factory=lambda **kw: _tiny_factory(model_kind="unit_test_linear", **kw),
        n_splits=2,
        targets=list(TARGET_NAMES[:2]),
    )
    assert np.isfinite(r.metrics["avg_mae_mm"])
