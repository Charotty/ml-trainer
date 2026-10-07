"""Strongly regularized LightGBM model kind (``lightgbm``)."""

from __future__ import annotations

from src.models.ensemble.model_kinds import register


@register("lightgbm", aliases=("lgbm",), description="LightGBM, shallow trees + strong L2")
def make_lightgbm(target_name: str, *, estimator_profile: str = "production", **ctx):
    del ctx
    from lightgbm import LGBMRegressor

    axis = str(target_name).split("_")[-1]
    tiny = estimator_profile == "tiny"
    return LGBMRegressor(
        n_estimators=20 if tiny else 300,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=10,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=5.0,
        reg_alpha=0.5,
        objective="huber" if axis == "z" else "regression_l1",
        random_state=42,
        n_jobs=2,
        verbose=-1,
    )
