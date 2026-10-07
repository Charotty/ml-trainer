"""Robust / sparse linear model kinds: ``elasticnet`` and ``huber_linear``.

Hyperparameters are chosen by inner 3-fold CV on the outer-train rows only
(one row per patient, so KFold == GroupKFold here).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNetCV, HuberRegressor
from sklearn.model_selection import GridSearchCV, KFold

from src.models.ensemble.model_kinds import register
from src.models.ensemble.model_kinds._common import DropSampleWeight

_CV = KFold(n_splits=3, shuffle=True, random_state=42)


@register("elasticnet", aliases=("enet", "elastic_net"), description="ElasticNetCV (inner 3-fold)")
def make_elasticnet(target_name: str, **ctx):
    del target_name, ctx
    return DropSampleWeight(
        ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            # Explicit grid: ``n_alphas`` was removed in newer sklearn; arrays work in 1.5 and 1.9.
            alphas=np.logspace(-3, 2, 30),
            cv=_CV,
            max_iter=20000,
            random_state=42,
        )
    )


@register("huber_linear", aliases=("huber", "huberregressor"), description="HuberRegressor, alpha by inner CV")
def make_huber_linear(target_name: str, **ctx):
    del target_name, ctx
    search = GridSearchCV(
        HuberRegressor(max_iter=2000),
        param_grid={"alpha": [1e-3, 1e-2, 0.1, 1.0, 10.0], "epsilon": [1.35, 1.75]},
        cv=_CV,
        scoring="neg_mean_absolute_error",
    )
    return DropSampleWeight(search)
