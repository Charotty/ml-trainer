"""Multi-output model kinds.

* ``mtenet``   — MultiTaskElasticNetCV over x/y/z of the same kidney.
* ``chain``    — RegressorChain X→Y→Z (RidgeCV links) per kidney.
* ``joint_z``  — MultiTaskElasticNetCV over left_z/right_z jointly; x/y use RidgeCV.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import MultiTaskElasticNetCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.multioutput import RegressorChain

from src.models.ensemble.model_kinds import register
from src.models.ensemble.model_kinds._common import (
    DropSampleWeight,
    MultiOutputColumnView,
    joint_z_group,
    side_axis_group,
)

_CV = KFold(n_splits=3, shuffle=True, random_state=42)
_ALPHAS = np.logspace(-2, 3, 20)


def _mten():
    return MultiTaskElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 2, 20), cv=_CV, max_iter=20000, random_state=42
    )


def _axis_order(target_names, cols):
    order = {"x": 0, "y": 1, "z": 2}
    return sorted(cols, key=lambda i: order.get(str(target_names[i]).split("_")[-1], 9))


@register("mtenet", aliases=("multitask_elasticnet",), description="MultiTaskElasticNet per kidney")
def make_mtenet(target_name: str, *, target_names=(), y_context=None, **ctx):
    del ctx
    cols, col = side_axis_group(target_name, list(target_names))
    return MultiOutputColumnView(base=_mten(), y_context=y_context, group_cols=cols, column=col)


@register("chain", aliases=("regressor_chain",), description="RegressorChain X->Y->Z per kidney")
def make_chain(target_name: str, *, target_names=(), y_context=None, **ctx):
    del ctx
    names = list(target_names)
    cols, col = side_axis_group(target_name, names)
    ordered = _axis_order(names, cols)
    base = RegressorChain(RidgeCV(alphas=_ALPHAS), order=list(range(len(ordered))))
    return MultiOutputColumnView(base=base, y_context=y_context, group_cols=ordered, column=col)


@register("joint_z", description="Joint left/right Z (MultiTaskElasticNet); X/Y via RidgeCV")
def make_joint_z(target_name: str, *, target_names=(), y_context=None, **ctx):
    del ctx
    if not str(target_name).endswith("_z"):
        return DropSampleWeight(RidgeCV(alphas=_ALPHAS))
    cols, col = joint_z_group(target_name, list(target_names))
    return MultiOutputColumnView(base=_mten(), y_context=y_context, group_cols=cols, column=col)
