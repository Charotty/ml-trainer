"""Kernel / neighbour model kinds: ``gpr``, ``svr_linear``, ``svr_rbf``, ``knn``.

All hyperparameters are chosen on inner 3-fold CV of the outer-train rows
(features are already imputed + standardized by the trainer).
"""

from __future__ import annotations

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.models.ensemble.model_kinds import register
from src.models.ensemble.model_kinds._common import DropSampleWeight

_CV = KFold(n_splits=3, shuffle=True, random_state=42)


@register("gpr", aliases=("gaussian_process",), description="GP with RBF + white noise")
def make_gpr(target_name: str, **ctx):
    del target_name, ctx
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(
        length_scale=10.0, length_scale_bounds=(1e-1, 1e3)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    return DropSampleWeight(
        GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=42
        )
    )


def _svr(kernel: str):
    grid = {"C": [0.3, 1.0, 3.0, 10.0], "epsilon": [0.5, 1.0, 2.0]}
    if kernel == "rbf":
        grid["gamma"] = ["scale", 0.003, 0.01]
    return DropSampleWeight(
        GridSearchCV(SVR(kernel=kernel), grid, cv=_CV, scoring="neg_mean_absolute_error")
    )


@register("svr_linear", description="Linear-kernel SVR, C/epsilon by inner CV")
def make_svr_linear(target_name: str, **ctx):
    del target_name, ctx
    return _svr("linear")


@register("svr_rbf", aliases=("svr",), description="RBF-kernel SVR, C/epsilon/gamma by inner CV")
def make_svr_rbf(target_name: str, **ctx):
    del target_name, ctx
    return _svr("rbf")


@register("knn", aliases=("kneighbors",), description="kNN 5-15 neighbours on standardized features")
def make_knn(target_name: str, **ctx):
    del target_name, ctx
    return DropSampleWeight(
        GridSearchCV(
            KNeighborsRegressor(),
            {"n_neighbors": [5, 7, 10, 15], "weights": ["uniform", "distance"]},
            cv=_CV,
            scoring="neg_mean_absolute_error",
        )
    )
