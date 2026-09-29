"""PLS regression model kinds: ``pls`` (per target) and ``pls_multi`` (3 axes of one kidney)."""

from __future__ import annotations

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold

from src.models.ensemble.model_kinds import register
from src.models.ensemble.model_kinds._common import (
    DropSampleWeight,
    MultiOutputColumnView,
    side_axis_group,
)

_CV = KFold(n_splits=3, shuffle=True, random_state=42)
_GRID = {"n_components": [1, 2, 3, 5]}


def _pls_search():
    return GridSearchCV(
        PLSRegression(scale=False, max_iter=1000),
        _GRID,
        cv=_CV,
        scoring="neg_mean_absolute_error",
    )


@register("pls", description="PLSRegression per target, n_components by inner CV")
def make_pls(target_name: str, **ctx):
    del target_name, ctx
    return DropSampleWeight(_pls_search())


@register("pls_multi", description="PLSRegression on x/y/z of the same kidney jointly")
def make_pls_multi(target_name: str, *, target_names=(), y_context=None, **ctx):
    del ctx
    cols, col = side_axis_group(target_name, list(target_names))
    return MultiOutputColumnView(
        base=_pls_search(), y_context=y_context, group_cols=cols, column=col
    )
