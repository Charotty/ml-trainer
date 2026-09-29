"""Shared helpers for model_kinds plugins (multi-output column views)."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class MultiOutputColumnView(BaseEstimator, RegressorMixin):
    """Fit a multi-output model on a target group, expose one column.

    ``y_context`` must be row-aligned with the ``X`` passed to :meth:`fit`.
    Rows with any missing value in the group columns are dropped for the
    joint fit; ``y`` (this target) is only used for the fallback median.
    """

    def __init__(
        self,
        base=None,
        y_context: np.ndarray | None = None,
        group_cols: Sequence[int] = (),
        column: int = 0,
    ):
        self.base = base
        self.y_context = y_context
        self.group_cols = tuple(group_cols)
        self.column = int(column)

    def fit(self, X, y, sample_weight=None):
        del sample_weight
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.fallback_ = float(np.nanmedian(y)) if y.size else 0.0
        self.model_ = None
        if self.y_context is None or not self.group_cols:
            return self
        Y = np.asarray(self.y_context, dtype=float)[:, list(self.group_cols)]
        if len(Y) != len(X):
            return self
        mask = np.all(np.isfinite(Y), axis=1)
        if int(mask.sum()) < 8:
            return self
        model = clone(self.base)
        model.fit(X[mask], Y[mask])
        self.model_ = model
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if getattr(self, "model_", None) is None:
            return np.full(len(X), getattr(self, "fallback_", 0.0))
        pred = np.asarray(self.model_.predict(X), dtype=float)
        if pred.ndim == 1:
            return pred
        pos = list(self.group_cols).index(self.column) if self.column in self.group_cols else 0
        return pred[:, pos]


def side_axis_group(target_name: str, target_names: Sequence[str]) -> tuple[list[int], int]:
    """Indices of the same-kidney x/y/z targets and the position of target_name."""
    side = "left" if "_left_" in target_name else "right"
    cols = [i for i, t in enumerate(target_names) if f"_{side}_" in t]
    return cols, list(target_names).index(target_name)


def joint_z_group(target_name: str, target_names: Sequence[str]) -> tuple[list[int], int]:
    cols = [i for i, t in enumerate(target_names) if t.endswith("_z")]
    return cols, list(target_names).index(target_name)
