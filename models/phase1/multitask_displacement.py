"""Lightweight multitask model: shared SVD trunk + per-target Ridge heads."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.phase1_schema import TARGET_NAMES


class MultitaskDisplacementModel:
    """Shared representation (scale + SVD) with independent Ridge heads."""

    def __init__(self, *, n_components: int = 24, alpha: float = 3.0):
        self.n_components = n_components
        self.alpha = alpha
        self.shared: Pipeline | None = None
        self.heads: dict[str, Ridge] = {}
        self.target_names: list[str] = []
        self.fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: Sequence[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "MultitaskDisplacementModel":
        self.target_names = list(target_names or TARGET_NAMES)
        n_comp = min(self.n_components, X.shape[1] - 1, X.shape[0] - 2)
        n_comp = max(4, n_comp)
        self.shared = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svd", TruncatedSVD(n_components=n_comp, random_state=42)),
            ]
        )
        Z = self.shared.fit_transform(X)
        self.heads = {}
        for i, name in enumerate(self.target_names):
            head = Ridge(alpha=self.alpha)
            if sample_weight is not None:
                head.fit(Z, y[:, i], sample_weight=sample_weight)
            else:
                head.fit(Z, y[:, i])
            self.heads[name] = head
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_ or self.shared is None:
            raise RuntimeError("MultitaskDisplacementModel is not fitted")
        Z = self.shared.transform(X)
        cols = [self.heads[name].predict(Z) for name in self.target_names]
        return np.column_stack(cols)

    def predict_dict(self, X: np.ndarray) -> dict[str, float]:
        mat = self.predict(X)
        return {name: float(mat[0, i]) for i, name in enumerate(self.target_names)}

    def blend_with_point_predictions(
        self,
        point_pred: dict[str, float],
        X: np.ndarray,
        *,
        z_blend: float = 0.35,
        xy_blend: float = 0.15,
    ) -> dict[str, float]:
        """Blend multitask heads with point ensemble (stronger on Z)."""
        mt = self.predict_dict(X)
        out = dict(point_pred)
        for name, value in mt.items():
            if name.endswith("_z"):
                if z_blend <= 0.0:
                    continue
                w = z_blend
            elif name.endswith("_y"):
                w = xy_blend
            else:
                w = xy_blend
            out[name] = (1.0 - w) * float(point_pred.get(name, value)) + w * float(value)
        return out
