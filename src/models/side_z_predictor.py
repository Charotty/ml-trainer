"""Dedicated trainers for kidney_left_delta_z and kidney_right_delta_z."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import VotingRegressor

LEFT_Z = "kidney_left_delta_z"
RIGHT_Z = "kidney_right_delta_z"

Z_FEATURE_HINTS: tuple[str, ...] = (
    "kidney_left_center_z_rel",
    "kidney_right_center_z_rel",
    "kidney_left_z_over_depth",
    "kidney_right_z_over_depth",
    "kidney_left_z_span_supine_mm",
    "kidney_right_z_span_supine_mm",
    "kidney_left_z_delta_span_mm",
    "kidney_right_z_delta_span_mm",
    "kidney_z_asymmetry_rel",
    "lordosis_x_depth",
    "proj_lat_kidney_left_center_z_rel",
    "proj_lat_kidney_right_center_z_rel",
    "proj_sup_kidney_left_center_z_rel",
    "proj_sup_kidney_right_center_z_rel",
    "proj_diff_left_z",
    "proj_diff_right_z",
    "body_depth_mm",
    "body_sagittal_index",
)


def z_feature_indices(feature_names: Sequence[str], n_cols: int) -> list[int]:
    names = list(feature_names)[:n_cols]
    picked = [
        i
        for i, n in enumerate(names)
        if n in Z_FEATURE_HINTS or n.endswith("_z_rel") or "_z_" in n
    ]
    if len(picked) < 8:
        picked = list(range(n_cols))
    return sorted({i for i in picked if 0 <= i < n_cols})


def _build_side_model(target: str) -> VotingRegressor:
    if target == LEFT_Z:
        gbt = GradientBoostingRegressor(
            loss="huber",
            n_estimators=280,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            random_state=42,
        )
    else:
        gbt = GradientBoostingRegressor(
            loss="huber",
            n_estimators=220,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            random_state=43,
        )
    rf = RandomForestRegressor(
        n_estimators=240,
        max_depth=12,
        min_samples_leaf=2,
        random_state=44,
        n_jobs=-1,
    )
    return VotingRegressor(
        estimators=[("gbt", gbt), ("rf", rf)],
        weights=[2.5, 1.0],
        n_jobs=1,
    )


class SideZPredictor:
    """Side-specific Z model with optional feature column subset."""

    def __init__(self, target: str):
        if target not in {LEFT_Z, RIGHT_Z}:
            raise ValueError(f"Unsupported Z target: {target}")
        self.target = target
        self.model = _build_side_model(target)
        self.feature_indices_: list[int] | None = None
        self.fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
        sample_weight: np.ndarray | None = None,
    ) -> "SideZPredictor":
        self.feature_indices_ = z_feature_indices(feature_names, X.shape[1])
        Xz = X[:, self.feature_indices_]
        if sample_weight is not None:
            self.model.fit(Xz, y, sample_weight=sample_weight)
        else:
            self.model.fit(Xz, y)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_ or self.feature_indices_ is None:
            raise RuntimeError("SideZPredictor is not fitted")
        return self.model.predict(X[:, self.feature_indices_])


class SideZModelPair:
    """Container for left/right Z models."""

    def __init__(self):
        self.left = SideZPredictor(LEFT_Z)
        self.right = SideZPredictor(RIGHT_Z)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: Sequence[str],
        feature_names: Sequence[str],
        sample_weight: np.ndarray | None = None,
    ) -> "SideZModelPair":
        names = list(target_names)
        li = names.index(LEFT_Z)
        ri = names.index(RIGHT_Z)
        w = sample_weight
        self.left.fit(X, y[:, li], feature_names, sample_weight=w)
        self.right.fit(X, y[:, ri], feature_names, sample_weight=w)
        return self

    def predict_matrix(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {
            LEFT_Z: self.left.predict(X),
            RIGHT_Z: self.right.predict(X),
        }

    def replace_in_models(self, models: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(models)
        out[LEFT_Z] = self.left.model
        out[RIGHT_Z] = self.right.model
        return out
