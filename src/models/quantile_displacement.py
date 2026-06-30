"""Quantile regression heads (P10 / P50 / P90) for displacement targets."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from src.features.phase1_schema import TARGET_NAMES

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
QUANTILE_LABELS: tuple[str, ...] = ("p10", "p50", "p90")


def _gbt_quantile(alpha: float) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=180,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        random_state=42,
    )


class QuantileDisplacementPredictor:
    """Per-target quantile models; P50 can serve as point estimate."""

    def __init__(self, targets: Sequence[str] | None = None):
        self.targets = list(targets or TARGET_NAMES)
        self.models: Dict[str, Dict[str, GradientBoostingRegressor]] = {}
        self.fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: Sequence[str],
        sample_weight: np.ndarray | None = None,
    ) -> "QuantileDisplacementPredictor":
        names = list(target_names)
        self.models = {}
        for i, target in enumerate(names):
            if target not in self.targets:
                continue
            self.models[target] = {}
            for alpha, label in zip(QUANTILES, QUANTILE_LABELS):
                model = _gbt_quantile(alpha)
                if sample_weight is not None:
                    model.fit(X, y[:, i], sample_weight=sample_weight)
                else:
                    model.fit(X, y[:, i])
                self.models[target][label] = model
        self.fitted_ = True
        return self

    def predict_quantiles(self, X: np.ndarray, target: str) -> Dict[str, np.ndarray]:
        if not self.fitted_ or target not in self.models:
            raise KeyError(f"No quantile models for {target}")
        return {label: model.predict(X) for label, model in self.models[target].items()}

    def predict_interval(self, X: np.ndarray, target: str) -> Dict[str, np.ndarray]:
        q = self.predict_quantiles(X, target)
        p10 = q["p10"]
        p90 = q["p90"]
        return {
            "p10": p10,
            "p50": q["p50"],
            "p90": p90,
            "width_mm": p90 - p10,
        }

    def predict_all(self, X: np.ndarray) -> Dict[str, Dict[str, float]]:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out: Dict[str, Dict[str, float]] = {}
        for target in self.models:
            interval = self.predict_interval(X, target)
            out[target] = {k: float(v[0]) for k, v in interval.items()}
        return out

    def coverage_rate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        target_names: Sequence[str],
    ) -> Dict[str, float]:
        """Fraction of samples with y in [p10, p90]."""
        names = list(target_names)
        rates: Dict[str, float] = {}
        for i, target in enumerate(names):
            if target not in self.models:
                continue
            interval = self.predict_interval(X, target)
            inside = (y_true[:, i] >= interval["p10"]) & (y_true[:, i] <= interval["p90"])
            rates[target] = float(np.mean(inside))
        return rates
