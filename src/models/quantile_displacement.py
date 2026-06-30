"""Quantile regression heads (P10 / P50 / P90) for displacement targets."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from src.features.phase1_schema import TARGET_NAMES

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
QUANTILE_LABELS: tuple[str, ...] = ("p10", "p50", "p90")
Z_TARGETS: tuple[str, ...] = ("kidney_left_delta_z", "kidney_right_delta_z")
DEFAULT_N_ESTIMATORS = 180
N_ESTIMATOR_CANDIDATES: tuple[int, ...] = (120, 180, 240, 300)
TARGET_COVERAGE = 0.80


def _gbt_quantile(alpha: float, *, n_estimators: int = DEFAULT_N_ESTIMATORS) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=n_estimators,
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
        self.n_estimators_by_target: Dict[str, int] = {}
        self.fitted_ = False

    @staticmethod
    def _oof_interval_coverage(
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        *,
        n_estimators: int,
        n_splits: int = 5,
    ) -> float:
        """OOF [p10, p90] coverage with GroupKFold (patient-level)."""
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_splits = min(n_splits, len(unique_groups))
        if n_splits < 2:
            return float("nan")

        splitter = GroupKFold(n_splits=n_splits)
        p10 = np.full(len(y), np.nan)
        p90 = np.full(len(y), np.nan)
        for train_idx, val_idx in splitter.split(X, y, groups=groups):
            m10 = _gbt_quantile(0.1, n_estimators=n_estimators)
            m90 = _gbt_quantile(0.9, n_estimators=n_estimators)
            m10.fit(X[train_idx], y[train_idx])
            m90.fit(X[train_idx], y[train_idx])
            p10[val_idx] = m10.predict(X[val_idx])
            p90[val_idx] = m90.predict(X[val_idx])

        valid = np.isfinite(p10) & np.isfinite(p90)
        if not np.any(valid):
            return float("nan")
        inside = (y[valid] >= p10[valid]) & (y[valid] <= p90[valid])
        return float(np.mean(inside))

    def tune_n_estimators_groupkfold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: Sequence[str],
        groups: np.ndarray,
        *,
        n_splits: int = 5,
        candidates: Sequence[int] | None = None,
    ) -> Dict[str, int]:
        """Pick n_estimators per Z target to approach 80% GroupKFold coverage."""
        names = list(target_names)
        picks: Dict[str, int] = {}
        for cand in candidates or N_ESTIMATOR_CANDIDATES:
            for i, target in enumerate(names):
                if target not in self.targets:
                    continue
                cov = self._oof_interval_coverage(
                    X,
                    y[:, i],
                    groups,
                    n_estimators=cand,
                    n_splits=n_splits,
                )
                if not np.isfinite(cov):
                    continue
                prev = picks.get(target)
                if prev is None:
                    picks[target] = cand
                    continue
                prev_cov = self._oof_interval_coverage(
                    X,
                    y[:, i],
                    groups,
                    n_estimators=prev,
                    n_splits=n_splits,
                )
                if abs(cov - TARGET_COVERAGE) < abs(prev_cov - TARGET_COVERAGE):
                    picks[target] = cand
        for target in self.targets:
            picks.setdefault(target, DEFAULT_N_ESTIMATORS)
        self.n_estimators_by_target = picks
        return picks

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: Sequence[str],
        sample_weight: np.ndarray | None = None,
        *,
        n_estimators_by_target: Dict[str, int] | None = None,
    ) -> "QuantileDisplacementPredictor":
        names = list(target_names)
        n_map = n_estimators_by_target or self.n_estimators_by_target
        self.models = {}
        for i, target in enumerate(names):
            if target not in self.targets:
                continue
            n_est = int(n_map.get(target, DEFAULT_N_ESTIMATORS))
            self.models[target] = {}
            for alpha, label in zip(QUANTILES, QUANTILE_LABELS):
                model = _gbt_quantile(alpha, n_estimators=n_est)
                if sample_weight is not None:
                    model.fit(X, y[:, i], sample_weight=sample_weight)
                else:
                    model.fit(X, y[:, i])
                self.models[target][label] = model
        self.fitted_ = True
        return self

    def fit_z_with_groupkfold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: Sequence[str],
        groups: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        n_splits: int = 5,
    ) -> "QuantileDisplacementPredictor":
        """Tune n_estimators on clinical groups, then fit Z-only quantile heads."""
        self.tune_n_estimators_groupkfold(
            X,
            y,
            target_names,
            groups,
            n_splits=n_splits,
        )
        return self.fit(
            X,
            y,
            target_names,
            sample_weight=sample_weight,
            n_estimators_by_target=self.n_estimators_by_target,
        )

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
