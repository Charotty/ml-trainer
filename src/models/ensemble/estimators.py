"""Estimator factory: create base models and voting ensembles."""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Lasso, Ridge

PACKAGE_STATUS = "production"

MODEL_KIND_ALIASES = {
    "ensemble": "ensemble",
    "adaptive_ensemble": "ensemble",
    "rf": "rf",
    "randomforest": "rf",
    "random_forest": "rf",
    "gbt": "gbt",
    "gb": "gbt",
    "gradientboosting": "gbt",
    "gradient_boosting": "gbt",
    "ridge": "ridge",
    "mean": "mean",
    "median": "median",
    "group_median": "group_median",
}
SINGLE_KIND_TO_NAME = {
    "rf": "RandomForest",
    "gbt": "GradientBoosting",
    "ridge": "Ridge",
}

DEFAULT_BEST_MODELS = {
    "kidney_left_delta_x": "RandomForest",
    "kidney_left_delta_y": "GradientBoosting",
    "kidney_left_delta_z": "GradientBoosting",
    "kidney_right_delta_x": "Ridge",
    "kidney_right_delta_y": "GradientBoosting",
    "kidney_right_delta_z": "GradientBoosting",
}

DEFAULT_ADAPTIVE_WEIGHTS = {
    "kidney_left_delta_x": {
        "RandomForest": 2.5,
        "Lasso": 1.0,
        "Ridge": 0.8,
        "GradientBoosting": 0.6,
    },
    "kidney_left_delta_y": {
        "RandomForest": 1.2,
        "Lasso": 1.0,
        "Ridge": 0.8,
        "GradientBoosting": 2.2,
    },
    "kidney_left_delta_z": {
        "RandomForest": 1.0,
        "Lasso": 0.6,
        "Ridge": 0.6,
        "GradientBoosting": 2.8,
    },
    "kidney_right_delta_x": {
        "RandomForest": 1.0,
        "Lasso": 0.8,
        "Ridge": 2.5,
        "GradientBoosting": 0.6,
    },
    "kidney_right_delta_y": {
        "RandomForest": 1.2,
        "Lasso": 1.0,
        "Ridge": 0.8,
        "GradientBoosting": 2.2,
    },
    "kidney_right_delta_z": {
        "RandomForest": 1.0,
        "Lasso": 0.6,
        "Ridge": 0.6,
        "GradientBoosting": 2.8,
    },
}


def make_base_models(
    *,
    estimator_profile: str = "production",
    target_name: str | None = None,
    loss_profile: str | None = None,
    estimator_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create unfitted RF/Lasso/Ridge/GBT members for one target."""
    if estimator_profile not in {"production", "tiny", "small_n"}:
        raise ValueError(
            f"estimator_profile must be 'production', 'tiny', or 'small_n', "
            f"got {estimator_profile!r}"
        )
    axis = target_name.split("_")[-1] if target_name else None

    if estimator_profile == "tiny":
        rf_config = {
            "n_estimators": 4,
            "max_depth": 2,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": 1,
        }
        gbt_config = {
            "n_estimators": 4,
            "learning_rate": 0.2,
            "max_depth": 2,
            "subsample": 1.0,
            "random_state": 42,
        }
        lasso_config = {"alpha": 0.5, "max_iter": 2000, "random_state": 42}
        ridge_config = {"alpha": 1.0, "solver": "auto", "random_state": 42}
    elif estimator_profile == "small_n":
        rf_config = {
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_split": 10,
            "min_samples_leaf": 10,
            "max_features": 0.4,
            "random_state": 42,
            "n_jobs": -1,
        }
        gbt_config = {
            "n_estimators": 200,
            "learning_rate": 0.04,
            "max_depth": 2,
            "min_samples_leaf": 10,
            "subsample": 0.7,
            "random_state": 42,
        }
        lasso_config = {"alpha": 0.15, "max_iter": 5000, "random_state": 42}
        ridge_config = {"alpha": 1.5, "solver": "auto", "random_state": 42}
    else:
        rf_config = {
            "n_estimators": 600 if axis in ("y", "z") else 500,
            "max_depth": 24 if axis in ("y", "z") else 20,
            "min_samples_split": 6 if axis in ("y", "z") else 10,
            "min_samples_leaf": 2 if axis in ("y", "z") else 4,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
        gbt_config = {
            "n_estimators": 700 if axis == "z" else 550,
            "learning_rate": 0.04 if axis in ("y", "z") else 0.05,
            "max_depth": 7 if axis == "z" else 6,
            "subsample": 0.85,
            "random_state": 42,
        }
        if axis == "z":
            gbt_config["loss"] = "huber"
            gbt_config["alpha"] = 0.9
        lasso_config = {
            "alpha": 0.08 if axis in ("y", "z") else 0.1,
            "max_iter": 5000,
            "random_state": 42,
        }
        ridge_config = {
            "alpha": 0.8 if axis in ("y", "z") else 1.0,
            "solver": "auto",
            "random_state": 42,
        }

    # loss_profile overlays (stage 3). Default keeps production Z-huber behaviour.
    profile = (loss_profile or "default").strip().lower()
    if profile == "absolute_error":
        rf_config["criterion"] = "absolute_error"
        gbt_config["loss"] = "absolute_error"
        gbt_config.pop("alpha", None)
    elif profile == "quantile_0.5":
        gbt_config["loss"] = "quantile"
        gbt_config["alpha"] = 0.5
    elif profile.startswith("huber"):
        gbt_config["loss"] = "huber"
        if profile == "huber_0.5":
            gbt_config["alpha"] = 0.5
        elif profile == "huber_0.7":
            gbt_config["alpha"] = 0.7
        elif profile in {"huber_0.9", "huber_all_axes"}:
            gbt_config["alpha"] = 0.9
        else:
            gbt_config["alpha"] = 0.9
        if profile == "huber_all_axes" or axis == "z":
            pass  # huber already set
    elif profile == "default":
        pass
    else:
        raise ValueError(f"Unknown loss_profile={loss_profile!r}")

    if estimator_overrides:
        rf_config.update(dict(estimator_overrides.get("RandomForest") or {}))
        gbt_config.update(dict(estimator_overrides.get("GradientBoosting") or {}))
        lasso_config.update(dict(estimator_overrides.get("Lasso") or {}))
        ridge_config.update(dict(estimator_overrides.get("Ridge") or {}))

    return {
        "RandomForest": RandomForestRegressor(**rf_config),
        "Lasso": Lasso(**lasso_config),
        "Ridge": Ridge(**ridge_config),
        "GradientBoosting": GradientBoostingRegressor(**gbt_config),
    }


class GroupMedianRegressor(BaseEstimator, RegressorMixin):
    """Predict the training median within sex×body_type (or global fallback)."""

    def __init__(self, group_indices: tuple[int, ...] | list[int] | None = None):
        self.group_indices = tuple(group_indices or ())

    def fit(self, X, y, sample_weight=None):
        del sample_weight
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.global_median_ = float(np.nanmedian(y)) if len(y) else 0.0
        buckets: dict[tuple, list[float]] = {}
        if self.group_indices and X.size:
            idxs = [i for i in self.group_indices if 0 <= int(i) < X.shape[1]]
            for row, target in zip(X, y):
                if not np.isfinite(target):
                    continue
                key = tuple(
                    float(np.round(row[i], 6)) if np.isfinite(row[i]) else None for i in idxs
                )
                buckets.setdefault(key, []).append(float(target))
        self.group_medians_ = {k: float(np.median(vals)) for k, vals in buckets.items()}
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.full(len(X), getattr(self, "global_median_", 0.0), dtype=float)
        medians = getattr(self, "group_medians_", {}) or {}
        if not self.group_indices or not medians:
            return out
        idxs = [i for i in self.group_indices if 0 <= int(i) < X.shape[1]]
        for i, row in enumerate(X):
            key = tuple(
                float(np.round(row[j], 6)) if np.isfinite(row[j]) else None for j in idxs
            )
            if key in medians:
                out[i] = medians[key]
        return out


_AXIS_TOKEN = re.compile(r"(?:^|_)([xyz])(?:_|$)")


def per_axis_feature_indices(feature_names: list[str], target_name: str) -> list[int] | None:
    """Columns allowed for ``target_name``: drop coordinate features of the other axes.

    A feature is axis-bound when its name has a standalone ``x``/``y``/``z`` token
    (``kidney_left_center_x_rel``, ``spine_center_z``, ``body_com_y`` ...).
    Axis-free features (anthropometry, volumes, distances, trends) are kept.
    """
    axis = str(target_name).rsplit("_", 1)[-1]
    if axis not in {"x", "y", "z"} or not feature_names:
        return None
    keep = []
    for j, name in enumerate(feature_names):
        tokens = set(_AXIS_TOKEN.findall(str(name)))
        if tokens and axis not in tokens:
            continue
        keep.append(j)
    if not keep or len(keep) == len(feature_names):
        return None
    return keep


class ColumnSubsetRegressor(RegressorMixin, BaseEstimator):
    """Fit/predict ``estimator`` on a fixed column subset of X (sample_weight forwarded)."""

    def __init__(self, estimator=None, columns: tuple[int, ...] = ()):
        self.estimator = estimator
        self.columns = columns

    def fit(self, X, y, sample_weight=None, **fit_params):
        cols = list(self.columns)
        self.estimator_ = clone(self.estimator)
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        self.estimator_.fit(np.asarray(X)[:, cols], y, **fit_params)
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X)[:, list(self.columns)])


class ZPostprocessWrapper(BaseEstimator, RegressorMixin):
    """Wrap an already-fitted regressor with Z-axis post-processing (stage 8).

    Modes:
      * ``median_shrink`` — pred = median + k·(raw − median); k chosen on inner
        3-fold OOF predictions of a cloned estimator (grid 0..1), never on the
        outer validation fold.
      * ``clip_q05_q95`` — clip to 5–95% quantiles of training Z.
      * ``sign_magnitude`` — keep the sign, shrink the magnitude toward the
        training median |Z| with inner-OOF-chosen k.
    """

    K_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __init__(self, estimator=None, mode: str = "median_shrink", k: float | None = None):
        self.estimator = estimator
        self.mode = mode
        self.k = k

    def _inner_oof(self, X, y) -> np.ndarray | None:
        from sklearn.model_selection import KFold

        X = np.asarray(X, dtype=float)
        n = len(y)
        if n < 9 or self.estimator is None:
            return None
        oof = np.full(n, np.nan)
        for tr, va in KFold(n_splits=3, shuffle=True, random_state=42).split(X):
            est = clone(self.estimator)
            try:
                est.fit(X[tr], y[tr])
                oof[va] = np.asarray(est.predict(X[va]), dtype=float).reshape(-1)
            except Exception:
                return None
        return oof

    def _apply(self, pred: np.ndarray, k: float) -> np.ndarray:
        mode = (self.mode or "").strip().lower()
        if mode == "median_shrink":
            return self.median_ + k * (pred - self.median_)
        if mode == "sign_magnitude":
            mag = np.abs(pred)
            return np.sign(pred) * (self.abs_median_ + k * (mag - self.abs_median_))
        if mode in {"clip_q05_q95", "clip"}:
            return np.clip(pred, self.q05_, self.q95_)
        return pred

    def fit(self, X, y, sample_weight=None):
        del sample_weight
        y = np.asarray(y, dtype=float).reshape(-1)
        finite = y[np.isfinite(y)]
        self.median_ = float(np.median(finite)) if finite.size else 0.0
        self.abs_median_ = float(np.median(np.abs(finite))) if finite.size else 0.0
        self.q05_ = float(np.percentile(finite, 5)) if finite.size else -50.0
        self.q95_ = float(np.percentile(finite, 95)) if finite.size else 50.0
        mode = (self.mode or "").strip().lower()
        self.k_ = 1.0 if self.k is None else float(self.k)
        if self.k is None and mode in {"median_shrink", "sign_magnitude"}:
            oof = self._inner_oof(X, y)
            if oof is not None:
                mask = np.isfinite(oof) & np.isfinite(y)
                best_k, best_mae = 1.0, float("inf")
                for k in self.K_GRID:
                    mae = float(np.mean(np.abs(self._apply(oof[mask], k) - y[mask])))
                    if mae < best_mae - 1e-9:
                        best_k, best_mae = k, mae
                self.k_ = best_k
        return self

    def predict(self, X):
        pred = np.asarray(self.estimator.predict(X), dtype=float).reshape(-1)
        return self._apply(pred, getattr(self, "k_", 1.0))


def make_single_estimator(
    kind: str,
    base_models: Mapping[str, Any],
    *,
    feature_names: list[str] | None = None,
) -> tuple[str, Any]:
    """Return (estimator_name, unfitted estimator) for a non-ensemble kind."""
    if kind in ("mean", "median"):
        return kind, DummyRegressor(strategy=kind)
    if kind == "group_median":
        names = list(feature_names or [])
        idxs = [names.index(col) for col in ("sex", "body_type") if col in names]
        return "GroupMedian", GroupMedianRegressor(group_indices=tuple(idxs))
    name = SINGLE_KIND_TO_NAME[kind]
    return name, clone(base_models[name])


def copy_estimator(model: Any) -> Any:
    try:
        return clone(model)
    except Exception:
        if hasattr(model, "get_params"):
            return type(model)(**model.get_params())
        return type(model)()


def fit_kwargs_for_model(model_name: str, sample_weight: np.ndarray | None) -> dict:
    del model_name
    if sample_weight is None:
        return {}
    return {"sample_weight": sample_weight}


def create_optimized_voting_ensemble(
    models: Mapping[str, Any],
    target_name: str,
    optimized_weights: Mapping[str, float],
    *,
    adaptive_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> VotingRegressor:
    estimators = [(name, clone(models[name])) for name in models.keys()]
    weights = np.array([optimized_weights[name] for name in models.keys()], dtype=float)
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
        target_weights = (adaptive_weights or DEFAULT_ADAPTIVE_WEIGHTS).get(target_name, {})
        weights = np.array(
            [target_weights.get(name, 1.0) for name in models.keys()],
            dtype=float,
        )
        if float(weights.sum()) <= 0:
            weights = np.ones(len(models), dtype=float)
        weights = weights / weights.sum()
    return VotingRegressor(estimators=estimators, weights=weights.tolist(), n_jobs=1)


def create_adaptive_voting_ensemble(
    models: Mapping[str, Any],
    target_name: str,
    *,
    adaptive_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> VotingRegressor:
    target_weights = (adaptive_weights or DEFAULT_ADAPTIVE_WEIGHTS)[target_name]
    estimators = []
    weights = []
    for model_name, weight in target_weights.items():
        if model_name in models:
            estimators.append((model_name, clone(models[model_name])))
            weights.append(weight)
    return VotingRegressor(estimators=estimators, weights=weights, n_jobs=1)


def create_standard_voting_ensemble(models: Mapping[str, Any]) -> VotingRegressor:
    estimators = [(name, clone(models[name])) for name in models.keys()]
    return VotingRegressor(estimators=estimators, weights=None, n_jobs=1)


def fit_voting_ensemble(
    ensemble: VotingRegressor,
    X,
    y,
    sample_weight=None,
) -> None:
    if sample_weight is None:
        ensemble.fit(X, y)
        return
    named = getattr(ensemble, "named_estimators", None)
    if named is None:
        ensemble.fit(X, y)
        return
    fitted = []
    for name, est in named.items():
        est_fitted = copy_estimator(est)
        est_fitted.fit(X, y, **fit_kwargs_for_model(name, sample_weight))
        fitted.append(est_fitted)
    ensemble.estimators_ = fitted
