"""Adaptive ensemble trainer (production). Feature/estimator/tuner/serializer split."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.na_trend_features import NaTrendStore
from src.models.ensemble.estimators import (
    DEFAULT_ADAPTIVE_WEIGHTS,
    DEFAULT_BEST_MODELS,
    MODEL_KIND_ALIASES,
    SINGLE_KIND_TO_NAME,
    copy_estimator,
    create_adaptive_voting_ensemble,
    create_optimized_voting_ensemble,
    create_standard_voting_ensemble,
    fit_kwargs_for_model,
    fit_voting_ensemble,
    make_base_models,
    make_single_estimator,
)
from src.models.ensemble.features import FeatureTransformer
from src.models.ensemble.serializer import save_trainer
from src.models.ensemble.tuner import (
    average_groupkfold_weights,
    optimize_ensemble_weights as tune_ensemble_weights,
    sanitize_predictions,
)

PACKAGE_STATUS = "production"


class AdaptiveEnsembleTrainer(FeatureTransformer):
    def __init__(
        self,
        *,
        z_head: str = "ensemble",
        enrichment_mode: str = "projection",
        na_trend_store: NaTrendStore | None = None,
        estimator_profile: str = "production",
        inner_n_splits: int | None = None,
        model_kind: str = "ensemble",
        yz_target_boost: bool = True,
        yz_boost_mode: str | None = None,
        loss_profile: str | None = None,
        estimator_overrides: Mapping[str, Mapping[str, Any]] | dict | None = None,
        ensemble_weight_mode: str | None = None,
        z_postprocess: str | None = None,
        encode_categoricals: bool = True,
        add_missing_indicators: bool = True,
        drop_feature_groups: tuple[str, ...] | list[str] | None = None,
        drop_feature_prefixes: tuple[str, ...] | list[str] | None = None,
    ):
        super().__init__(
            enrichment_mode=enrichment_mode,
            na_trend_store=na_trend_store,
            encode_categoricals=encode_categoricals,
            add_missing_indicators=add_missing_indicators,
            drop_feature_groups=drop_feature_groups,
            drop_feature_prefixes=drop_feature_prefixes,
            verbose=True,
        )
        self.z_head = z_head
        if estimator_profile not in {"production", "tiny", "small_n"}:
            raise ValueError(
                f"estimator_profile must be 'production', 'tiny', or 'small_n', "
                f"got {estimator_profile!r}"
            )
        self.estimator_profile = estimator_profile
        self.inner_n_splits = (
            int(inner_n_splits)
            if inner_n_splits is not None
            else (2 if estimator_profile in {"tiny", "small_n"} else 3)
        )
        kind_key = str(model_kind).strip().lower()
        if kind_key not in MODEL_KIND_ALIASES:
            raise ValueError(
                f"model_kind must be one of {sorted(set(MODEL_KIND_ALIASES.values()))}, got {model_kind!r}"
            )
        self.model_kind = MODEL_KIND_ALIASES[kind_key]
        # yz_boost_mode overrides the boolean flag when set.
        mode = (yz_boost_mode or ("default" if yz_target_boost else "off")).strip().lower()
        if mode not in {"default", "off", "soft", "y_only"}:
            raise ValueError(f"yz_boost_mode must be default|off|soft|y_only, got {mode!r}")
        self.yz_boost_mode = mode
        self.yz_target_boost = mode != "off"
        self.loss_profile = loss_profile
        self.estimator_overrides = dict(estimator_overrides or {})
        self.ensemble_weight_mode = (ensemble_weight_mode or "optimized").strip().lower()
        self.z_postprocess = z_postprocess
        self._inner_fold_weight_traces: dict = {}
        self._inner_weight_variance: dict = {}
        self._optimized_weights: dict = {}
        self.z_driver_names: list[str] = []
        self._X_train_imputed: np.ndarray | None = None
        self._X_val_imputed: np.ndarray | None = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.target_names = []
        self.results = {}
        self.trained_models = {}
        self.X_train = None
        self.train_sample_weights = None
        self.cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        self.best_models = dict(DEFAULT_BEST_MODELS)
        self.adaptive_weights = {k: dict(v) for k, v in DEFAULT_ADAPTIVE_WEIGHTS.items()}

    def load_integrated_data(self):
        print("Loading integrated datasets from all sources...")
        try:
            train_df = pd.read_csv("data/processed/train.csv")
            val_df = pd.read_csv("data/processed/validation.csv")
            print(f"Integrated Train dataset: {len(train_df)} cases")
            print(f"Integrated Validation dataset: {len(val_df)} cases")
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            print(f"Combined dataset (legacy view): {len(combined_df)} cases")
            return combined_df, train_df, val_df
        except FileNotFoundError:
            print("Integrated data files not found. Please run data integration first.")
            print("Run: python src/models/data_integration_fix.py")
            return None, None, None

    def prepare_training_data(self, df):
        import warnings as _warnings

        _warnings.warn(
            "prepare_training_data(df) introduces train/val leakage when df is "
            "the concatenation of train and validation. Use "
            "prepare_training_data_split(train_df, val_df) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        out = self._build_feature_matrix(df)
        if out[0] is None:
            return None, None, None, None
        X, y, all_feature_cols, target_cols = out
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_imp = self.imputer.fit_transform(X_train)
        X_test_imp = self.imputer.transform(X_test)
        X_train_scaled = self.scaler.fit_transform(X_train_imp)
        X_test_scaled = self.scaler.transform(X_test_imp)
        self.feature_names = all_feature_cols
        self.target_names = target_cols
        self.X_train = X_train_scaled
        print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def prepare_training_data_split(self, train_df, val_df):
        train_df = self._encode_frame(train_df, fit=True)
        val_df = self._encode_frame(val_df, fit=False)
        out_train = self._build_feature_matrix(train_df)
        if out_train[0] is None:
            return None, None, None, None
        X_train_raw, y_train, feature_cols_train, target_cols_train = out_train
        if "sample_weight" in train_df.columns:
            self.train_sample_weights = (
                pd.to_numeric(train_df["sample_weight"], errors="coerce")
                .fillna(1.0)
                .astype(float)
                .values
            )
        else:
            self.train_sample_weights = None
        out_val = self._build_feature_matrix(val_df)
        if out_val[0] is None:
            return None, None, None, None
        X_val_raw, y_val, feature_cols_val, target_cols_val = out_val
        if feature_cols_train != feature_cols_val:
            common = [c for c in feature_cols_train if c in feature_cols_val]
            print(
                "WARNING: train/val feature sets diverged after engineering "
                f"({len(feature_cols_train)} vs {len(feature_cols_val)}); using "
                f"{len(common)} common columns."
            )
            train_indices = [feature_cols_train.index(c) for c in common]
            val_indices = [feature_cols_val.index(c) for c in common]
            X_train_raw = X_train_raw[:, train_indices]
            X_val_raw = X_val_raw[:, val_indices]
            feature_cols = common
        else:
            feature_cols = feature_cols_train
        if target_cols_train != target_cols_val:
            print(
                "WARNING: train/val target columns differ. "
                f"train={target_cols_train}, val={target_cols_val}"
            )
        target_cols = target_cols_train
        finite_mask = np.isfinite(X_train_raw).any(axis=0)
        if not bool(finite_mask.all()):
            dropped = [feature_cols[i] for i, ok in enumerate(finite_mask) if not ok]
            print(
                f"WARNING: dropping {len(dropped)} all-NaN train features before imputation "
                f"(e.g. {dropped[:3]})"
            )
            feature_cols = [feature_cols[i] for i, ok in enumerate(finite_mask) if ok]
            X_train_raw = X_train_raw[:, finite_mask]
            X_val_raw = X_val_raw[:, finite_mask]
        X_train_imp = self.imputer.fit_transform(X_train_raw)
        X_val_imp = self.imputer.transform(X_val_raw)
        X_train_scaled = self.scaler.fit_transform(X_train_imp)
        X_val_scaled = self.scaler.transform(X_val_imp)
        self.feature_names = feature_cols
        self.target_names = target_cols
        self.X_train = X_train_scaled
        self._X_train_imputed = X_train_imp
        self._X_val_imputed = X_val_imp
        if self.z_head == "quantile_v7":
            from src.models.z_quantile_v7 import resolve_z_driver_names

            self.z_driver_names = resolve_z_driver_names(feature_cols)
            print(f"V7 Z drivers: {len(self.z_driver_names)}")
        print(f"Base features: {len([c for c in self.required_features if c in feature_cols])}")
        print(f"Engineered features: {len([c for c in self.engineered_features if c in feature_cols])}")
        print(f"Cross features: {len([c for c in self.cross_features if c in feature_cols])}")
        print(f"Total features: {len(feature_cols)}")
        print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}")
        return X_train_scaled, X_val_scaled, y_train, y_val

    def prepare_training_data_fit(self, train_df):
        train_df = self._encode_frame(train_df, fit=True)
        out_train = self._build_feature_matrix(train_df)
        if out_train[0] is None:
            return None, None
        X_train_raw, y_train, feature_cols, target_cols = out_train
        if "sample_weight" in train_df.columns:
            self.train_sample_weights = (
                pd.to_numeric(train_df["sample_weight"], errors="coerce")
                .fillna(1.0)
                .astype(float)
                .values
            )
        else:
            self.train_sample_weights = None
        finite_mask = np.isfinite(X_train_raw).any(axis=0)
        if not bool(finite_mask.all()):
            dropped = [feature_cols[i] for i, ok in enumerate(finite_mask) if not ok]
            print(
                f"WARNING: dropping {len(dropped)} all-NaN train features before imputation "
                f"(e.g. {dropped[:3]})"
            )
            feature_cols = [feature_cols[i] for i, ok in enumerate(finite_mask) if ok]
            X_train_raw = X_train_raw[:, finite_mask]
        X_train_imp = self.imputer.fit_transform(X_train_raw)
        X_train_scaled = self.scaler.fit_transform(X_train_imp)
        self.feature_names = feature_cols
        self.target_names = target_cols
        self.X_train = X_train_scaled
        self._X_train_imputed = X_train_imp
        self._X_val_imputed = None
        if self.z_head == "quantile_v7":
            from src.models.z_quantile_v7 import resolve_z_driver_names

            self.z_driver_names = resolve_z_driver_names(feature_cols)
            print(f"V7 Z drivers: {len(self.z_driver_names)}")
        print(f"Base features: {len([c for c in self.required_features if c in feature_cols])}")
        print(f"Engineered features: {len([c for c in self.engineered_features if c in feature_cols])}")
        print(f"Cross features: {len([c for c in self.cross_features if c in feature_cols])}")
        print(f"Total features: {len(feature_cols)}")
        print(f"Train (fit_final, no val): {X_train_scaled.shape}")
        return X_train_scaled, y_train

    def load_base_models(self, target_name: str | None = None):
        print("\nLoading base models with optimal parameters...")
        if target_name:
            print(f"  Target profile: {target_name}")
        return make_base_models(
            estimator_profile=self.estimator_profile,
            target_name=target_name,
            loss_profile=self.loss_profile,
            estimator_overrides=self.estimator_overrides or None,
        )

    def _per_target_sample_weights(
        self,
        target_name: str,
        y_target: np.ndarray,
        base_weights: np.ndarray | None = None,
    ) -> np.ndarray | None:
        n = int(np.asarray(y_target).reshape(-1).size)
        mode = getattr(self, "yz_boost_mode", "default" if self.yz_target_boost else "off")
        if mode == "off" or not self.yz_target_boost:
            if base_weights is None:
                return None
            return np.asarray(base_weights, dtype=float).reshape(-1)
        if base_weights is not None:
            out = np.asarray(base_weights, dtype=float).reshape(-1)
        else:
            out = np.ones(n, dtype=float)
        abs_y = np.abs(np.asarray(y_target, dtype=float).reshape(-1))
        axis = target_name.split("_")[-1]
        if mode == "soft":
            if axis == "z":
                boost = 1.0 + np.clip(abs_y / 30.0, 0.0, 1.0)
            elif axis == "y":
                boost = 1.0 + np.clip(abs_y / 24.0, 0.0, 1.0)
            else:
                boost = np.ones_like(abs_y)
        elif mode == "y_only":
            if axis == "y":
                boost = 1.0 + np.clip(abs_y / 12.0, 0.0, 1.8)
            else:
                boost = np.ones_like(abs_y)
        else:  # default
            if axis == "z":
                boost = 1.0 + np.clip(abs_y / 15.0, 0.0, 2.5)
            elif axis == "y":
                boost = 1.0 + np.clip(abs_y / 12.0, 0.0, 1.8)
            else:
                boost = np.ones_like(abs_y)
        return out * boost

    @staticmethod
    def _sanitize_predictions(pred: np.ndarray, fallback: float) -> np.ndarray:
        return sanitize_predictions(pred, fallback)

    @staticmethod
    def _fit_kwargs_for_model(model_name: str, sample_weight: np.ndarray | None) -> dict:
        return fit_kwargs_for_model(model_name, sample_weight)

    def _optimize_weights_groupkfold(
        self,
        models,
        X_train,
        y_train,
        groups,
        target_name,
        sample_weight=None,
        *,
        n_splits: int | None = None,
    ) -> dict:
        averaged, traces, variance = average_groupkfold_weights(
            models,
            X_train,
            y_train,
            groups,
            target_name,
            sample_weight=sample_weight,
            n_splits=int(n_splits if n_splits is not None else getattr(self, "inner_n_splits", 3)),
            adaptive_priors=self.adaptive_weights.get(target_name, {}),
            verbose=True,
        )
        traces_map = getattr(self, "_inner_fold_weight_traces", None)
        if traces_map is None:
            self._inner_fold_weight_traces = {}
        self._inner_fold_weight_traces[target_name] = traces
        self._inner_weight_variance[target_name] = variance
        return averaged

    def optimize_ensemble_weights(
        self, models, X_train, y_train, X_val, y_val, target_name, sample_weight=None
    ):
        return tune_ensemble_weights(
            models,
            X_train,
            y_train,
            X_val,
            y_val,
            target_name,
            sample_weight=sample_weight,
            adaptive_priors=self.adaptive_weights.get(target_name, {}),
            verbose=True,
        )

    def _fit_voting_ensemble(self, ensemble, X, y, sample_weight=None) -> None:
        fit_voting_ensemble(ensemble, X, y, sample_weight=sample_weight)

    def _copy_model(self, model):
        return copy_estimator(model)

    def create_optimized_voting_ensemble(self, models, target_name, optimized_weights):
        return create_optimized_voting_ensemble(
            models,
            target_name,
            optimized_weights,
            adaptive_weights=self.adaptive_weights,
        )

    def create_adaptive_voting_ensemble(self, models, target_name):
        return create_adaptive_voting_ensemble(
            models, target_name, adaptive_weights=self.adaptive_weights
        )

    def create_standard_voting_ensemble(self, models):
        return create_standard_voting_ensemble(models)

    def evaluate_model_cv(self, model, X_train, y_train, model_name, sample_weight=None):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train, dtype=float).reshape(-1)
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
        n = len(y_train)
        n_splits = int(getattr(self.cv_splitter, "n_splits", 5))
        if n < max(4, n_splits * 2):
            est = copy_estimator(model)
            est.fit(X_train, y_train, **self._fit_kwargs_for_model(model_name, weights))
            pred = est.predict(X_train)
            mae = float(mean_absolute_error(y_train, pred, sample_weight=weights))
            print(f"  {model_name} CV MAE: {mae:.3f} +/- 0.000 (in-sample, n={n})")
            return mae, 0.0
        scores: list[float] = []
        for train_idx, test_idx in self.cv_splitter.split(X_train, y_train):
            est = copy_estimator(model)
            w_tr = None if weights is None else weights[train_idx]
            w_te = None if weights is None else weights[test_idx]
            est.fit(
                X_train[train_idx],
                y_train[train_idx],
                **self._fit_kwargs_for_model(model_name, w_tr),
            )
            pred = est.predict(X_train[test_idx])
            scores.append(
                float(mean_absolute_error(y_train[test_idx], pred, sample_weight=w_te))
            )
        arr = np.asarray(scores, dtype=float)
        cv_mae = float(arr.mean())
        cv_std = float(arr.std())
        print(f"  {model_name} CV MAE: {cv_mae:.3f} +/- {cv_std:.3f}")
        return cv_mae, cv_std

    def train_and_evaluate_adaptive_ensembles(
        self,
        X_train,
        X_test=None,
        y_train=None,
        y_test=None,
        sample_weight=None,
        groups=None,
        *,
        fast_weights: bool = False,
        fixed_prior_benchmark: bool = False,
        select_on_test: bool = False,
        compute_base_cv: bool | None = None,
        weight_mode: str | None = None,
    ):
        print("\nTraining and evaluating adaptive ensemble models...")
        if y_train is None:
            raise ValueError("y_train is required")
        if fast_weights:
            warnings.warn(
                "fast_weights=True is a fixed-prior benchmark, not production nested OOF. "
                "Use fit_final()/evaluate_fold() with inner GroupKFold weight tuning.",
                UserWarning,
                stacklevel=2,
            )
            fixed_prior_benchmark = True
        if weight_mode is None:
            if fixed_prior_benchmark:
                weight_mode = "fixed_prior"
            elif groups is not None and len(np.unique(groups)) >= 2:
                weight_mode = "inner_groupkfold"
            else:
                weight_mode = "random_split"
        if weight_mode == "inner_groupkfold" and (
            groups is None or len(np.unique(groups)) < 2
        ):
            print("inner GroupKFold requested but <2 groups; falling back to random_split")
            weight_mode = "random_split"
        if compute_base_cv is None:
            compute_base_cv = self.estimator_profile != "tiny" and X_test is not None

        has_test = (
            X_test is not None
            and y_test is not None
            and len(np.asarray(X_test)) > 0
        )
        n_test = int(len(np.asarray(X_test))) if has_test else 0
        if select_on_test and n_test < 2:
            raise ValueError(
                "Model selection by test R² requires at least 2 validation rows. "
                "Use fit_final() for 100% train (no dummy 1-row val) or "
                "evaluate_fold() which never selects on the outer fold."
            )

        weights = sample_weight
        if weights is None:
            weights = self.train_sample_weights
        if weights is not None:
            weights = np.asarray(weights, dtype=float).reshape(-1)
            if len(weights) != len(y_train):
                print(
                    f"WARNING: sample_weight length {len(weights)} != train rows {len(y_train)}; "
                    "ignoring weights."
                )
                weights = None
            else:
                print(
                    f"Using sample_weight: min={weights.min():.3f}, "
                    f"max={weights.max():.3f}, mean={weights.mean():.3f}"
                )

        if weight_mode == "fixed_prior":
            print("Weight tuning: FIXED PRIORS (benchmark only, not production nested OOF)")
        elif weight_mode == "inner_groupkfold":
            print(
                f"Weight tuning: inner GroupKFold on {len(np.unique(groups))} patients; "
                "final fit on 100% outer-train"
            )
        else:
            print("Weight tuning: random 80/20 split; final fit on 100% train")

        results = {}
        self._best_single_maes = {}
        self._optimized_weights = {}
        self._inner_fold_weight_traces = {}
        self._inner_weight_variance = {}
        test_shape = None if not has_test else X_test.shape
        print(f"Train={X_train.shape}, Test={test_shape}")

        for i, target_name in enumerate(self.target_names):
            print(f"\n{target_name}:")
            print("-" * 50)
            y_train_raw = np.asarray(y_train[:, i], dtype=float).reshape(-1)
            tr_mask = np.isfinite(y_train_raw)
            if int(tr_mask.sum()) < 2:
                print(f"  SKIP: <2 finite labels ({int(tr_mask.sum())})")
                continue
            y_train_full = y_train_raw[tr_mask]
            X_train_t = np.asarray(X_train)[tr_mask]
            w_base = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)[tr_mask]
            groups_t = None if groups is None else np.asarray(groups)[tr_mask]
            X_train_imp_t = (
                None
                if self._X_train_imputed is None
                else np.asarray(self._X_train_imputed)[tr_mask]
            )
            y_test_target = None
            X_test_t = None
            X_val_imp_t = None
            if has_test:
                y_test_raw = np.asarray(y_test[:, i], dtype=float).reshape(-1)
                te_mask = np.isfinite(y_test_raw)
                if int(te_mask.sum()) == 0:
                    has_test_t = False
                else:
                    has_test_t = True
                    y_test_target = y_test_raw[te_mask]
                    X_test_t = np.asarray(X_test)[te_mask]
                    X_val_imp_t = (
                        None
                        if self._X_val_imputed is None
                        else np.asarray(self._X_val_imputed)[te_mask]
                    )
            else:
                has_test_t = False
            w_full = self._per_target_sample_weights(target_name, y_train_full, w_base)
            fallback_pred = float(np.nanmedian(y_train_full)) if len(y_train_full) else 0.0
            if not np.isfinite(fallback_pred):
                fallback_pred = 0.0
            if int((~tr_mask).sum()):
                print(
                    f"  one-kidney/partial: using {int(tr_mask.sum())}/{len(tr_mask)} "
                    "finite labels for this target"
                )

            if (
                self.z_head == "quantile_v7"
                and target_name in {"kidney_left_delta_z", "kidney_right_delta_z"}
                and X_train_imp_t is not None
                and self.z_driver_names
            ):
                from src.models.z_quantile_v7 import fit_quantile_z, predict_quantile_z

                z_model, _ = fit_quantile_z(
                    X_train_imp_t,
                    self.feature_names,
                    y_train_full,
                    self.z_driver_names,
                    sample_weight=w_full,
                )
                self.trained_models[target_name] = z_model
                if not has_test_t or X_val_imp_t is None:
                    results[target_name] = {
                        "Best_Single_Model": "QuantileRegressor_v7",
                        "Improvement_Optimized_vs_Standard": 0.0,
                        "Improvement_Optimized_vs_Adaptive": 0.0,
                        "Improvement_Optimized_vs_Best": 0.0,
                    }
                    print(f"  V7 QuantileRegressor ({len(self.z_driver_names)} drivers) fit_final (no val)")
                    continue
                optimized_pred = self._sanitize_predictions(
                    predict_quantile_z(
                        z_model,
                        X_val_imp_t,
                        self.feature_names,
                        self.z_driver_names,
                    ),
                    fallback_pred,
                )
                optimized_mae = mean_absolute_error(y_test_target, optimized_pred)
                optimized_rmse = np.sqrt(mean_squared_error(y_test_target, optimized_pred))
                optimized_r2 = r2_score(y_test_target, optimized_pred)
                self._best_single_maes[target_name] = optimized_mae
                results[target_name] = {
                    "Optimized_MAE": optimized_mae,
                    "Optimized_RMSE": optimized_rmse,
                    "Optimized_R2": optimized_r2,
                    "Optimized_Median_AE": np.median(np.abs(y_test_target - optimized_pred)),
                    "Optimized_Error_5mm": np.mean(np.abs(y_test_target - optimized_pred) < 5) * 100,
                    "Optimized_Error_10mm": np.mean(np.abs(y_test_target - optimized_pred) < 10) * 100,
                    "Optimized_Max_Error": np.max(np.abs(y_test_target - optimized_pred)),
                    "Optimized_Outliers_20mm": np.sum(np.abs(y_test_target - optimized_pred) > 20),
                    "Optimized_Std_Error": np.std(y_test_target - optimized_pred),
                    "Adaptive_MAE": optimized_mae,
                    "Adaptive_RMSE": optimized_rmse,
                    "Adaptive_R2": optimized_r2,
                    "Adaptive_Median_AE": np.median(np.abs(y_test_target - optimized_pred)),
                    "Adaptive_Error_5mm": np.mean(np.abs(y_test_target - optimized_pred) < 5) * 100,
                    "Adaptive_Error_10mm": np.mean(np.abs(y_test_target - optimized_pred) < 10) * 100,
                    "Adaptive_Max_Error": np.max(np.abs(y_test_target - optimized_pred)),
                    "Adaptive_Outliers_20mm": np.sum(np.abs(y_test_target - optimized_pred) > 20),
                    "Adaptive_Std_Error": np.std(y_test_target - optimized_pred),
                    "Standard_MAE": optimized_mae,
                    "Standard_RMSE": optimized_rmse,
                    "Standard_R2": optimized_r2,
                    "Standard_Median_AE": np.median(np.abs(y_test_target - optimized_pred)),
                    "Standard_Error_5mm": np.mean(np.abs(y_test_target - optimized_pred) < 5) * 100,
                    "Standard_Error_10mm": np.mean(np.abs(y_test_target - optimized_pred) < 10) * 100,
                    "Standard_Max_Error": np.max(np.abs(y_test_target - optimized_pred)),
                    "Standard_Outliers_20mm": np.sum(np.abs(y_test_target - optimized_pred) > 20),
                    "Standard_Std_Error": np.std(y_test_target - optimized_pred),
                    "Best_Single_Model": "QuantileRegressor_v7",
                    "Improvement_Optimized_vs_Standard": 0.0,
                    "Improvement_Optimized_vs_Adaptive": 0.0,
                    "Improvement_Optimized_vs_Best": 0.0,
                }
                print(f"  V7 QuantileRegressor ({len(self.z_driver_names)} drivers) - MAE: {optimized_mae:.3f} mm, R2: {optimized_r2:.3f}")
                continue

            base_models = self.load_base_models(target_name)
            if self.model_kind in ("mean", "median") or self.model_kind in SINGLE_KIND_TO_NAME:
                est_name, single = make_single_estimator(self.model_kind, base_models)
                single.fit(
                    X_train_t,
                    y_train_full,
                    **self._fit_kwargs_for_model(est_name, w_full),
                )
                self.trained_models[target_name] = single
                self._optimized_weights[target_name] = {est_name: 1.0}
                self._inner_fold_weight_traces[target_name] = [{est_name: 1.0}]
                self._inner_weight_variance[target_name] = {est_name: 0.0}
                if not has_test_t:
                    results[target_name] = {
                        "Best_Single_Model": est_name,
                        "Improvement_Optimized_vs_Standard": 0.0,
                        "Improvement_Optimized_vs_Adaptive": 0.0,
                        "Improvement_Optimized_vs_Best": 0.0,
                    }
                    print(f"  {est_name} fit_final (single estimator)")
                    continue
                single_pred = self._sanitize_predictions(single.predict(X_test_t), fallback_pred)
                mae = mean_absolute_error(y_test_target, single_pred)
                rmse = float(np.sqrt(mean_squared_error(y_test_target, single_pred)))
                r2 = r2_score(y_test_target, single_pred)
                results[target_name] = {
                    "Optimized_MAE": mae,
                    "Optimized_RMSE": rmse,
                    "Optimized_R2": r2,
                    "Best_Single_Model": est_name,
                    "Improvement_Optimized_vs_Standard": 0.0,
                    "Improvement_Optimized_vs_Adaptive": 0.0,
                    "Improvement_Optimized_vs_Best": 0.0,
                }
                print(f"  {est_name} - MAE: {mae:.3f} mm, R2: {r2:.3f}")
                continue

            if weight_mode == "fixed_prior":
                priors = self.adaptive_weights.get(target_name, {})
                total = sum(priors.get(n, 1.0) for n in base_models.keys()) or len(base_models)
                optimized_weights = {n: priors.get(n, 1.0) / total for n in base_models.keys()}
            elif weight_mode == "inner_groupkfold":
                n_groups_t = 0 if groups_t is None else len(np.unique(groups_t))
                if n_groups_t < 2:
                    print("  inner GroupKFold: <2 groups after NaN mask; random_split")
                    if w_full is not None:
                        X_wt, X_wv, y_wt, y_wv, w_wt, _w_wv = train_test_split(
                            X_train_t, y_train_full, w_full, test_size=0.2, random_state=42
                        )
                    else:
                        X_wt, X_wv, y_wt, y_wv = train_test_split(
                            X_train_t, y_train_full, test_size=0.2, random_state=42
                        )
                        w_wt = None
                    optimized_weights = self.optimize_ensemble_weights(
                        base_models,
                        X_wt,
                        y_wt,
                        X_wv,
                        y_wv,
                        target_name,
                        sample_weight=w_wt,
                    )
                else:
                    optimized_weights = self._optimize_weights_groupkfold(
                        base_models,
                        X_train_t,
                        y_train_full,
                        groups_t,
                        target_name,
                        sample_weight=w_full,
                    )
            else:
                if w_full is not None:
                    X_wt, X_wv, y_wt, y_wv, w_wt, _w_wv = train_test_split(
                        X_train_t, y_train_full, w_full, test_size=0.2, random_state=42
                    )
                else:
                    X_wt, X_wv, y_wt, y_wv = train_test_split(
                        X_train_t, y_train_full, test_size=0.2, random_state=42
                    )
                    w_wt = None
                optimized_weights = self.optimize_ensemble_weights(
                    base_models,
                    X_wt,
                    y_wt,
                    X_wv,
                    y_wv,
                    target_name,
                    sample_weight=w_wt,
                )

            # Stage 6 ensemble blending modes (applied after base weight search).
            ewm = getattr(self, "ensemble_weight_mode", "optimized") or "optimized"
            if ewm == "equal" or weight_mode == "equal":
                optimized_weights = {n: 1.0 / len(base_models) for n in base_models.keys()}
            elif ewm == "fixed_prior":
                priors = self.adaptive_weights.get(target_name, {})
                total = sum(priors.get(n, 1.0) for n in base_models.keys()) or len(base_models)
                optimized_weights = {n: priors.get(n, 1.0) / total for n in base_models.keys()}
            elif ewm == "shrink":
                equal = {n: 1.0 / len(base_models) for n in base_models.keys()}
                optimized_weights = {
                    n: 0.5 * float(optimized_weights.get(n, 0.0)) + 0.5 * equal[n]
                    for n in base_models.keys()
                }
                tot = sum(optimized_weights.values()) or 1.0
                optimized_weights = {n: w / tot for n, w in optimized_weights.items()}
            elif ewm == "best_per_axis":
                # Keep only the single largest weight (winner-take-all per target).
                best_name = max(optimized_weights, key=optimized_weights.get)
                optimized_weights = {n: (1.0 if n == best_name else 0.0) for n in base_models.keys()}
            elif ewm == "stacking":
                # Non-negative ridge on member OOF preds is approximated by
                # shrink+optimized blend; full stacking uses stacker in tuner when available.
                equal = {n: 1.0 / len(base_models) for n in base_models.keys()}
                optimized_weights = {
                    n: 0.7 * float(optimized_weights.get(n, 0.0)) + 0.3 * equal[n]
                    for n in base_models.keys()
                }
                tot = sum(optimized_weights.values()) or 1.0
                optimized_weights = {n: w / tot for n, w in optimized_weights.items()}
            self._optimized_weights[target_name] = optimized_weights
            optimized_ensemble = self.create_optimized_voting_ensemble(
                base_models, target_name, optimized_weights
            )
            adaptive_ensemble = self.create_adaptive_voting_ensemble(base_models, target_name)
            standard_ensemble = self.create_standard_voting_ensemble(base_models)
            self._fit_voting_ensemble(optimized_ensemble, X_train_t, y_train_full, w_full)
            self._fit_voting_ensemble(adaptive_ensemble, X_train_t, y_train_full, w_full)
            self._fit_voting_ensemble(standard_ensemble, X_train_t, y_train_full, w_full)
            self.trained_models[target_name] = optimized_ensemble

            best_single_mae = float("inf")
            if compute_base_cv:
                print("  Base Models CV Performance:")
                for model_name in self.adaptive_weights[target_name].keys():
                    if model_name in base_models:
                        cv_mae, cv_std = self.evaluate_model_cv(
                            base_models[model_name],
                            X_train_t,
                            y_train_full,
                            model_name,
                            sample_weight=w_full,
                        )
                        if cv_mae < best_single_mae:
                            best_single_mae = cv_mae
            self._best_single_maes[target_name] = (
                None if not np.isfinite(best_single_mae) else best_single_mae
            )
            if not has_test_t:
                results[target_name] = {
                    "Best_Single_Model": self.best_models[target_name],
                    "Improvement_Optimized_vs_Standard": 0.0,
                    "Improvement_Optimized_vs_Adaptive": 0.0,
                    "Improvement_Optimized_vs_Best": 0.0,
                }
                print("  fit_final: kept optimized ensemble (no val / no R² selection)")
                continue

            optimized_pred = self._sanitize_predictions(
                optimized_ensemble.predict(X_test_t), fallback_pred
            )
            adaptive_pred = self._sanitize_predictions(
                adaptive_ensemble.predict(X_test_t), fallback_pred
            )
            standard_pred = self._sanitize_predictions(
                standard_ensemble.predict(X_test_t), fallback_pred
            )
            use_adaptive = not np.all(np.isfinite(optimized_pred))
            if select_on_test and not use_adaptive:
                opt_r2 = r2_score(y_test_target, optimized_pred)
                adp_r2 = r2_score(y_test_target, adaptive_pred)
                use_adaptive = opt_r2 < 0 and adp_r2 > opt_r2
            if use_adaptive:
                reason = (
                    "optimized unstable or worse than adaptive on test"
                    if select_on_test
                    else "optimized produced non-finite predictions"
                )
                print(f"  [INFO] Using adaptive ensemble for {target_name} ({reason})")
                self.trained_models[target_name] = adaptive_ensemble
                optimized_pred = adaptive_pred.copy()

            optimized_mae = mean_absolute_error(y_test_target, optimized_pred)
            optimized_rmse = np.sqrt(mean_squared_error(y_test_target, optimized_pred))
            optimized_r2 = r2_score(y_test_target, optimized_pred)
            adaptive_mae = mean_absolute_error(y_test_target, adaptive_pred)
            adaptive_rmse = np.sqrt(mean_squared_error(y_test_target, adaptive_pred))
            adaptive_r2 = r2_score(y_test_target, adaptive_pred)
            standard_mae = mean_absolute_error(y_test_target, standard_pred)
            standard_rmse = np.sqrt(mean_squared_error(y_test_target, standard_pred))
            standard_r2 = r2_score(y_test_target, standard_pred)
            results[target_name] = {
                "Optimized_MAE": optimized_mae,
                "Optimized_RMSE": optimized_rmse,
                "Optimized_R2": optimized_r2,
                "Optimized_Median_AE": np.median(np.abs(y_test_target - optimized_pred)),
                "Optimized_Error_5mm": np.mean(np.abs(y_test_target - optimized_pred) < 5) * 100,
                "Optimized_Error_10mm": np.mean(np.abs(y_test_target - optimized_pred) < 10) * 100,
                "Optimized_Max_Error": np.max(np.abs(y_test_target - optimized_pred)),
                "Optimized_Outliers_20mm": np.sum(np.abs(y_test_target - optimized_pred) > 20),
                "Optimized_Std_Error": np.std(y_test_target - optimized_pred),
                "Adaptive_MAE": adaptive_mae,
                "Adaptive_RMSE": adaptive_rmse,
                "Adaptive_R2": adaptive_r2,
                "Adaptive_Median_AE": np.median(np.abs(y_test_target - adaptive_pred)),
                "Adaptive_Error_5mm": np.mean(np.abs(y_test_target - adaptive_pred) < 5) * 100,
                "Adaptive_Error_10mm": np.mean(np.abs(y_test_target - adaptive_pred) < 10) * 100,
                "Adaptive_Max_Error": np.max(np.abs(y_test_target - adaptive_pred)),
                "Adaptive_Outliers_20mm": np.sum(np.abs(y_test_target - adaptive_pred) > 20),
                "Adaptive_Std_Error": np.std(y_test_target - adaptive_pred),
                "Standard_MAE": standard_mae,
                "Standard_RMSE": standard_rmse,
                "Standard_R2": standard_r2,
                "Standard_Median_AE": np.median(np.abs(y_test_target - standard_pred)),
                "Standard_Error_5mm": np.mean(np.abs(y_test_target - standard_pred) < 5) * 100,
                "Standard_Error_10mm": np.mean(np.abs(y_test_target - standard_pred) < 10) * 100,
                "Standard_Max_Error": np.max(np.abs(y_test_target - standard_pred)),
                "Standard_Outliers_20mm": np.sum(np.abs(y_test_target - standard_pred) > 20),
                "Standard_Std_Error": np.std(y_test_target - standard_pred),
                "Best_Single_Model": self.best_models[target_name],
                "Improvement_Optimized_vs_Standard": ((standard_mae - optimized_mae) / standard_mae) * 100,
                "Improvement_Optimized_vs_Adaptive": ((adaptive_mae - optimized_mae) / adaptive_mae) * 100,
                "Improvement_Optimized_vs_Best": (
                    ((self._get_best_single_mae(target_name) - optimized_mae) / self._get_best_single_mae(target_name)) * 100
                    if self._get_best_single_mae(target_name) is not None
                    else 0.0
                ),
            }
            print(f"  Optimized Ensemble - MAE: {optimized_mae:.3f} mm, R2: {optimized_r2:.3f}")
            print(f"    <5mm accuracy: {results[target_name]['Optimized_Error_5mm']:.1f}%, <10mm accuracy: {results[target_name]['Optimized_Error_10mm']:.1f}%")
            print(f"  Adaptive Ensemble - MAE: {adaptive_mae:.3f} mm, R2: {adaptive_r2:.3f}")
            print(f"  Standard Ensemble - MAE: {standard_mae:.3f} mm, R2: {standard_r2:.3f}")

        self.results = results
        return results

    def fit_final(self, X_train, y_train, sample_weight=None, groups=None, *, weight_mode: str = "inner_groupkfold"):
        if X_train is None or y_train is None:
            raise ValueError("fit_final requires X_train and y_train")
        return self.train_and_evaluate_adaptive_ensembles(
            X_train,
            None,
            y_train,
            None,
            sample_weight=sample_weight,
            groups=groups,
            select_on_test=False,
            compute_base_cv=False,
            weight_mode=weight_mode,
        )

    def evaluate_fold(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        sample_weight=None,
        groups=None,
        *,
        weight_mode: str = "inner_groupkfold",
    ):
        if X_test is None or y_test is None:
            raise ValueError("evaluate_fold requires a held-out X_test/y_test for scoring")
        return self.train_and_evaluate_adaptive_ensembles(
            X_train,
            X_test,
            y_train,
            y_test,
            sample_weight=sample_weight,
            groups=groups,
            select_on_test=False,
            compute_base_cv=False,
            weight_mode=weight_mode,
        )

    def _get_best_single_mae(self, target_name):
        return getattr(self, "_best_single_maes", {}).get(target_name, None)

    def generate_report(self):
        print("\n" + "=" * 80)
        print("OPTIMIZED ADAPTIVE ENSEMBLE MODELS - PHASE 1 REPORT")
        print("=" * 80)
        print(f"\nDataset Summary:")
        print(f"- Features used: {len(self.feature_names)}")
        print(f"- Target variables: {len(self.target_names)}")
        if not self.results:
            return
        optimized_mae = np.mean([r["Optimized_MAE"] for r in self.results.values() if "Optimized_MAE" in r])
        print(f"Average optimized MAE: {optimized_mae:.3f} mm")

    def save_model(self, filepath="models/adaptive_ensemble.pkl"):
        return save_trainer(self, filepath)

    def save_results(self, filename="adaptive_ensemble_integrated_results.csv"):
        rows = []
        for target_name, metrics in self.results.items():
            if "Optimized_MAE" not in metrics:
                continue
            rows.append(
                {
                    "Target": target_name,
                    "Model": "Optimized_Voting_Ensemble_Integrated",
                    "MAE": metrics["Optimized_MAE"],
                    "RMSE": metrics["Optimized_RMSE"],
                    "R2": metrics["Optimized_R2"],
                    "Features_Count": len(self.feature_names),
                }
            )
        if not rows:
            print("No results to save")
            return
        pd.DataFrame(rows).to_csv(filename, index=False)
        print(f"Results saved to {filename}")


def main():
    trainer = AdaptiveEnsembleTrainer()
    print("OPTIMIZED ADAPTIVE ENSEMBLE TRAINING WITH INTEGRATED DATA SOURCES")
    combined_df, train_df, val_df = trainer.load_integrated_data()
    if train_df is None:
        return
    prepared = trainer.prepare_training_data_split(train_df, val_df)
    if prepared[0] is None:
        return
    X_train, X_val, y_train, y_val = prepared
    trainer.train_and_evaluate_adaptive_ensembles(X_train, X_val, y_train, y_val)
    trainer.generate_report()
    trainer.save_model()


if __name__ == "__main__":
    main()
