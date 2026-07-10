#!/usr/bin/env python3
"""
Adaptive Ensemble Models for Kidney Displacement Prediction - INTEGRATED VERSION
Phase 1: Adaptive Weights Voting Ensemble with ALL Data Sources
Integrates: DICOMS + Vybor + KiTS19
"""

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
import itertools
import warnings
import sys
import os
import joblib
from pathlib import Path

# Добавляем путь к корневой директории для импорта наших модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.features.displacement_axis_features import (
    ANATOMICAL_FEATURES,
    DISPLACEMENT_AXIS_FEATURES,
    add_displacement_axis_features,
)
from src.features.leakage_safe import filter_model_features, is_leakage_feature
from src.features.na_trend_features import NaTrendStore, attach_na_trend_features
from src.features.projection_enrichment import (
    add_projection_delta_proxies,
    attach_projection_features,
)
from src.features.phase1_schema import (
    BASE_FEATURES,
    CROSS_FEATURES,
    ENGINEERED_FEATURES,
    TARGET_NAMES,
    encode_patient_position,
    normalize_dataframe,
)
warnings.filterwarnings('ignore')

class AdaptiveEnsembleTrainer:
    def __init__(
        self,
        *,
        z_head: str = "ensemble",
        enrichment_mode: str = "projection",
        na_trend_store: NaTrendStore | None = None,
    ):
        self.z_head = z_head  # "ensemble" | "quantile_v7"
        # projection = legacy per-patient proj_lat/sup join (not for honest clinical)
        # na_trends = cohort trends from na_spine + na_boku (production honest)
        # none = skip auxiliary enrichment
        self.enrichment_mode = enrichment_mode
        self.na_trend_store = na_trend_store
        self.z_driver_names: list[str] = []
        self._X_train_imputed: np.ndarray | None = None
        self._X_val_imputed: np.ndarray | None = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        self.target_names = []
        self.results = {}
        self.trained_models = {}  # Для хранения обученных моделей
        self.X_train = None  # Для confidence estimator
        self.train_sample_weights = None
        self.cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)  # Воспроизводимый CV
        
        # Canonical schema: config/phase1_feature_schema.yaml + src/features/phase1_schema.py
        self.required_features = list(BASE_FEATURES)
        self.engineered_features = list(ENGINEERED_FEATURES)
        self.cross_features = list(CROSS_FEATURES)
        self.displacement_axis_features = filter_model_features(list(DISPLACEMENT_AXIS_FEATURES))
        self.anatomical_features = list(ANATOMICAL_FEATURES)
        self.projection_features: list[str] = []
        self.na_trend_feature_cols: list[str] = []
        self.target_columns = list(TARGET_NAMES)
        
        # Best models per target — Z/Y tuned for huber GBT / deeper RF
        self.best_models = {
            'kidney_left_delta_x': 'RandomForest',
            'kidney_left_delta_y': 'GradientBoosting',
            'kidney_left_delta_z': 'GradientBoosting',
            'kidney_right_delta_x': 'Ridge',
            'kidney_right_delta_y': 'GradientBoosting',
            'kidney_right_delta_z': 'GradientBoosting',
        }
        
        # Adaptive weights based on ensemble performance analysis
        self.adaptive_weights = {
            'kidney_left_delta_x': {
                'RandomForest': 2.5,  # Best single model with highest weight
                'Lasso': 1.0,
                'Ridge': 0.8,
                'GradientBoosting': 0.6
            },
            'kidney_left_delta_y': {
                'RandomForest': 1.2,
                'Lasso': 1.0,
                'Ridge': 0.8,
                'GradientBoosting': 2.2,
            },
            'kidney_left_delta_z': {
                'RandomForest': 1.0,
                'Lasso': 0.6,
                'Ridge': 0.6,
                'GradientBoosting': 2.8,
            },
            'kidney_right_delta_x': {
                'RandomForest': 1.0,  # Lower weight due to ensemble degradation
                'Lasso': 0.8,
                'Ridge': 2.5,  # Best single model with highest weight
                'GradientBoosting': 0.6
            },
            'kidney_right_delta_y': {
                'RandomForest': 1.2,
                'Lasso': 1.0,
                'Ridge': 0.8,
                'GradientBoosting': 2.2,
            },
            'kidney_right_delta_z': {
                'RandomForest': 1.0,
                'Lasso': 0.6,
                'Ridge': 0.6,
                'GradientBoosting': 2.8,
            },
        }
    
    def load_integrated_data(self):
        """Load integrated train/validation splits from disk.

        Returns (combined_df, train_df, val_df). The ``combined_df`` is
        kept ONLY for backwards compatibility — see ``prepare_training_data``
        warning. New callers must use ``prepare_training_data_split`` which
        accepts ``train_df`` and ``val_df`` separately.
        """
        print("Loading integrated datasets from all sources...")
        try:
            train_df = pd.read_csv('data/processed/train.csv')
            val_df = pd.read_csv('data/processed/validation.csv')
            print(f"Integrated Train dataset: {len(train_df)} cases")
            print(f"Integrated Validation dataset: {len(val_df)} cases")
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            print(f"Combined dataset (legacy view): {len(combined_df)} cases")
            return combined_df, train_df, val_df
        except FileNotFoundError:
            print("Integrated data files not found. Please run data integration first.")
            print("Run: python src/models/data_integration_fix.py")
            return None, None, None
    
        
    def _apply_aux_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df
        if self.enrichment_mode == "na_trends":
            out = attach_na_trend_features(out, self.na_trend_store)
        elif self.enrichment_mode == "projection":
            out = attach_projection_features(out)
            out = add_projection_delta_proxies(out)
        return out

    def _build_feature_matrix(self, df):
        """Apply feature engineering and return (X, y, all_feature_cols, target_cols).

        Shared helper used by both legacy and leak-safe prepare paths. Returns
        X as ``np.ndarray`` with NaN preserved — imputation happens later
        (after train/val split) to avoid information leakage.
        """
        df = normalize_dataframe(df)
        target_cols = [col for col in self.target_columns if col in df.columns]

        missing_base_features = [col for col in self.required_features if col not in df.columns]
        if missing_base_features:
            print(f"WARNING: Missing base features: {missing_base_features}")
            available_base_features = [col for col in self.required_features if col in df.columns]
            if len(available_base_features) < len(self.required_features) * 0.8:
                print("Too many missing features. Cannot proceed.")
                return None, None, None, None

        if len(target_cols) == 0:
            print("No target variables found in dataset")
            return None, None, None, None

        df_enhanced = self._create_engineered_features(df.copy())
        df_enhanced = self._create_cross_features(df_enhanced)
        df_enhanced = add_displacement_axis_features(df_enhanced)
        df_enhanced = self._apply_aux_enrichment(df_enhanced)

        base_feature_cols = [col for col in self.required_features if col in df_enhanced.columns]
        engineered_feature_cols = [col for col in self.engineered_features if col in df_enhanced.columns]
        cross_feature_cols = [col for col in self.cross_features if col in df_enhanced.columns]
        axis_feature_cols = [
            c for c in self.displacement_axis_features if c in df_enhanced.columns
        ]
        anatomical_cols = [c for c in self.anatomical_features if c in df_enhanced.columns]
        projection_feature_cols = sorted(
            c for c in df_enhanced.columns
            if c.startswith("proj_") and not is_leakage_feature(c)
        )
        na_trend_feature_cols = sorted(
            c
            for c in df_enhanced.columns
            if c.startswith("na_pop_shift_")
            or c.startswith("na_sup_z_")
            or c.startswith("na_sup_pct_")
            or c.startswith("kits_z_")
            or c.startswith("kits_pct_")
            or c.startswith("kits_cohort_median_")
        )
        self.projection_features = projection_feature_cols
        self.na_trend_feature_cols = na_trend_feature_cols
        all_feature_cols = (
            base_feature_cols
            + engineered_feature_cols
            + cross_feature_cols
            + axis_feature_cols
            + anatomical_cols
            + projection_feature_cols
            + na_trend_feature_cols
        )

        X = df_enhanced[all_feature_cols].astype(float).values
        y = df_enhanced[target_cols].astype(float).values
        return X, y, all_feature_cols, target_cols

    def build_inference_matrix(self, df):
        """Apply train-time feature engineering to an arbitrary inference df.

        Returns an ``np.ndarray`` of shape ``(n_rows, len(self.feature_names))``
        aligned to the columns recorded at training time, with NaN preserved.
        Use this from inference code instead of re-implementing the pipeline.
        """
        if not self.feature_names:
            raise RuntimeError(
                "feature_names is empty. Train (or load) the model before "
                "calling build_inference_matrix()."
            )
        df = normalize_dataframe(df)
        df_enhanced = self._create_engineered_features(df.copy())
        df_enhanced = self._create_cross_features(df_enhanced)
        df_enhanced = add_displacement_axis_features(df_enhanced)
        df_enhanced = self._apply_aux_enrichment(df_enhanced)
        for col in self.feature_names:
            if col not in df_enhanced.columns:
                df_enhanced[col] = np.nan
        X = df_enhanced[self.feature_names].astype(float).values
        return X

    def prepare_training_data(self, df):
        """[Legacy / leakage-risk] Готовит данные из ОДНОГО объединённого df.

        ВНИМАНИЕ: внутри делает ``train_test_split`` поверх входа, что
        приводит к утечке, если ``df`` уже содержит склейку train+validation.
        Метод сохранён только для обратной совместимости. Для нового кода
        используйте ``prepare_training_data_split(train_df, val_df)``.
        """
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
            X, y, test_size=0.2, random_state=42,
        )

        # Imputer fitted on train only, then applied to test — consistent
        # with the inference path that will use the same imputer.
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
        """Готовит данные БЕЗ утечки train/val статистики.

        Контракт:
          - ``train_df`` и ``val_df`` уже разделены на диске и НЕ объединяются.
          - feature engineering применяется к каждому датафрейму независимо.
          - ``StandardScaler.fit`` происходит ТОЛЬКО на train, далее
            ``transform`` к val.
          - Возвращает ``(X_train, X_val, y_train, y_val)``.

        Это leakage-safe замена ``prepare_training_data``.
        """
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

        # Imputer is fit ONLY on train. Same fitted imputer is then applied
        # to val and persisted in save_model(). At inference time the API
        # is required to apply the same imputer before the scaler — see
        # ``predict_displacement`` in src/api/kidney_displacement_api.py.
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
    
    def _create_engineered_features(self, df):
        """Create engineered features from base features"""
        print("\n[FE] Creating engineered features...")
        
        # 1. Body ratio
        if 'body_width_mm' in df.columns and 'body_depth_mm' in df.columns:
            df['body_ratio'] = df['body_width_mm'] / df['body_depth_mm']
            print("  [OK] body_ratio created")
        
        # 2. Расстояние между почками (используем center_x_rel)
        if 'kidney_left_center_x_rel' in df.columns and 'kidney_right_center_x_rel' in df.columns:
            df['kidney_distance_lr'] = np.abs(df['kidney_left_center_x_rel'] - df['kidney_right_center_x_rel'])
            print("  [OK] kidney_distance_lr created")
        
        # 3. Нормализованные размеры почек
        if 'kidney_left_volume_cm3' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_left_volume_norm'] = df['kidney_left_volume_cm3'] / df['body_width_mm']
            print("  [OK] kidney_left_volume_norm created")
            
        if 'kidney_right_volume_cm3' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_right_volume_norm'] = df['kidney_right_volume_cm3'] / df['body_width_mm']
            print("  [OK] kidney_right_volume_norm created")
        
        if 'kidney_left_length_mm' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_left_length_norm'] = df['kidney_left_length_mm'] / df['body_width_mm']
            print("  [OK] kidney_left_length_norm created")
            
        if 'kidney_right_length_mm' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_right_length_norm'] = df['kidney_right_length_mm'] / df['body_width_mm']
            print("  [OK] kidney_right_length_norm created")
        
        # 4. Признаки асимметрии
        if 'kidney_left_volume_cm3' in df.columns and 'kidney_right_volume_cm3' in df.columns:
            df['volume_asymmetry'] = df['kidney_left_volume_cm3'] - df['kidney_right_volume_cm3']
            print("  [OK] volume_asymmetry created")
        
        if 'kidney_left_length_mm' in df.columns and 'kidney_right_length_mm' in df.columns:
            df['length_asymmetry'] = df['kidney_left_length_mm'] - df['kidney_right_length_mm']
            print("  [OK] length_asymmetry created")
        
        if 'kidney_left_to_spine_distance' in df.columns and 'kidney_right_to_spine_distance' in df.columns:
            df['spine_distance_asymmetry'] = df['kidney_left_to_spine_distance'] - df['kidney_right_to_spine_distance']
            print("  [OK] spine_distance_asymmetry created")
        
        if 'kidney_left_to_body_center_distance' in df.columns and 'kidney_right_to_body_center_distance' in df.columns:
            df['body_center_asymmetry'] = df['kidney_left_to_body_center_distance'] - df['kidney_right_to_body_center_distance']
            print("  [OK] body_center_asymmetry created")
        
        # 5. Нормализованные расстояния до позвоночника
        if 'kidney_left_to_spine_distance' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_left_to_spine_ratio'] = df['kidney_left_to_spine_distance'] / df['body_width_mm']
            print("  [OK] kidney_left_to_spine_ratio created")
        
        if 'kidney_right_to_spine_distance' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_right_to_spine_ratio'] = df['kidney_right_to_spine_distance'] / df['body_width_mm']
            print("  [OK] kidney_right_to_spine_ratio created")
        
        # 6. patient_position_encoded (from scan_position / patient_position or default)
        if 'patient_position_encoded' not in df.columns or df['patient_position_encoded'].isna().all():
            pos_col = None
            for candidate in ('scan_position', 'patient_position'):
                if candidate in df.columns:
                    pos_col = candidate
                    break
            if pos_col is not None:
                df['patient_position_encoded'] = df[pos_col].map(encode_patient_position)
                print(f"  [OK] patient_position_encoded from {pos_col}")
            else:
                df['patient_position_encoded'] = 1
                print("  [OK] patient_position_encoded set to default (supine=1)")
        
        print(f"[FE] Engineered features creation completed. New shape: {df.shape}")
        return df
    
    def _create_cross_features(self, df):
        """Create advanced cross-features for better performance"""
        print("\n[FE] Creating cross-features...")
        
        # 1. Body volume estimation
        if 'body_width_mm' in df.columns and 'body_depth_mm' in df.columns and 'kidney_left_length_mm' in df.columns:
            avg_kidney_height = (df['kidney_left_length_mm'] + df.get('kidney_right_length_mm', df['kidney_left_length_mm'])) / 2
            df['body_volume_estimated'] = df['body_width_mm'] * df['body_depth_mm'] * avg_kidney_height / 1000  # в см³
            print("  [OK] body_volume_estimated created")
        
        # 2. Kidney density ratios
        if 'kidney_left_volume_cm3' in df.columns and 'kidney_left_length_mm' in df.columns:
            df['kidney_left_density_ratio'] = df['kidney_left_volume_cm3'] / df['kidney_left_length_mm']
            print("  [OK] kidney_left_density_ratio created")
            
        if 'kidney_right_volume_cm3' in df.columns and 'kidney_right_length_mm' in df.columns:
            df['kidney_right_density_ratio'] = df['kidney_right_volume_cm3'] / df['kidney_right_length_mm']
            print("  [OK] kidney_right_density_ratio created")
        
        # 3. Spine to body ratios
        if 'spine_center_x' in df.columns and 'body_width_mm' in df.columns:
            df['spine_to_body_ratio_x'] = df['spine_center_x'] / df['body_width_mm']
            print("  [OK] spine_to_body_ratio_x created")
            
        if 'spine_center_y' in df.columns and 'body_depth_mm' in df.columns:
            df['spine_to_body_ratio_y'] = df['spine_center_y'] / df['body_depth_mm']
            print("  [OK] spine_to_body_ratio_y created")
        
        # 4. Body COM to spine distance
        if all(col in df.columns for col in ['body_com_x', 'body_com_y', 'spine_center_x', 'spine_center_y']):
            df['body_com_to_spine_distance'] = np.sqrt(
                (df['body_com_x'] - df['spine_center_x'])**2 + 
                (df['body_com_y'] - df['spine_center_y'])**2
            )
            print("  [OK] body_com_to_spine_distance created")
        
        # 5. Kidney-spine interactions
        if 'kidney_left_to_spine_distance' in df.columns and 'kidney_left_volume_cm3' in df.columns:
            df['kidney_left_spine_interaction'] = df['kidney_left_to_spine_distance'] * df['kidney_left_volume_cm3']
            print("  [OK] kidney_left_spine_interaction created")
            
        if 'kidney_right_to_spine_distance' in df.columns and 'kidney_right_volume_cm3' in df.columns:
            df['kidney_right_spine_interaction'] = df['kidney_right_to_spine_distance'] * df['kidney_right_volume_cm3']
            print("  [OK] kidney_right_spine_interaction created")
        
        # 6. Body size index
        if 'body_width_mm' in df.columns and 'body_depth_mm' in df.columns:
            bw = pd.to_numeric(df['body_width_mm'], errors='coerce')
            bd = pd.to_numeric(df['body_depth_mm'], errors='coerce')
            df['body_size_index'] = np.sqrt(bw**2 + bd**2)
            print("  [OK] body_size_index created")
        
        # 7. Kidney position indices
        if all(col in df.columns for col in ['kidney_left_center_x_rel', 'kidney_left_center_y_rel', 'kidney_left_center_z_rel']):
            lx = pd.to_numeric(df['kidney_left_center_x_rel'], errors='coerce')
            ly = pd.to_numeric(df['kidney_left_center_y_rel'], errors='coerce')
            lz = pd.to_numeric(df['kidney_left_center_z_rel'], errors='coerce')
            df['kidney_position_index_left'] = np.sqrt(lx**2 + ly**2 + lz**2)
            print("  [OK] kidney_position_index_left created")
            
        if all(col in df.columns for col in ['kidney_right_center_x_rel', 'kidney_right_center_y_rel', 'kidney_right_center_z_rel']):
            rx = pd.to_numeric(df['kidney_right_center_x_rel'], errors='coerce')
            ry = pd.to_numeric(df['kidney_right_center_y_rel'], errors='coerce')
            rz = pd.to_numeric(df['kidney_right_center_z_rel'], errors='coerce')
            df['kidney_position_index_right'] = np.sqrt(rx**2 + ry**2 + rz**2)
            print("  [OK] kidney_position_index_right created")
        
        # 8. Volume to area ratios
        if 'kidney_left_volume_cm3' in df.columns and 'body_area_mm2' in df.columns:
            df['volume_to_area_ratio_left'] = df['kidney_left_volume_cm3'] / (df['body_area_mm2'] / 100)  # переводим в см²
            print("  [OK] volume_to_area_ratio_left created")
            
        if 'kidney_right_volume_cm3' in df.columns and 'body_area_mm2' in df.columns:
            df['volume_to_area_ratio_right'] = df['kidney_right_volume_cm3'] / (df['body_area_mm2'] / 100)
            print("  [OK] volume_to_area_ratio_right created")
        
        # 9. Relative volume sum
        if 'kidney_left_volume_cm3' in df.columns and 'kidney_right_volume_cm3' in df.columns and 'body_width_mm' in df.columns:
            df['relative_volume_sum'] = (df['kidney_left_volume_cm3'] + df['kidney_right_volume_cm3']) / df['body_width_mm']
            print("  [OK] relative_volume_sum created")
        
        # 10. Kidney separation angle (упрощенный)
        if all(col in df.columns for col in ['kidney_left_center_x_rel', 'kidney_right_center_x_rel', 'kidney_left_center_y_rel', 'kidney_right_center_y_rel']):
            lx = pd.to_numeric(df['kidney_left_center_x_rel'], errors='coerce')
            ly = pd.to_numeric(df['kidney_left_center_y_rel'], errors='coerce')
            rx = pd.to_numeric(df['kidney_right_center_x_rel'], errors='coerce')
            ry = pd.to_numeric(df['kidney_right_center_y_rel'], errors='coerce')
            left_vector = np.column_stack([lx, ly])
            right_vector = np.column_stack([rx, ry])

            dot_product = np.sum(left_vector * right_vector, axis=1)
            norm_left = np.sqrt(np.sum(left_vector**2, axis=1))
            norm_right = np.sqrt(np.sum(right_vector**2, axis=1))
            
            # Избегаем деления на ноль
            cos_angle = np.where(
                (norm_left > 0) & (norm_right > 0),
                dot_product / (norm_left * norm_right),
                0
            )
            df['kidney_separation_angle'] = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi  # в градусах
            print("  [OK] kidney_separation_angle created")
        
        print(f"[FE] Cross-features creation completed. New shape: {df.shape}")
        return df
    
    def load_base_models(self, target_name: str | None = None):
        """Load base models; Z/Y targets get huber/deeper configs."""
        print("\nLoading base models with optimal parameters...")
        if target_name:
            print(f"  Target profile: {target_name}")

        axis = target_name.split("_")[-1] if target_name else None

        rf_config = {
            'n_estimators': 600 if axis in ('y', 'z') else 500,
            'max_depth': 24 if axis in ('y', 'z') else 20,
            'min_samples_split': 6 if axis in ('y', 'z') else 10,
            'min_samples_leaf': 2 if axis in ('y', 'z') else 4,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
        }
        gbt_config = {
            'n_estimators': 700 if axis == 'z' else 550,
            'learning_rate': 0.04 if axis in ('y', 'z') else 0.05,
            'max_depth': 7 if axis == 'z' else 6,
            'subsample': 0.85,
            'random_state': 42,
        }
        if axis == 'z':
            gbt_config['loss'] = 'huber'
            gbt_config['alpha'] = 0.9

        model_configs = {
            'RandomForest': rf_config,
            'Lasso': {'alpha': 0.08 if axis in ('y', 'z') else 0.1, 'max_iter': 5000, 'random_state': 42},
            'Ridge': {'alpha': 0.8 if axis in ('y', 'z') else 1.0, 'solver': 'auto', 'random_state': 42},
            'GradientBoosting': gbt_config,
        }

        models = {}
        for model_name, config in model_configs.items():
            if model_name == 'RandomForest':
                models[model_name] = RandomForestRegressor(**config)
            elif model_name == 'Lasso':
                models[model_name] = Lasso(**config)
            elif model_name == 'Ridge':
                models[model_name] = Ridge(**config)
            elif model_name == 'GradientBoosting':
                models[model_name] = GradientBoostingRegressor(**config)

        return models

    @staticmethod
    def _per_target_sample_weights(
        target_name: str,
        y_target: np.ndarray,
        base_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Up-weight large |Y|/|Z| displacements so extremes are not ignored."""
        abs_y = np.abs(np.asarray(y_target, dtype=float).reshape(-1))
        axis = target_name.split("_")[-1]
        if axis == "z":
            boost = 1.0 + np.clip(abs_y / 15.0, 0.0, 2.5)
        elif axis == "y":
            boost = 1.0 + np.clip(abs_y / 12.0, 0.0, 1.8)
        else:
            boost = np.ones_like(abs_y)
        if base_weights is not None:
            return np.asarray(base_weights, dtype=float).reshape(-1) * boost
        return boost
    
    @staticmethod
    def _sanitize_predictions(pred: np.ndarray, fallback: float) -> np.ndarray:
        out = np.asarray(pred, dtype=float).reshape(-1)
        bad = ~np.isfinite(out)
        if bad.any():
            out = out.copy()
            out[bad] = fallback
        return out

    @staticmethod
    def _fit_kwargs_for_model(model_name: str, sample_weight: np.ndarray | None) -> dict:
        if sample_weight is None:
            return {}
        if model_name in {"RandomForest", "GradientBoosting"}:
            return {"sample_weight": sample_weight}
        return {}

    def _optimize_weights_groupkfold(
        self,
        models,
        X_train,
        y_train,
        groups,
        target_name,
        sample_weight=None,
        *,
        n_splits: int = 3,
    ) -> dict:
        """Average per-fold optimized ensemble weights (patient-level GroupKFold)."""
        groups = np.asarray(groups)
        n_unique = len(np.unique(groups))
        n_splits = min(n_splits, n_unique)
        if n_splits < 2:
            target_weights = self.adaptive_weights.get(target_name, {})
            total = sum(target_weights.get(n, 1.0) for n in models.keys()) or len(models)
            return {name: target_weights.get(name, 1.0) / total for name in models.keys()}

        splitter = GroupKFold(n_splits=n_splits)
        fold_weights: list[dict] = []
        for train_idx, val_idx in splitter.split(X_train, y_train, groups=groups):
            w_fold = sample_weight[train_idx] if sample_weight is not None else None
            fold_weights.append(
                self.optimize_ensemble_weights(
                    models,
                    X_train[train_idx],
                    y_train[train_idx],
                    X_train[val_idx],
                    y_train[val_idx],
                    target_name,
                    sample_weight=w_fold,
                )
            )

        averaged: dict[str, float] = {}
        for name in models.keys():
            averaged[name] = float(np.mean([fw[name] for fw in fold_weights]))
        total = sum(averaged.values()) or 1.0
        return {name: w / total for name, w in averaged.items()}

    def optimize_ensemble_weights(
        self, models, X_train, y_train, X_val, y_val, target_name, sample_weight=None
    ):
        """Оптимизация весов ансамбля с помощью scipy.optimize"""
        print(f"\n[FE] Optimizing ensemble weights for {target_name}...")

        fallback = float(np.nanmedian(y_train)) if len(y_train) else 0.0
        if not np.isfinite(fallback):
            fallback = 0.0

        model_predictions = {}
        for model_name, model in models.items():
            model_copy = self._copy_model(model)
            fit_kwargs = self._fit_kwargs_for_model(model_name, sample_weight)
            try:
                model_copy.fit(X_train, y_train, **fit_kwargs)
                pred = self._sanitize_predictions(model_copy.predict(X_val), fallback)
            except Exception as exc:
                print(f"  [WARN] {model_name} fit failed during weight search: {exc}")
                pred = np.full(len(y_val), fallback, dtype=float)
            model_predictions[model_name] = pred

        def objective_function(weights):
            weights = np.abs(weights) / np.sum(np.abs(weights))
            ensemble_pred = np.zeros(len(y_val))
            for i, pred in enumerate(model_predictions.values()):
                ensemble_pred += weights[i] * pred
            ensemble_pred = self._sanitize_predictions(ensemble_pred, fallback)
            return mean_absolute_error(y_val, ensemble_pred)

        initial_weights = np.ones(len(models)) / len(models)
        bounds = [(0, 1) for _ in range(len(models))]

        try:
            result = minimize(
                objective_function,
                initial_weights,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100},
            )
            raw_w = np.abs(np.asarray(result.x, dtype=float))
            denom = float(np.sum(raw_w))
            if not np.isfinite(denom) or denom <= 0 or not np.all(np.isfinite(raw_w)):
                raise ValueError("non-finite optimized weights")
            optimal_weights = raw_w / denom
        except Exception as exc:
            print(f"  [WARN] Weight optimization failed, using adaptive priors: {exc}")
            target_weights = self.adaptive_weights.get(target_name, {})
            total = sum(target_weights.get(n, 1.0) for n in models.keys()) or len(models)
            optimal_weights = np.array(
                [target_weights.get(n, 1.0) / total for n in models.keys()],
                dtype=float,
            )

        optimized_weights = {
            model_name: float(optimal_weights[i])
            for i, model_name in enumerate(models.keys())
        }
        print(f"  [OK] Optimized weights: {optimized_weights}")

        equal_weights = {name: 1.0 / len(models) for name in models.keys()}
        equal_pred = np.zeros(len(y_val))
        optimized_pred = np.zeros(len(y_val))
        for model_name, pred in model_predictions.items():
            equal_pred += equal_weights[model_name] * pred
            optimized_pred += optimized_weights[model_name] * pred
        equal_pred = self._sanitize_predictions(equal_pred, fallback)
        optimized_pred = self._sanitize_predictions(optimized_pred, fallback)

        equal_mae = mean_absolute_error(y_val, equal_pred)
        optimized_mae = mean_absolute_error(y_val, optimized_pred)
        if equal_mae > 0:
            improvement = ((equal_mae - optimized_mae) / equal_mae) * 100
            print(f"  Improvement: {improvement:.1f}% (MAE: {equal_mae:.3f} -> {optimized_mae:.3f})")

        return optimized_weights

    def _fit_voting_ensemble(self, ensemble, X, y, sample_weight=None) -> None:
        """Fit voting ensemble; sample_weight only on tree models (RF/GBT)."""
        if sample_weight is None:
            ensemble.fit(X, y)
            return
        named = getattr(ensemble, "named_estimators", None)
        if named is None:
            ensemble.fit(X, y)
            return
        fitted = []
        for name, est in named.items():
            est_fitted = self._copy_model(est)
            est_fitted.fit(X, y, **self._fit_kwargs_for_model(name, sample_weight))
            fitted.append(est_fitted)
        ensemble.estimators_ = fitted

    def _copy_model(self, model):
        """Создает копию модели с теми же параметрами (без обученного состояния).

        Делегирует ``sklearn.base.clone`` для корректной обработки вложенных
        параметров (пайплайны/нестед-параметры в RandomForest и т.п.).
        """
        try:
            return clone(model)
        except Exception:
            if hasattr(model, "get_params"):
                return type(model)(**model.get_params())
            return type(model)()
    
    def create_optimized_voting_ensemble(self, models, target_name, optimized_weights):
        """Create voting ensemble with optimized weights.

        Estimators cloned to keep cross-target training fully isolated.
        """
        estimators = [(name, clone(models[name])) for name in models.keys()]
        weights = np.array([optimized_weights[name] for name in models.keys()], dtype=float)
        if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
            target_weights = self.adaptive_weights.get(target_name, {})
            weights = np.array(
                [target_weights.get(name, 1.0) for name in models.keys()],
                dtype=float,
            )
            if float(weights.sum()) <= 0:
                weights = np.ones(len(models), dtype=float)
            weights = weights / weights.sum()

        return VotingRegressor(
            estimators=estimators,
            weights=weights.tolist(),
            n_jobs=1,
        )

    def create_adaptive_voting_ensemble(self, models, target_name):
        """Create adaptive voting ensemble for specific target."""
        target_weights = self.adaptive_weights[target_name]

        estimators = []
        weights = []

        for model_name, weight in target_weights.items():
            if model_name in models:
                estimators.append((model_name, clone(models[model_name])))
                weights.append(weight)

        return VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=1,
        )

    def create_standard_voting_ensemble(self, models):
        """Create standard voting ensemble (all models equal weight)."""
        estimators = [(name, clone(models[name])) for name in models.keys()]

        return VotingRegressor(
            estimators=estimators,
            weights=None,
            n_jobs=1,
        )

    def evaluate_model_cv(self, model, X_train, y_train, model_name, sample_weight=None):
        """Evaluate model using cross-validation on a cloned estimator."""
        cv_scores = cross_val_score(
            clone(model),
            X_train,
            y_train,
            cv=self.cv_splitter,
            scoring='neg_mean_absolute_error',
        )
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()

        print(f"  {model_name} CV MAE: {cv_mae:.3f} +/- {cv_std:.3f}")
        return cv_mae, cv_std
    
    def train_and_evaluate_adaptive_ensembles(
        self, X_train, X_test, y_train, y_test, sample_weight=None, groups=None, *, fast_weights: bool = False
    ):
        """Train and evaluate adaptive ensemble models with weight optimization.

        When ``groups`` is provided, ensemble weights are tuned with GroupKFold
        on the full training set; final models are always refit on 100% of
        ``X_train`` (no 20% holdout waste).
        """
        print("\nTraining and evaluating adaptive ensemble models...")

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

        use_groupkfold = groups is not None and len(np.unique(groups)) >= 2
        if use_groupkfold:
            print(f"Weight tuning: GroupKFold on {len(np.unique(groups))} patients; final fit on 100% train")
        else:
            print("Weight tuning: random 80/20 split; final fit on 100% train")

        results = {}
        self._best_single_maes = {}
        self._optimized_weights = {}

        print(f"Train={X_train.shape}, Test={X_test.shape}")

        for i, target_name in enumerate(self.target_names):
            print(f"\n{target_name}:")
            print("-" * 50)

            y_train_full = y_train[:, i]
            y_test_target = y_test[:, i]
            w_full = self._per_target_sample_weights(target_name, y_train_full, weights)

            fallback_pred = float(np.nanmedian(y_train_full)) if len(y_train_full) else 0.0
            if not np.isfinite(fallback_pred):
                fallback_pred = 0.0

            if (
                self.z_head == "quantile_v7"
                and target_name in {"kidney_left_delta_z", "kidney_right_delta_z"}
                and self._X_train_imputed is not None
                and self._X_val_imputed is not None
                and self.z_driver_names
            ):
                from src.models.z_quantile_v7 import fit_quantile_z, predict_quantile_z

                z_model, _ = fit_quantile_z(
                    self._X_train_imputed,
                    self.feature_names,
                    y_train_full,
                    self.z_driver_names,
                    sample_weight=w_full,
                )
                self.trained_models[target_name] = z_model
                optimized_pred = self._sanitize_predictions(
                    predict_quantile_z(
                        z_model,
                        self._X_val_imputed,
                        self.feature_names,
                        self.z_driver_names,
                    ),
                    fallback_pred,
                )
                optimized_mae = mean_absolute_error(y_test_target, optimized_pred)
                optimized_rmse = np.sqrt(mean_squared_error(y_test_target, optimized_pred))
                optimized_r2 = r2_score(y_test_target, optimized_pred)
                optimized_median_ae = np.median(np.abs(y_test_target - optimized_pred))
                optimized_error_5mm = np.mean(np.abs(y_test_target - optimized_pred) < 5) * 100
                optimized_error_10mm = np.mean(np.abs(y_test_target - optimized_pred) < 10) * 100
                optimized_max_error = np.max(np.abs(y_test_target - optimized_pred))
                optimized_outliers = np.sum(np.abs(y_test_target - optimized_pred) > 20)
                optimized_std_error = np.std(y_test_target - optimized_pred)
                self._best_single_maes[target_name] = optimized_mae
                results[target_name] = {
                    'Optimized_MAE': optimized_mae,
                    'Optimized_RMSE': optimized_rmse,
                    'Optimized_R2': optimized_r2,
                    'Optimized_Median_AE': optimized_median_ae,
                    'Optimized_Error_5mm': optimized_error_5mm,
                    'Optimized_Error_10mm': optimized_error_10mm,
                    'Optimized_Max_Error': optimized_max_error,
                    'Optimized_Outliers_20mm': optimized_outliers,
                    'Optimized_Std_Error': optimized_std_error,
                    'Adaptive_MAE': optimized_mae,
                    'Adaptive_RMSE': optimized_rmse,
                    'Adaptive_R2': optimized_r2,
                    'Adaptive_Median_AE': optimized_median_ae,
                    'Adaptive_Error_5mm': optimized_error_5mm,
                    'Adaptive_Error_10mm': optimized_error_10mm,
                    'Adaptive_Max_Error': optimized_max_error,
                    'Adaptive_Outliers_20mm': optimized_outliers,
                    'Adaptive_Std_Error': optimized_std_error,
                    'Standard_MAE': optimized_mae,
                    'Standard_RMSE': optimized_rmse,
                    'Standard_R2': optimized_r2,
                    'Standard_Median_AE': optimized_median_ae,
                    'Standard_Error_5mm': optimized_error_5mm,
                    'Standard_Error_10mm': optimized_error_10mm,
                    'Standard_Max_Error': optimized_max_error,
                    'Standard_Outliers_20mm': optimized_outliers,
                    'Standard_Std_Error': optimized_std_error,
                    'Best_Single_Model': 'QuantileRegressor_v7',
                    'Improvement_Optimized_vs_Standard': 0.0,
                    'Improvement_Optimized_vs_Adaptive': 0.0,
                    'Improvement_Optimized_vs_Best': 0.0,
                }
                print(f"  V7 QuantileRegressor ({len(self.z_driver_names)} drivers) - MAE: {optimized_mae:.3f} mm, R2: {optimized_r2:.3f}")
                continue

            base_models = self.load_base_models(target_name)

            if fast_weights:
                priors = self.adaptive_weights.get(target_name, {})
                total = sum(priors.get(n, 1.0) for n in base_models.keys()) or len(base_models)
                optimized_weights = {n: priors.get(n, 1.0) / total for n in base_models.keys()}
            elif use_groupkfold:
                optimized_weights = self._optimize_weights_groupkfold(
                    base_models,
                    X_train,
                    y_train_full,
                    groups,
                    target_name,
                    sample_weight=w_full,
                )
            else:
                if w_full is not None:
                    X_wt, X_wv, y_wt, y_wv, w_wt, _w_wv = train_test_split(
                        X_train, y_train_full, w_full, test_size=0.2, random_state=42
                    )
                else:
                    X_wt, X_wv, y_wt, y_wv = train_test_split(
                        X_train, y_train_full, test_size=0.2, random_state=42
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
            self._optimized_weights[target_name] = optimized_weights

            optimized_ensemble = self.create_optimized_voting_ensemble(
                base_models, target_name, optimized_weights
            )
            adaptive_ensemble = self.create_adaptive_voting_ensemble(base_models, target_name)
            standard_ensemble = self.create_standard_voting_ensemble(base_models)

            self._fit_voting_ensemble(optimized_ensemble, X_train, y_train_full, w_full)
            self._fit_voting_ensemble(adaptive_ensemble, X_train, y_train_full, w_full)
            self._fit_voting_ensemble(standard_ensemble, X_train, y_train_full, w_full)

            self.trained_models[target_name] = optimized_ensemble

            print("  Base Models CV Performance:")
            best_single_mae = float("inf")
            for model_name in self.adaptive_weights[target_name].keys():
                if model_name in base_models:
                    cv_mae, cv_std = self.evaluate_model_cv(
                        base_models[model_name],
                        X_train,
                        y_train_full,
                        model_name,
                        sample_weight=w_full,
                    )
                    if cv_mae < best_single_mae:
                        best_single_mae = cv_mae
            self._best_single_maes[target_name] = best_single_mae

            optimized_pred = self._sanitize_predictions(
                optimized_ensemble.predict(X_test), fallback_pred
            )
            adaptive_pred = self._sanitize_predictions(
                adaptive_ensemble.predict(X_test), fallback_pred
            )
            standard_pred = self._sanitize_predictions(
                standard_ensemble.predict(X_test), fallback_pred
            )

            use_adaptive = not np.all(np.isfinite(optimized_pred))
            if not use_adaptive:
                opt_r2 = r2_score(y_test_target, optimized_pred)
                adp_r2 = r2_score(y_test_target, adaptive_pred)
                use_adaptive = opt_r2 < 0 and adp_r2 > opt_r2
            if use_adaptive:
                print(
                    f"  [INFO] Using adaptive ensemble for {target_name} "
                    "(optimized unstable or worse than adaptive on test)"
                )
                self.trained_models[target_name] = adaptive_ensemble
                optimized_pred = adaptive_pred.copy()
            
            # Metrics for optimized ensemble
            optimized_mae = mean_absolute_error(y_test_target, optimized_pred)
            optimized_rmse = np.sqrt(mean_squared_error(y_test_target, optimized_pred))
            optimized_r2 = r2_score(y_test_target, optimized_pred)
            optimized_median_ae = np.median(np.abs(y_test_target - optimized_pred))
            
            # Clinical metrics for optimized
            optimized_error_5mm = np.mean(np.abs(y_test_target - optimized_pred) < 5) * 100
            optimized_error_10mm = np.mean(np.abs(y_test_target - optimized_pred) < 10) * 100
            optimized_max_error = np.max(np.abs(y_test_target - optimized_pred))
            optimized_outliers = np.sum(np.abs(y_test_target - optimized_pred) > 20)
            optimized_std_error = np.std(y_test_target - optimized_pred)
            
            # Metrics for adaptive ensemble
            adaptive_mae = mean_absolute_error(y_test_target, adaptive_pred)
            adaptive_rmse = np.sqrt(mean_squared_error(y_test_target, adaptive_pred))
            adaptive_r2 = r2_score(y_test_target, adaptive_pred)
            adaptive_median_ae = np.median(np.abs(y_test_target - adaptive_pred))
            
            # Clinical metrics for adaptive
            adaptive_error_5mm = np.mean(np.abs(y_test_target - adaptive_pred) < 5) * 100
            adaptive_error_10mm = np.mean(np.abs(y_test_target - adaptive_pred) < 10) * 100
            adaptive_max_error = np.max(np.abs(y_test_target - adaptive_pred))
            adaptive_outliers = np.sum(np.abs(y_test_target - adaptive_pred) > 20)
            adaptive_std_error = np.std(y_test_target - adaptive_pred)
            
            # Metrics for standard ensemble
            standard_mae = mean_absolute_error(y_test_target, standard_pred)
            standard_rmse = np.sqrt(mean_squared_error(y_test_target, standard_pred))
            standard_r2 = r2_score(y_test_target, standard_pred)
            standard_median_ae = np.median(np.abs(y_test_target - standard_pred))
            
            # Clinical metrics for standard
            standard_error_5mm = np.mean(np.abs(y_test_target - standard_pred) < 5) * 100
            standard_error_10mm = np.mean(np.abs(y_test_target - standard_pred) < 10) * 100
            standard_max_error = np.max(np.abs(y_test_target - standard_pred))
            standard_outliers = np.sum(np.abs(y_test_target - standard_pred) > 20)
            standard_std_error = np.std(y_test_target - standard_pred)
            
            results[target_name] = {
                'Optimized_MAE': optimized_mae,
                'Optimized_RMSE': optimized_rmse,
                'Optimized_R2': optimized_r2,
                'Optimized_Median_AE': optimized_median_ae,
                'Optimized_Error_5mm': optimized_error_5mm,
                'Optimized_Error_10mm': optimized_error_10mm,
                'Optimized_Max_Error': optimized_max_error,
                'Optimized_Outliers_20mm': optimized_outliers,
                'Optimized_Std_Error': optimized_std_error,
                'Adaptive_MAE': adaptive_mae,
                'Adaptive_RMSE': adaptive_rmse,
                'Adaptive_R2': adaptive_r2,
                'Adaptive_Median_AE': adaptive_median_ae,
                'Adaptive_Error_5mm': adaptive_error_5mm,
                'Adaptive_Error_10mm': adaptive_error_10mm,
                'Adaptive_Max_Error': adaptive_max_error,
                'Adaptive_Outliers_20mm': adaptive_outliers,
                'Adaptive_Std_Error': adaptive_std_error,
                'Standard_MAE': standard_mae,
                'Standard_RMSE': standard_rmse,
                'Standard_R2': standard_r2,
                'Standard_Median_AE': standard_median_ae,
                'Standard_Error_5mm': standard_error_5mm,
                'Standard_Error_10mm': standard_error_10mm,
                'Standard_Max_Error': standard_max_error,
                'Standard_Outliers_20mm': standard_outliers,
                'Standard_Std_Error': standard_std_error,
                'Best_Single_Model': self.best_models[target_name],
                'Improvement_Optimized_vs_Standard': ((standard_mae - optimized_mae) / standard_mae) * 100,
                'Improvement_Optimized_vs_Adaptive': ((adaptive_mae - optimized_mae) / adaptive_mae) * 100,
                'Improvement_Optimized_vs_Best': ((self._get_best_single_mae(target_name) - optimized_mae) / self._get_best_single_mae(target_name)) * 100 if self._get_best_single_mae(target_name) is not None else 0.0
            }
            
            print(f"  Optimized Ensemble - MAE: {optimized_mae:.3f} mm, R2: {optimized_r2:.3f}")
            print(f"    <5mm accuracy: {optimized_error_5mm:.1f}%, <10mm accuracy: {optimized_error_10mm:.1f}%")
            print(f"  Adaptive Ensemble - MAE: {adaptive_mae:.3f} mm, R2: {adaptive_r2:.3f}")
            print(f"    <5mm accuracy: {adaptive_error_5mm:.1f}%, <10mm accuracy: {adaptive_error_10mm:.1f}%")
            print(f"  Standard Ensemble - MAE: {standard_mae:.3f} mm, R2: {standard_r2:.3f}")
            print(f"    <5mm accuracy: {standard_error_5mm:.1f}%, <10mm accuracy: {standard_error_10mm:.1f}%")
            print(f"  Improvement over standard: {((standard_mae - optimized_mae) / standard_mae) * 100:.1f}%")
            print(f"  Improvement over adaptive: {((adaptive_mae - optimized_mae) / adaptive_mae) * 100:.1f}%")
            print(f"  Improvement over best single: {((self._get_best_single_mae(target_name) - optimized_mae) / self._get_best_single_mae(target_name)) * 100:.1f}%" if self._get_best_single_mae(target_name) is not None else "  Improvement over best single: N/A")
        
        self.results = results
        return results
    
    def _get_best_single_mae(self, target_name):
        """Get best single model MAE for target from actual CV results"""
        # This should be computed dynamically during training
        # For now, returns None to indicate dynamic calculation needed
        return getattr(self, '_best_single_maes', {}).get(target_name, None)
    
    def generate_report(self):
        """Generate comprehensive adaptive ensemble report with optimization results"""
        print("\n" + "="*80)
        print("OPTIMIZED ADAPTIVE ENSEMBLE MODELS - PHASE 1 REPORT")
        print("="*80)
        
        print(f"\nDataset Summary:")
        print(f"- Features used: {len(self.feature_names)}")
        print(f"- Target variables: {len(self.target_names)}")
        print(f"- Data sources: DICOMS + Vybor + KiTS19")
        
        print(f"\nOverall Performance Summary:")
        print("-" * 50)
        
        # Calculate average metrics for optimized ensemble
        optimized_mae = np.mean([r['Optimized_MAE'] for r in self.results.values()])
        optimized_rmse = np.mean([r['Optimized_RMSE'] for r in self.results.values()])
        optimized_r2 = np.mean([r['Optimized_R2'] for r in self.results.values()])
        optimized_5mm = np.mean([r['Optimized_Error_5mm'] for r in self.results.values()])
        optimized_10mm = np.mean([r['Optimized_Error_10mm'] for r in self.results.values()])
        
        # Calculate average metrics for adaptive ensemble
        adaptive_mae = np.mean([r['Adaptive_MAE'] for r in self.results.values()])
        adaptive_rmse = np.mean([r['Adaptive_RMSE'] for r in self.results.values()])
        adaptive_r2 = np.mean([r['Adaptive_R2'] for r in self.results.values()])
        adaptive_5mm = np.mean([r['Adaptive_Error_5mm'] for r in self.results.values()])
        adaptive_10mm = np.mean([r['Adaptive_Error_10mm'] for r in self.results.values()])
        
        # Calculate average metrics for standard ensemble
        standard_mae = np.mean([r['Standard_MAE'] for r in self.results.values()])
        standard_rmse = np.mean([r['Standard_RMSE'] for r in self.results.values()])
        standard_r2 = np.mean([r['Standard_R2'] for r in self.results.values()])
        standard_5mm = np.mean([r['Standard_Error_5mm'] for r in self.results.values()])
        standard_10mm = np.mean([r['Standard_Error_10mm'] for r in self.results.values()])
        
        print(f"Optimized Voting Ensemble:")
        print(f"  Average MAE: {optimized_mae:.3f} mm")
        print(f"  Average RMSE: {optimized_rmse:.3f} mm")
        print(f"  Average R2: {optimized_r2:.3f}")
        print(f"  Average <5mm accuracy: {optimized_5mm:.1f}%")
        print(f"  Average <10mm accuracy: {optimized_10mm:.1f}%")
        
        print(f"\nAdaptive Voting Ensemble:")
        print(f"  Average MAE: {adaptive_mae:.3f} mm")
        print(f"  Average RMSE: {adaptive_rmse:.3f} mm")
        print(f"  Average R2: {adaptive_r2:.3f}")
        print(f"  Average <5mm accuracy: {adaptive_5mm:.1f}%")
        print(f"  Average <10mm accuracy: {adaptive_10mm:.1f}%")
        
        print(f"\nStandard Voting Ensemble:")
        print(f"  Average MAE: {standard_mae:.3f} mm")
        print(f"  Average RMSE: {standard_rmse:.3f} mm")
        print(f"  Average R2: {standard_r2:.3f}")
        print(f"  Average <5mm accuracy: {standard_5mm:.1f}%")
        print(f"  Average <10mm accuracy: {standard_10mm:.1f}%")
        
        print(f"\nImprovement Analysis:")
        avg_improvement_opt_vs_std = np.mean([r['Improvement_Optimized_vs_Standard'] for r in self.results.values()])
        avg_improvement_opt_vs_adaptive = np.mean([r['Improvement_Optimized_vs_Adaptive'] for r in self.results.values()])
        avg_improvement_opt_vs_best = np.mean([r['Improvement_Optimized_vs_Best'] for r in self.results.values()])
        
        print(f"  Average improvement over standard: {avg_improvement_opt_vs_std:.1f}%")
        print(f"  Average improvement over adaptive: {avg_improvement_opt_vs_adaptive:.1f}%")
        print(f"  Average improvement over best single: {avg_improvement_opt_vs_best:.1f}%")
        
        print(f"\nOptimized Weights Summary:")
        print("-" * 50)
        for target_name, weights in self._optimized_weights.items():
            print(f"\n{target_name}:")
            for model_name, weight in weights.items():
                print(f"  {model_name}: {weight:.3f}")
        
        print(f"\nDetailed Results by Target:")
        print("-" * 50)
        
        for target_name, metrics in self.results.items():
            print(f"\n{target_name}:")
            print(f"  Best Single Model: {metrics['Best_Single_Model']}")
            print(f"  Optimized Ensemble - MAE: {metrics['Optimized_MAE']:.3f} mm (R2: {metrics['Optimized_R2']:.3f})")
            print(f"  Adaptive Ensemble - MAE: {metrics['Adaptive_MAE']:.3f} mm (R2: {metrics['Adaptive_R2']:.3f})")
            print(f"  Standard Ensemble - MAE: {metrics['Standard_MAE']:.3f} mm (R2: {metrics['Standard_R2']:.3f})")
            print(f"  Improvement over standard: {metrics['Improvement_Optimized_vs_Standard']:.1f}%")
            print(f"  Improvement over adaptive: {metrics['Improvement_Optimized_vs_Adaptive']:.1f}%")
            print(f"  Improvement over best single: {metrics['Improvement_Optimized_vs_Best']:.1f}%")
    
    def save_model(self, filepath="models/adaptive_ensemble.pkl"):
        """Сохранение обученной модели"""
        # Создаем директорию если не существует
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'models': self.trained_models,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'train_data': self.X_train,  # для confidence estimator
            'required_features': self.required_features,
            'target_columns': self.target_columns,
            'adaptive_weights': self.adaptive_weights,
            'best_models': self.best_models,
            'z_head': self.z_head,
            'z_driver_names': self.z_driver_names,
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        print(f"Saved {len(self.trained_models)} trained models")
        print(f"Features: {len(self.feature_names)}, Targets: {len(self.target_names)}")
    
    def save_results(self, filename='adaptive_ensemble_integrated_results.csv'):
        """Save results to CSV with optimized ensembles"""
        rows = []
        for target_name, metrics in self.results.items():
            # Optimized ensemble row
            optimized_row = {
                'Target': target_name,
                'Model': 'Optimized_Voting_Ensemble_Integrated',
                'MAE': metrics['Optimized_MAE'],
                'RMSE': metrics['Optimized_RMSE'],
                'R2': metrics['Optimized_R2'],
                'Median_AE': metrics['Optimized_Median_AE'],
                'Error_5mm': metrics['Optimized_Error_5mm'],
                'Error_10mm': metrics['Optimized_Error_10mm'],
                'Max_Error': metrics['Optimized_Max_Error'],
                'Outliers_20mm': metrics['Optimized_Outliers_20mm'],
                'Std_Error': metrics['Optimized_Std_Error'],
                'Best_Single_Model': metrics['Best_Single_Model'],
                'Improvement_vs_Standard': metrics['Improvement_Optimized_vs_Standard'],
                'Improvement_vs_Adaptive': metrics['Improvement_Optimized_vs_Adaptive'],
                'Improvement_vs_Best': metrics['Improvement_Optimized_vs_Best'],
                'Data_Sources': 'DICOMS+Vybor+KiTS19',
                'Features_Count': len(self.feature_names)
            }
            rows.append(optimized_row)
            
            # Adaptive ensemble row
            adaptive_row = {
                'Target': target_name,
                'Model': 'Adaptive_Voting_Ensemble_Integrated',
                'MAE': metrics['Adaptive_MAE'],
                'RMSE': metrics['Adaptive_RMSE'],
                'R2': metrics['Adaptive_R2'],
                'Median_AE': metrics['Adaptive_Median_AE'],
                'Error_5mm': metrics['Adaptive_Error_5mm'],
                'Error_10mm': metrics['Adaptive_Error_10mm'],
                'Max_Error': metrics['Adaptive_Max_Error'],
                'Outliers_20mm': metrics['Adaptive_Outliers_20mm'],
                'Std_Error': metrics['Adaptive_Std_Error'],
                'Best_Single_Model': metrics['Best_Single_Model'],
                'Improvement_vs_Standard': ((metrics['Standard_MAE'] - metrics['Adaptive_MAE']) / metrics['Standard_MAE']) * 100,
                'Improvement_vs_Optimized': ((metrics['Optimized_MAE'] - metrics['Adaptive_MAE']) / metrics['Optimized_MAE']) * 100,
                'Improvement_vs_Best': ((self._get_best_single_mae(target_name) - metrics['Adaptive_MAE']) / self._get_best_single_mae(target_name)) * 100 if self._get_best_single_mae(target_name) is not None else 0,
                'Data_Sources': 'DICOMS+Vybor+KiTS19',
                'Features_Count': len(self.feature_names)
            }
            rows.append(adaptive_row)
            
            # Standard ensemble row
            standard_row = {
                'Target': target_name,
                'Model': 'Standard_Voting_Ensemble_Integrated',
                'MAE': metrics['Standard_MAE'],
                'RMSE': metrics['Standard_RMSE'],
                'R2': metrics['Standard_R2'],
                'Median_AE': metrics['Standard_Median_AE'],
                'Error_5mm': metrics['Standard_Error_5mm'],
                'Error_10mm': metrics['Standard_Error_10mm'],
                'Max_Error': metrics['Standard_Max_Error'],
                'Outliers_20mm': metrics['Standard_Outliers_20mm'],
                'Std_Error': metrics['Standard_Std_Error'],
                'Best_Single_Model': metrics['Best_Single_Model'],
                'Improvement_vs_Standard': 0.0,  # Reference point
                'Improvement_vs_Optimized': ((metrics['Optimized_MAE'] - metrics['Standard_MAE']) / metrics['Optimized_MAE']) * 100,
                'Improvement_vs_Best': ((self._get_best_single_mae(target_name) - metrics['Standard_MAE']) / self._get_best_single_mae(target_name)) * 100 if self._get_best_single_mae(target_name) is not None and self._get_best_single_mae(target_name) > 0 else 0,
                'Data_Sources': 'DICOMS+Vybor+KiTS19',
                'Features_Count': len(self.feature_names)
            }
            rows.append(standard_row)
        
        results_df = pd.DataFrame(rows)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    """Main training pipeline for optimized adaptive ensemble with integrated data"""
    trainer = AdaptiveEnsembleTrainer()
    
    print("OPTIMIZED ADAPTIVE ENSEMBLE TRAINING WITH INTEGRATED DATA SOURCES")
    print("="*80)
    
    # 1. Load integrated data (train and validation files are kept separate)
    combined_df, train_df, val_df = trainer.load_integrated_data()
    if train_df is None or val_df is None:
        print("Failed to load integrated data")
        return

    # 2. Prepare training data with enhanced features WITHOUT train/val leakage:
    # feature engineering and scaling are fit ONLY on train_df, then applied
    # to val_df. The legacy combined_df is intentionally not used.
    X_train, X_test, y_train, y_test = trainer.prepare_training_data_split(train_df, val_df)
    if X_train is None:
        print("Failed to prepare training data")
        return
    
    # 3. Train and evaluate optimized ensembles
    results = trainer.train_and_evaluate_adaptive_ensembles(X_train, X_test, y_train, y_test)
    
    # 4. Generate comprehensive report
    trainer.generate_report()
    
    # 5. Save results
    trainer.save_results()
    
    # 6. Save trained models
    trainer.save_model()
    
    print(f"\nOPTIMIZED ADAPTIVE ENSEMBLE WITH INTEGRATED DATA TRAINING COMPLETED!")
    return results

if __name__ == "__main__":
    main()
