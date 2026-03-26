#!/usr/bin/env python3
"""
Adaptive Ensemble Models for Kidney Displacement Prediction - INTEGRATED VERSION
Phase 1: Adaptive Weights Voting Ensemble with ALL Data Sources
Integrates: DICOMS + Vybor + KiTS19
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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
warnings.filterwarnings('ignore')

class AdaptiveEnsembleTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        self.target_names = []
        self.results = {}
        self.trained_models = {}  # Для хранения обученных моделей
        self.X_train = None  # Для confidence estimator
        self.cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)  # Воспроизводимый CV
        
        # Расширенный список признаков с учетом всех источников + критически важные
        # Адаптировано под реальные данные в train.csv
        self.required_features = [
            # Базовые геометрические признаки (доступные в данных)
            'kidney_left_center_x_rel', 'kidney_left_center_y_rel', 'kidney_left_center_z_rel',
            'kidney_right_center_x_rel', 'kidney_right_center_y_rel', 'kidney_right_center_z_rel',
            
            # Размеры почек
            'kidney_left_length_mm', 'kidney_left_volume_cm3',
            'kidney_right_length_mm', 'kidney_right_volume_cm3',
            
            # Геометрия тела
            'body_width_mm', 'body_depth_mm', 'body_area_mm2',
            
            # Относительные расстояния
            'kidney_left_to_spine_distance', 'kidney_right_to_spine_distance',
            'kidney_left_to_body_center_distance', 'kidney_right_to_body_center_distance',
            
            # Центры (только для расчета относительных признаков)
            'spine_center_x', 'spine_center_y', 'spine_center_z',
            'body_com_x', 'body_com_y', 'body_com_z',
        ]
        
        # Инженерные признаки (будут добавлены в prepare_training_data)
        self.engineered_features = [
            'body_ratio',  # body_width / body_depth
            'kidney_distance_lr',  # расстояние между почками
            'kidney_left_volume_norm',  # kidney_left_volume / body_width
            'kidney_right_volume_norm',  # kidney_right_volume / body_width
            'kidney_left_length_norm',  # kidney_left_length / body_width
            'kidney_right_length_norm',  # kidney_right_length / body_width
            'volume_asymmetry',  # left_volume - right_volume
            'length_asymmetry',  # left_length - right_length
            'spine_distance_asymmetry',  # left_to_spine - right_to_spine
            'body_center_asymmetry',  # left_to_body - right_to_body
            'kidney_left_to_spine_ratio',  # left_to_spine / body_width
            'kidney_right_to_spine_ratio',  # right_to_spine / body_width
            'patient_position_encoded',  # будет создан из данных
        ]
        
        # Дополнительные cross-features для улучшения производительности
        self.cross_features = [
            'body_volume_estimated',  # body_width * body_depth * средняя высота почки
            'kidney_left_density_ratio',  # volume / length
            'kidney_right_density_ratio',  # volume / length
            'spine_to_body_ratio_x',  # spine_center_x / body_width
            'spine_to_body_ratio_y',  # spine_center_y / body_depth
            'body_com_to_spine_distance',  # расстояние между центром масс и позвоночником
            'kidney_left_spine_interaction',  # left_to_spine * left_volume
            'kidney_right_spine_interaction',  # right_to_spine * right_volume
            'body_size_index',  # sqrt(width^2 + depth^2)
            'kidney_position_index_left',  # sqrt(x_rel^2 + y_rel^2 + z_rel^2)
            'kidney_position_index_right',  # sqrt(x_rel^2 + y_rel^2 + z_rel^2)
            'volume_to_area_ratio_left',  # left_volume / body_area
            'volume_to_area_ratio_right',  # right_volume / body_area
            'relative_volume_sum',  # (left_volume + right_volume) / body_width
            'kidney_separation_angle',  # угол между почками относительно позвоночника
        ]

        self.target_columns = [
            'kidney_left_delta_x', 'kidney_left_delta_y', 'kidney_left_delta_z',
            'kidney_right_delta_x', 'kidney_right_delta_y', 'kidney_right_delta_z'
        ]
        
        # Best models per target based on comparison results
        self.best_models = {
            'kidney_left_delta_x': 'RandomForest',
            'kidney_left_delta_y': 'RandomForest', 
            'kidney_left_delta_z': 'Lasso',
            'kidney_right_delta_x': 'Ridge',
            'kidney_right_delta_y': 'RandomForest',
            'kidney_right_delta_z': 'GradientBoosting'
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
                'RandomForest': 1.5,  # Best single model but weighted lower to prevent overfitting
                'Lasso': 2.0,  # Increased weight for diversity
                'Ridge': 1.2,
                'GradientBoosting': 0.8
            },
            'kidney_left_delta_z': {
                'RandomForest': 1.2,  # Lower weight due to ensemble degradation
                'Lasso': 2.5,  # Best single model with highest weight
                'Ridge': 1.0,
                'GradientBoosting': 0.8
            },
            'kidney_right_delta_x': {
                'RandomForest': 1.0,  # Lower weight due to ensemble degradation
                'Lasso': 0.8,
                'Ridge': 2.5,  # Best single model with highest weight
                'GradientBoosting': 0.6
            },
            'kidney_right_delta_y': {
                'RandomForest': 1.5,  # Best single model but weighted lower to prevent overfitting
                'Lasso': 1.2,
                'Ridge': 1.0,
                'GradientBoosting': 0.8
            },
            'kidney_right_delta_z': {
                'RandomForest': 1.2,  # Lower weight due to ensemble degradation
                'Lasso': 1.0,
                'Ridge': 0.8,
                'GradientBoosting': 2.0  # Best single model with highest weight
            }
        }
    
    def load_integrated_data(self):
        """Load integrated data from all sources"""
        print("Loading integrated datasets from all sources...")
        
        # Используем наши интегрированные данные
        try:
            train_df = pd.read_csv('data/processed/train.csv')
            val_df = pd.read_csv('data/processed/validation.csv')
            
            print(f"Integrated Train dataset: {len(train_df)} cases")
            print(f"Integrated Validation dataset: {len(val_df)} cases")
            
            # Объединяем для обучения
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            print(f"Combined dataset: {len(combined_df)} cases")
            
            return combined_df, train_df, val_df
            
        except FileNotFoundError:
            print("Integrated data files not found. Please run data integration first.")
            print("Run: python src/models/data_integration_fix.py")
            return None, None, None
    
        
    def prepare_training_data(self, df):
        """Prepare features and targets for training with engineered features"""
        target_cols = [col for col in self.target_columns if col in df.columns]
        
        # Проверяем наличие базовых признаков
        missing_base_features = [col for col in self.required_features if col not in df.columns]
        if missing_base_features:
            print(f"❌ CRITICAL: Missing base features: {missing_base_features}")
            print("This will cause silent feature dropping and poor performance!")
            # Не продолжаем с отсутствующими критическими признаками
            available_base_features = [col for col in self.required_features if col in df.columns]
            if len(available_base_features) < len(self.required_features) * 0.8:
                print("Too many missing features. Cannot proceed.")
                return None, None, None, None
        
        if len(target_cols) == 0:
            print("No target variables found in dataset")
            return None, None, None, None
        
        # Создаем инженерные признаки
        df_enhanced = self._create_engineered_features(df.copy())
        
        # Создаем cross-features
        df_enhanced = self._create_cross_features(df_enhanced)
        
        # Объединяем базовые, инженерные и cross-features
        base_feature_cols = [col for col in self.required_features if col in df_enhanced.columns]
        engineered_feature_cols = [col for col in self.engineered_features if col in df_enhanced.columns]
        cross_feature_cols = [col for col in self.cross_features if col in df_enhanced.columns]
        
        all_feature_cols = base_feature_cols + engineered_feature_cols + cross_feature_cols
        
        print(f"✅ Base features: {len(base_feature_cols)}")
        print(f"✅ Engineered features: {len(engineered_feature_cols)}")
        print(f"✅ Cross features: {len(cross_feature_cols)}")
        print(f"✅ Total features: {len(all_feature_cols)}")
        
        # Разделяем признаки и цели
        X = df_enhanced[all_feature_cols].values
        y = df_enhanced[target_cols].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = all_feature_cols
        self.target_names = target_cols
        self.X_train = X_train_scaled  # Сохраняем для confidence estimator
        
        print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _create_engineered_features(self, df):
        """Create engineered features from base features"""
        print("\n🔧 Creating engineered features...")
        
        # 1. Body ratio
        if 'body_width_mm' in df.columns and 'body_depth_mm' in df.columns:
            df['body_ratio'] = df['body_width_mm'] / df['body_depth_mm']
            print("  ✅ body_ratio created")
        
        # 2. Расстояние между почками (используем center_x_rel)
        if 'kidney_left_center_x_rel' in df.columns and 'kidney_right_center_x_rel' in df.columns:
            df['kidney_distance_lr'] = np.abs(df['kidney_left_center_x_rel'] - df['kidney_right_center_x_rel'])
            print("  ✅ kidney_distance_lr created")
        
        # 3. Нормализованные размеры почек
        if 'kidney_left_volume_cm3' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_left_volume_norm'] = df['kidney_left_volume_cm3'] / df['body_width_mm']
            print("  ✅ kidney_left_volume_norm created")
            
        if 'kidney_right_volume_cm3' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_right_volume_norm'] = df['kidney_right_volume_cm3'] / df['body_width_mm']
            print("  ✅ kidney_right_volume_norm created")
        
        if 'kidney_left_length_mm' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_left_length_norm'] = df['kidney_left_length_mm'] / df['body_width_mm']
            print("  ✅ kidney_left_length_norm created")
            
        if 'kidney_right_length_mm' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_right_length_norm'] = df['kidney_right_length_mm'] / df['body_width_mm']
            print("  ✅ kidney_right_length_norm created")
        
        # 4. Признаки асимметрии
        if 'kidney_left_volume_cm3' in df.columns and 'kidney_right_volume_cm3' in df.columns:
            df['volume_asymmetry'] = df['kidney_left_volume_cm3'] - df['kidney_right_volume_cm3']
            print("  ✅ volume_asymmetry created")
        
        if 'kidney_left_length_mm' in df.columns and 'kidney_right_length_mm' in df.columns:
            df['length_asymmetry'] = df['kidney_left_length_mm'] - df['kidney_right_length_mm']
            print("  ✅ length_asymmetry created")
        
        if 'kidney_left_to_spine_distance' in df.columns and 'kidney_right_to_spine_distance' in df.columns:
            df['spine_distance_asymmetry'] = df['kidney_left_to_spine_distance'] - df['kidney_right_to_spine_distance']
            print("  ✅ spine_distance_asymmetry created")
        
        if 'kidney_left_to_body_center_distance' in df.columns and 'kidney_right_to_body_center_distance' in df.columns:
            df['body_center_asymmetry'] = df['kidney_left_to_body_center_distance'] - df['kidney_right_to_body_center_distance']
            print("  ✅ body_center_asymmetry created")
        
        # 5. Нормализованные расстояния до позвоночника
        if 'kidney_left_to_spine_distance' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_left_to_spine_ratio'] = df['kidney_left_to_spine_distance'] / df['body_width_mm']
            print("  ✅ kidney_left_to_spine_ratio created")
        
        if 'kidney_right_to_spine_distance' in df.columns and 'body_width_mm' in df.columns:
            df['kidney_right_to_spine_ratio'] = df['kidney_right_to_spine_distance'] / df['body_width_mm']
            print("  ✅ kidney_right_to_spine_ratio created")
        
        # 6. Добавляем patient_position_encoded
        if 'patient_position_encoded' not in df.columns:
            # Проверяем есть ли scan_position в исходных данных
            # Если нет, используем значение по умолчанию (supine=1)
            df['patient_position_encoded'] = 1  # Все пациенты в положении supine по умолчанию
            print("  ✅ patient_position_encoded set to default (supine=1)")
        
        print(f"🔧 Engineered features creation completed. New shape: {df.shape}")
        return df
    
    def _create_cross_features(self, df):
        """Create advanced cross-features for better performance"""
        print("\n🔧 Creating cross-features...")
        
        # 1. Body volume estimation
        if 'body_width_mm' in df.columns and 'body_depth_mm' in df.columns and 'kidney_left_length_mm' in df.columns:
            avg_kidney_height = (df['kidney_left_length_mm'] + df.get('kidney_right_length_mm', df['kidney_left_length_mm'])) / 2
            df['body_volume_estimated'] = df['body_width_mm'] * df['body_depth_mm'] * avg_kidney_height / 1000  # в см³
            print("  ✅ body_volume_estimated created")
        
        # 2. Kidney density ratios
        if 'kidney_left_volume_cm3' in df.columns and 'kidney_left_length_mm' in df.columns:
            df['kidney_left_density_ratio'] = df['kidney_left_volume_cm3'] / df['kidney_left_length_mm']
            print("  ✅ kidney_left_density_ratio created")
            
        if 'kidney_right_volume_cm3' in df.columns and 'kidney_right_length_mm' in df.columns:
            df['kidney_right_density_ratio'] = df['kidney_right_volume_cm3'] / df['kidney_right_length_mm']
            print("  ✅ kidney_right_density_ratio created")
        
        # 3. Spine to body ratios
        if 'spine_center_x' in df.columns and 'body_width_mm' in df.columns:
            df['spine_to_body_ratio_x'] = df['spine_center_x'] / df['body_width_mm']
            print("  ✅ spine_to_body_ratio_x created")
            
        if 'spine_center_y' in df.columns and 'body_depth_mm' in df.columns:
            df['spine_to_body_ratio_y'] = df['spine_center_y'] / df['body_depth_mm']
            print("  ✅ spine_to_body_ratio_y created")
        
        # 4. Body COM to spine distance
        if all(col in df.columns for col in ['body_com_x', 'body_com_y', 'spine_center_x', 'spine_center_y']):
            df['body_com_to_spine_distance'] = np.sqrt(
                (df['body_com_x'] - df['spine_center_x'])**2 + 
                (df['body_com_y'] - df['spine_center_y'])**2
            )
            print("  ✅ body_com_to_spine_distance created")
        
        # 5. Kidney-spine interactions
        if 'kidney_left_to_spine_distance' in df.columns and 'kidney_left_volume_cm3' in df.columns:
            df['kidney_left_spine_interaction'] = df['kidney_left_to_spine_distance'] * df['kidney_left_volume_cm3']
            print("  ✅ kidney_left_spine_interaction created")
            
        if 'kidney_right_to_spine_distance' in df.columns and 'kidney_right_volume_cm3' in df.columns:
            df['kidney_right_spine_interaction'] = df['kidney_right_to_spine_distance'] * df['kidney_right_volume_cm3']
            print("  ✅ kidney_right_spine_interaction created")
        
        # 6. Body size index
        if 'body_width_mm' in df.columns and 'body_depth_mm' in df.columns:
            df['body_size_index'] = np.sqrt(df['body_width_mm']**2 + df['body_depth_mm']**2)
            print("  ✅ body_size_index created")
        
        # 7. Kidney position indices
        if all(col in df.columns for col in ['kidney_left_center_x_rel', 'kidney_left_center_y_rel', 'kidney_left_center_z_rel']):
            df['kidney_position_index_left'] = np.sqrt(
                df['kidney_left_center_x_rel']**2 + 
                df['kidney_left_center_y_rel']**2 + 
                df['kidney_left_center_z_rel']**2
            )
            print("  ✅ kidney_position_index_left created")
            
        if all(col in df.columns for col in ['kidney_right_center_x_rel', 'kidney_right_center_y_rel', 'kidney_right_center_z_rel']):
            df['kidney_position_index_right'] = np.sqrt(
                df['kidney_right_center_x_rel']**2 + 
                df['kidney_right_center_y_rel']**2 + 
                df['kidney_right_center_z_rel']**2
            )
            print("  ✅ kidney_position_index_right created")
        
        # 8. Volume to area ratios
        if 'kidney_left_volume_cm3' in df.columns and 'body_area_mm2' in df.columns:
            df['volume_to_area_ratio_left'] = df['kidney_left_volume_cm3'] / (df['body_area_mm2'] / 100)  # переводим в см²
            print("  ✅ volume_to_area_ratio_left created")
            
        if 'kidney_right_volume_cm3' in df.columns and 'body_area_mm2' in df.columns:
            df['volume_to_area_ratio_right'] = df['kidney_right_volume_cm3'] / (df['body_area_mm2'] / 100)
            print("  ✅ volume_to_area_ratio_right created")
        
        # 9. Relative volume sum
        if 'kidney_left_volume_cm3' in df.columns and 'kidney_right_volume_cm3' in df.columns and 'body_width_mm' in df.columns:
            df['relative_volume_sum'] = (df['kidney_left_volume_cm3'] + df['kidney_right_volume_cm3']) / df['body_width_mm']
            print("  ✅ relative_volume_sum created")
        
        # 10. Kidney separation angle (упрощенный)
        if all(col in df.columns for col in ['kidney_left_center_x_rel', 'kidney_right_center_x_rel', 'kidney_left_center_y_rel', 'kidney_right_center_y_rel']):
            # Вычисляем угол между векторами от центра к почкам
            left_vector = np.array([df['kidney_left_center_x_rel'], df['kidney_left_center_y_rel']]).T
            right_vector = np.array([df['kidney_right_center_x_rel'], df['kidney_right_center_y_rel']]).T
            
            # Вычисляем угол между векторами
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
            print("  ✅ kidney_separation_angle created")
        
        print(f"🔧 Cross-features creation completed. New shape: {df.shape}")
        return df
    
    def load_base_models(self):
        """Load base models with optimal parameters"""
        print("\nLoading base models with optimal parameters...")
        
        models = {}
        
        # Use best parameters found during optimization
        model_configs = {
            'RandomForest': {
                'n_estimators': 500,
                'max_depth': 20,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            },
            'Lasso': {
                'alpha': 0.1,
                'max_iter': 5000,
                'random_state': 42
            },
            'Ridge': {
                'alpha': 1.0,
                'solver': 'auto',
                'random_state': 42
            },
            'GradientBoosting': {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 5,
                'subsample': 0.8,
                'random_state': 42
            }
        }
        
        for model_name, config in model_configs.items():
            if model_name == 'RandomForest':
                model = RandomForestRegressor(**config)
                model.random_state = 42  # Явная установка random_state
                models[model_name] = model
            elif model_name == 'Lasso':
                models[model_name] = Lasso(**config)
            elif model_name == 'Ridge':
                models[model_name] = Ridge(**config)
            elif model_name == 'GradientBoosting':
                models[model_name] = GradientBoostingRegressor(**config)
        
        return models
    
    def optimize_ensemble_weights(self, models, X_train, y_train, X_val, y_val, target_name):
        """Оптимизация весов ансамбля с помощью scipy.optimize"""
        print(f"\n🔧 Optimizing ensemble weights for {target_name}...")
        
        # Получаем предсказания каждой модели на валидационном наборе
        model_predictions = {}
        for model_name, model in models.items():
            # Создаем копию модели для оптимизации весов
            model_copy = self._copy_model(model)
            model_copy.fit(X_train, y_train)
            pred = model_copy.predict(X_val)
            model_predictions[model_name] = pred
        
        def objective_function(weights):
            """Целевая функция для минимизации (MAE)"""
            # Нормализуем веса чтобы сумма была 1
            weights = np.abs(weights) / np.sum(np.abs(weights))
            
            # Взвешенное предсказание
            ensemble_pred = np.zeros(len(y_val))
            for i, (model_name, pred) in enumerate(model_predictions.items()):
                ensemble_pred += weights[i] * pred
            
            # Возвращаем MAE
            return mean_absolute_error(y_val, ensemble_pred)
        
        # Начальные веса (равные)
        initial_weights = np.ones(len(models)) / len(models)
        
        # Ограничения: веса должны быть неотрицательными
        bounds = [(0, 1) for _ in range(len(models))]
        
        # Оптимизация
        result = minimize(
            objective_function,
            initial_weights,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        # Нормализуем оптимальные веса
        optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))
        
        # Создаем словарь весов
        optimized_weights = {}
        for i, model_name in enumerate(models.keys()):
            optimized_weights[model_name] = optimal_weights[i]
        
        print(f"  ✅ Optimized weights: {optimized_weights}")
        
        # Сравниваем с равными весами
        equal_weights = {name: 1.0/len(models) for name in models.keys()}
        equal_pred = np.zeros(len(y_val))
        for model_name, pred in model_predictions.items():
            equal_pred += equal_weights[model_name] * pred
        
        optimized_pred = np.zeros(len(y_val))
        for model_name, pred in model_predictions.items():
            optimized_pred += optimized_weights[model_name] * pred
        
        equal_mae = mean_absolute_error(y_val, equal_pred)
        optimized_mae = mean_absolute_error(y_val, optimized_pred)
        
        improvement = ((equal_mae - optimized_mae) / equal_mae) * 100
        print(f"  📈 Improvement: {improvement:.1f}% (MAE: {equal_mae:.3f} → {optimized_mae:.3f})")
        
        return optimized_weights
    
    def _copy_model(self, model):
        """Создает копию модели с теми же параметрами"""
        if hasattr(model, 'get_params'):
            return type(model)(**model.get_params())
        else:
            # Fallback для простых моделей
            return type(model)()
    
    def create_optimized_voting_ensemble(self, models, target_name, optimized_weights):
        """Create voting ensemble with optimized weights"""
        estimators = [(name, models[name]) for name in models.keys()]
        weights = [optimized_weights[name] for name in models.keys()]
        
        return VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=1
        )
    
    def create_adaptive_voting_ensemble(self, models, target_name):
        """Create adaptive voting ensemble for specific target"""
        target_weights = self.adaptive_weights[target_name]
        
        # Create estimators with weights
        estimators = []
        weights = []
        
        for model_name, weight in target_weights.items():
            if model_name in models:
                estimators.append((model_name, models[model_name]))
                weights.append(weight)
        
        return VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=1  # Отключаем параллелизм для детерминированности
        )
    
    def create_standard_voting_ensemble(self, models):
        """Create standard voting ensemble (all models equal weight)"""
        estimators = [(name, models[name]) for name in models.keys()]
        
        return VotingRegressor(
            estimators=estimators,
            weights=None,  # Equal weights
            n_jobs=1  # Отключаем параллелизм для детерминированности
        )
    
    def evaluate_model_cv(self, model, X_train, y_train, model_name):
        """Evaluate model using cross-validation"""
        cv_scores = cross_val_score(model, X_train, y_train, 
                                  cv=self.cv_splitter,
                                  scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"  {model_name} CV MAE: {cv_mae:.3f} ± {cv_std:.3f}")
        return cv_mae, cv_std
    
    def train_and_evaluate_adaptive_ensembles(self, X_train, X_test, y_train, y_test):
        """Train and evaluate adaptive ensemble models with weight optimization"""
        print("\nTraining and evaluating adaptive ensemble models...")
        
        # Load base models
        base_models = self.load_base_models()
        
        results = {}
        self._best_single_maes = {}  # Store actual CV results
        self._optimized_weights = {}  # Store optimized weights
        
        # Дополнительное разделение для валидации при оптимизации весов
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"Data split for optimization: Train={X_train_main.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
        for i, target_name in enumerate(self.target_names):
            print(f"\n{target_name}:")
            print("-" * 50)
            
            y_train_target = y_train_main[:, i]
            y_val_target = y_val[:, i]
            y_test_target = y_test[:, i]
            
            # Оптимизация весов
            optimized_weights = self.optimize_ensemble_weights(
                base_models, X_train_main, y_train_target, X_val, y_val_target, target_name
            )
            self._optimized_weights[target_name] = optimized_weights
            
            # Create ensembles
            optimized_ensemble = self.create_optimized_voting_ensemble(base_models, target_name, optimized_weights)
            adaptive_ensemble = self.create_adaptive_voting_ensemble(base_models, target_name)
            standard_ensemble = self.create_standard_voting_ensemble(base_models)
            
            # Train ensembles on full training data
            optimized_ensemble.fit(X_train_main, y_train_target)
            adaptive_ensemble.fit(X_train_main, y_train_target)
            standard_ensemble.fit(X_train_main, y_train_target)
            
            # Сохраняем обученную оптимизированную модель
            self.trained_models[target_name] = optimized_ensemble
            
            # CV evaluation of base models to get actual best single MAE
            print("  Base Models CV Performance:")
            best_single_mae = float('inf')
            for model_name in self.adaptive_weights[target_name].keys():
                if model_name in base_models:
                    cv_mae, cv_std = self.evaluate_model_cv(
                        base_models[model_name], X_train_main, y_train_target, model_name
                    )
                    # Track best single model MAE
                    if cv_mae < best_single_mae:
                        best_single_mae = cv_mae
            self._best_single_maes[target_name] = best_single_mae
            
            # Predictions
            optimized_pred = optimized_ensemble.predict(X_test)
            adaptive_pred = adaptive_ensemble.predict(X_test)
            standard_pred = standard_ensemble.predict(X_test)
            
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
            
            print(f"  Optimized Ensemble - MAE: {optimized_mae:.3f} mm, R²: {optimized_r2:.3f}")
            print(f"    <5mm accuracy: {optimized_error_5mm:.1f}%, <10mm accuracy: {optimized_error_10mm:.1f}%")
            print(f"  Adaptive Ensemble - MAE: {adaptive_mae:.3f} mm, R²: {adaptive_r2:.3f}")
            print(f"    <5mm accuracy: {adaptive_error_5mm:.1f}%, <10mm accuracy: {adaptive_error_10mm:.1f}%")
            print(f"  Standard Ensemble - MAE: {standard_mae:.3f} mm, R²: {standard_r2:.3f}")
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
        print(f"  Average R²: {optimized_r2:.3f}")
        print(f"  Average <5mm accuracy: {optimized_5mm:.1f}%")
        print(f"  Average <10mm accuracy: {optimized_10mm:.1f}%")
        
        print(f"\nAdaptive Voting Ensemble:")
        print(f"  Average MAE: {adaptive_mae:.3f} mm")
        print(f"  Average RMSE: {adaptive_rmse:.3f} mm")
        print(f"  Average R²: {adaptive_r2:.3f}")
        print(f"  Average <5mm accuracy: {adaptive_5mm:.1f}%")
        print(f"  Average <10mm accuracy: {adaptive_10mm:.1f}%")
        
        print(f"\nStandard Voting Ensemble:")
        print(f"  Average MAE: {standard_mae:.3f} mm")
        print(f"  Average RMSE: {standard_rmse:.3f} mm")
        print(f"  Average R²: {standard_r2:.3f}")
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
            print(f"  Optimized Ensemble - MAE: {metrics['Optimized_MAE']:.3f} mm (R²: {metrics['Optimized_R2']:.3f})")
            print(f"  Adaptive Ensemble - MAE: {metrics['Adaptive_MAE']:.3f} mm (R²: {metrics['Adaptive_R2']:.3f})")
            print(f"  Standard Ensemble - MAE: {metrics['Standard_MAE']:.3f} mm (R²: {metrics['Standard_R2']:.3f})")
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
            'best_models': self.best_models
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
    
    # 1. Load integrated data
    combined_df, train_df, val_df = trainer.load_integrated_data()
    if combined_df is None:
        print("Failed to load integrated data")
        return
    
    # 2. Prepare training data with enhanced features
    X_train, X_test, y_train, y_test = trainer.prepare_training_data(combined_df)
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
