"""
[LEGACY] FeatureSchema_v1 (36 features) — used by AR/unpaired branch only.

Phase 1 displacement model uses src/features/phase1_schema.py instead.
"""

import warnings

warnings.warn(
    "unified_pipeline.FeatureSchema_v1 is not the Phase 1 production schema. "
    "Use src/features/phase1_schema.py for kidney displacement training/inference.",
    DeprecationWarning,
    stacklevel=1,
)

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FeatureSchema_v1:
    """Фиксированная схема признаков версия 1.0"""
    
    # Финальный список признаков (36 признаков)
    FEATURE_SCHEMA = {
        # Демографические (6)
        'age': float,
        'sex_encoded': float,
        'bmi': float,
        'body_type_encoded': float,
        'weight_kg': float,
        'height_m': float,
        
        # Относительные координаты (8)
        'kidney_left_x_rel_spine': float,
        'kidney_left_y_rel_spine': float,
        'kidney_right_x_rel_spine': float,
        'kidney_right_y_rel_spine': float,
        'kidney_left_x_norm': float,
        'kidney_left_y_norm': float,
        'kidney_right_x_norm': float,
        'kidney_right_y_norm': float,
        
        # Анатомические отношения (6)
        'kidney_left_spine_dist': float,
        'kidney_right_spine_dist': float,
        'kidney_left_skin_ratio': float,
        'kidney_right_skin_ratio': float,
        'kidneys_symmetry_x': float,
        'kidneys_asymmetry_x': float,
        
        # Геометрические (8)
        'kidney_left_axis_vector_z': float,
        'kidney_right_axis_vector_z': float,
        'kidney_left_tilt_angle': float,
        'kidney_right_tilt_angle': float,
        'kidney_left_aspect_ratio': float,
        'kidney_right_aspect_ratio': float,
        'kidney_left_volume_est': float,
        'kidney_right_volume_est': float,
        
        # Композиция тела (4)
        'fat_ratio': float,
        'bone_ratio': float,
        'obesity_class': float,
        'bmi_normalized': float,
        
        # Положение пациента (4)
        'weight_normalized': float,
        'height_normalized': float,
        'age_group': float,
        'body_width_mm_median': float
    }
    
    def __init__(self):
        self.feature_names = list(self.FEATURE_SCHEMA.keys())
        self.feature_types = self.FEATURE_SCHEMA
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение признаков согласно схеме"""
        logger.info(f"Извлечение {len(self.feature_names)} признаков по схеме v1.0")
        
        # Создаем DataFrame с нужными колонками
        features_df = pd.DataFrame(index=df.index)
        
        for feature_name in self.feature_names:
            if feature_name in df.columns:
                features_df[feature_name] = df[feature_name]
            else:
                # Если признака нет, заполняем NaN
                features_df[feature_name] = np.nan
                logger.warning(f"Признак {feature_name} отсутствует в данных, заполнен NaN")
        
        return features_df
    
    def validate_features(self, df: pd.DataFrame) -> Dict:
        """Валидация признаков"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_features': [],
            'extra_features': []
        }
        
        # Проверка наличия всех требуемых признаков
        for feature in self.feature_names:
            if feature not in df.columns:
                validation_result['missing_features'].append(feature)
                validation_result['errors'].append(f"Missing required feature: {feature}")
                validation_result['is_valid'] = False
        
        # Проверка лишних признаков
        for feature in df.columns:
            if feature not in self.feature_names:
                validation_result['extra_features'].append(feature)
                validation_result['warnings'].append(f"Extra feature found: {feature}")
        
        # Проверка типов данных
        for feature in self.feature_names:
            if feature in df.columns:
                expected_type = self.feature_types[feature]
                actual_type = df[feature].dtype
                
                if not self._is_compatible_type(actual_type, expected_type):
                    validation_result['warnings'].append(
                        f"Type mismatch for {feature}: expected {expected_type}, got {actual_type}"
                    )
        
        return validation_result
    
    def _is_compatible_type(self, actual_type, expected_type) -> bool:
        """Проверка совместимости типов данных"""
        # pandas dtype compatibility
        if expected_type == float:
            return actual_type in ['float64', 'float32', 'int64', 'int32']
        return str(actual_type) == str(expected_type)

class UnifiedPreprocessingPipeline:
    """Единый pipeline подготовки данных"""
    
    def __init__(self, feature_schema_version: str = "v1"):
        self.scaler = StandardScaler()
        
        if feature_schema_version == "v1":
            self.feature_schema = FeatureSchema_v1()
        else:
            raise ValueError(f"Unsupported feature schema version: {feature_schema_version}")
        
        self.is_fitted = False
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame = None):
        """Обучение pipeline на тренировочных данных"""
        logger.info("Обучение preprocessing pipeline")
        
        # 1. Извлечение признаков по схеме
        X_features = self.feature_schema.extract_features(X_train)
        
        # 2. Валидация признаков
        validation = self.feature_schema.validate_features(X_features)
        if not validation['is_valid']:
            logger.error(f"Feature validation failed: {validation['errors']}")
            raise ValueError("Feature validation failed")
        
        # 3. Заполнение пропусков
        X_filled = self._fill_missing_values(X_features)
        
        # 4. Обучение scaler
        self.scaler.fit(X_filled)
        
        self.is_fitted = True
        logger.info("Pipeline обучен успешно")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Применение pipeline к новым данным"""
        if not self.is_fitted:
            raise ValueError("Pipeline не обучен. Вызовите fit() сначала.")
        
        logger.info("Применение preprocessing pipeline")
        
        # 1. Извлечение признаков по схеме
        X_features = self.feature_schema.extract_features(X)
        
        # 2. Валидация признаков
        validation = self.feature_schema.validate_features(X_features)
        if not validation['is_valid']:
            logger.warning(f"Feature validation warnings: {validation['warnings']}")
        
        # 3. Заполнение пропусков
        X_filled = self._fill_missing_values(X_features)
        
        # 4. Масштабирование
        X_scaled = self.scaler.transform(X_filled)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> np.ndarray:
        """Обучение и применение за один вызов"""
        return self.fit(X, y).transform(X)
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Заполнение пропусков"""
        logger.info("Заполнение пропусков в данных")
        
        filled_df = df.copy()
        
        # Числовые признаки - медиана
        for col in filled_df.columns:
            if filled_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                median_val = filled_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                filled_df[col] = filled_df[col].fillna(median_val)
                
                # Добавляем индикатор пропуска
                filled_df[f'missing_{col}'] = (df[col].isna()).astype(float)
        
        return filled_df
    
    def get_feature_names(self) -> List[str]:
        """Получение списка признаков"""
        return self.feature_schema.feature_names
    
    def get_scaler(self) -> StandardScaler:
        """Получение обученного scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler не обучен")
        return self.scaler
    
    def save_pipeline(self, filepath: str):
        """Сохранение pipeline"""
        import joblib
        
        pipeline_data = {
            'scaler': self.scaler,
            'feature_schema': self.feature_schema,
            'is_fitted': self.is_fitted,
            'feature_names': self.get_feature_names()
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline сохранен в {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: str):
        """Загрузка pipeline"""
        import joblib
        
        pipeline_data = joblib.load(filepath)
        
        pipeline = cls()
        pipeline.scaler = pipeline_data['scaler']
        pipeline.feature_schema = pipeline_data['feature_schema']
        pipeline.is_fitted = pipeline_data['is_fitted']
        
        logger.info(f"Pipeline загружен из {filepath}")
        return pipeline

if __name__ == "__main__":
    # Тестирование pipeline
    logger.info("Тестирование preprocessing pipeline")
    
    # Создание тестовых данных
    test_data = pd.DataFrame({
        'age': [25, 35, 45, 55, 65],
        'sex_encoded': [1, 0, 1, 0, 1],
        'bmi': [22.5, 24.0, 26.5, 28.0, 30.5],
        'kidney_left_x_rel_spine': [-45.2, -48.1, -42.3, -46.7, -44.5],
        'kidney_right_x_rel_spine': [52.1, 49.8, 54.3, 51.2, 53.7],
        'extra_feature': [1, 2, 3, 4, 5]  # лишний признак
    })
    
    # Создание pipeline
    pipeline = UnifiedPreprocessingPipeline()
    
    # Обучение
    pipeline.fit(test_data)
    
    # Применение
    transformed = pipeline.transform(test_data)
    
    print(f"Исходные данные: {test_data.shape}")
    print(f"Преобразованные данные: {transformed.shape}")
    print(f"Признаки: {pipeline.get_feature_names()}")
    
    logger.info("Preprocessing pipeline протестирован успешно")
