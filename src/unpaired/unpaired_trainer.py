import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys

# Добавляем src в Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from features.advanced_features import AdvancedFeatureEngineer
from preprocessing.unified_pipeline import UnifiedPreprocessingPipeline
from versioning.version_manager import VersionManager

logger = logging.getLogger(__name__)

class UnpairedDataProcessor:
    """Обработка непарных данных"""
    
    def __init__(self):
        self.unpaired_stats = {}
        self.distribution_thresholds = {}
        self.feature_columns = []
    
    def fit_unpaired_data(self, unpaired_ct_data: pd.DataFrame):
        """
        Обучение на непарных данных
        
        Args:
            unpaired_ct_data: данные КТ без парных измерений
        """
        logger.info(f"Обучение на {len(unpaired_ct_data)} непарных данных")
        
        # Расчет статистики
        for column in unpaired_ct_data.columns:
            if unpaired_ct_data[column].dtype in ['float64', 'int64']:
                col_data = unpaired_ct_data[column].dropna()
                if len(col_data) > 0:
                    self.unpaired_stats[column] = {
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'median': col_data.median(),
                        'q25': col_data.quantile(0.25),
                        'q75': col_data.quantile(0.75)
                    }
                    
                    # Пороги для аномалий (3σ)
                    if self.unpaired_stats[column]['std'] > 0:
                        self.distribution_thresholds[column] = (
                            self.unpaired_stats[column]['mean'] - 3 * self.unpaired_stats[column]['std'],
                            self.unpaired_stats[column]['mean'] + 3 * self.unpaired_stats[column]['std']
                        )
        
        self.feature_columns = list(self.unpaired_stats.keys())
        logger.info(f"Рассчитана статистика для {len(self.feature_columns)} признаков")
    
    def validate_with_unpaired(self, features: Dict) -> List[str]:
        """
        Валидация с использованием непарных данных
        
        Args:
            features: признаки пациента
            
        Returns:
            список аномалий
        """
        anomalies = []
        
        for feature, value in features.items():
            if feature in self.distribution_thresholds:
                min_thresh, max_thresh = self.distribution_thresholds[feature]
                if not np.isnan(value) and not (min_thresh <= value <= max_thresh):
                    anomalies.append(f"{feature} anomaly: {value:.2f} (expected {min_thresh:.2f}-{max_thresh:.2f})")
        
        return anomalies
    
    def get_feature_statistics(self) -> Dict:
        """Получение статистики признаков"""
        return self.unpaired_stats
    
    def normalize_features(self, features: Dict) -> Dict:
        """
        Нормализация признаков на основе статистики непарных данных
        
        Args:
            features: исходные признаки
            
        Returns:
            нормализованные признаки
        """
        normalized = features.copy()
        
        for feature, value in features.items():
            if feature in self.unpaired_stats and not np.isnan(value):
                stats = self.unpaired_stats[feature]
                if stats['std'] > 0:
                    # Z-score нормализация
                    normalized[f"{feature}_zscore"] = (value - stats['mean']) / stats['std']
                    
                    # Min-max нормализация
                    if stats['max'] != stats['min']:
                        normalized[f"{feature}_norm"] = (value - stats['min']) / (stats['max'] - stats['min'])
        
        return normalized

class EnhancedModelTrainer:
    """Расширенный тренер моделей с использованием непарных данных"""
    
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.unpaired_processor = UnpairedDataProcessor()
        self.version_manager = VersionManager()
        self.models = {}
        self.metrics = {}
        
    def train_with_unpaired_data(self, paired_data: pd.DataFrame, 
                               unpaired_data: pd.DataFrame,
                               target_cols: List[str]) -> Dict:
        """
        Обучение с использованием непарных данных
        
        Args:
            paired_data: парные данные (с разметкой)
            unpaired_data: непарные данные (только КТ)
            target_cols: целевые колонки
            
        Returns:
            результаты обучения
        """
        logger.info("Начало обучения с непарными данными")
        
        # 1. Обучение на непарных данных
        self.unpaired_processor.fit_unpaired_data(unpaired_data)
        
        # 2. Feature engineering для парных данных
        paired_enhanced = self.feature_engineer.engineer_features(paired_data)
        
        # 3. Валидация с использованием статистики непарных данных
        validation_results = []
        for idx, row in paired_enhanced.iterrows():
            anomalies = self.unpaired_processor.validate_with_unpaired(row.to_dict())
            if anomalies:
                validation_results.append({
                    'index': idx,
                    'anomalies': anomalies
                })
        
        logger.info(f"Найдено аномалий в {len(validation_results)} из {len(paired_enhanced)} записей")
        
        # 4. Подготовка данных для обучения
        feature_cols = [col for col in self.feature_engineer.get_feature_names() 
                       if col in paired_enhanced.columns]
        
        X = paired_enhanced[feature_cols].fillna(0)
        y = paired_enhanced[target_cols].fillna(0)
        
        # 5. Разделение данных
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 6. Обучение моделей
        models = {}
        for i, target_col in enumerate(target_cols):
            logger.info(f"Обучение модели для {target_col}")
            
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train.iloc[:, i])
            
            # Валидация
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val.iloc[:, i], y_pred)
            
            models[target_col] = {
                'model': model,
                'mae': mae,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
            logger.info(f"{target_col} - MAE: {mae:.3f}")
        
        # 7. Сохранение моделей
        self.models = models
        self._save_models()
        
        # 8. Расчет метрик
        overall_metrics = self._calculate_overall_metrics(models, X_val, y_val)
        self.metrics = overall_metrics
        
        # 9. Сохранение результатов
        results = {
            'models_count': len(models),
            'validation_anomalies': len(validation_results),
            'overall_metrics': overall_metrics,
            'feature_engineering_features': len(feature_cols),
            'unpaired_stats_features': len(self.unpaired_processor.feature_columns)
        }
        
        # Сохранение снепшота
        snapshot_id = self.version_manager.create_version_snapshot(
            f"Training with {len(paired_data)} paired + {len(unpaired_data)} unpaired samples"
        )
        results['snapshot_id'] = snapshot_id
        
        logger.info(f"Обучение завершено. Снепшот: {snapshot_id}")
        
        return results
    
    def _save_models(self):
        """Сохранение моделей с версионированием"""
        for target_col, model_data in self.models.items():
            # Сохранение модели
            self.version_manager.save_versioned_artifact(
                model_data['model'],
                'model',
                f"{target_col}_v1.0",
                metadata={
                    'target': target_col,
                    'mae': model_data['mae'],
                    'feature_importance': model_data['feature_importance']
                }
            )
    
    def _calculate_overall_metrics(self, models: Dict, X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict:
        """Расчет общих метрик"""
        predictions = []
        targets = []
        
        for target_col in y_val.columns:
            if target_col in models:
                pred = models[target_col]['model'].predict(X_val)
                predictions.append(pred)
                targets.append(y_val[target_col].values)
        
        if predictions:
            predictions = np.column_stack(predictions)
            targets = np.column_stack(targets)
            
            # Общий MAE
            overall_mae = mean_absolute_error(targets, predictions)
            
            # MAE по осям
            axis_maes = mean_absolute_error(targets, predictions, multioutput='raw_values')
            
            # Процент предсказаний в пределах 5 мм
            within_5mm = np.mean(np.abs(targets - predictions) < 5) * 100
            
            return {
                'overall_mae': float(overall_mae),
                'axis_maes': axis_maes.tolist(),
                'within_5mm_percent': float(within_5mm),
                'total_samples': len(X_val)
            }
        
        return {}
    
    def predict_with_confidence(self, features: Dict) -> Tuple[np.ndarray, float]:
        """
        Предсказание с оценкой уверенности
        
        Args:
            features: признаки пациента
            
        Returns:
            предсказание и уверенность
        """
        # 1. Валидация с непарными данными
        anomalies = self.unpaired_processor.validate_with_unpaired(features)
        
        # 2. Feature engineering
        df = pd.DataFrame([features])
        enhanced_features = self.feature_engineer.engineer_features(df)
        
        # 3. Предсказание
        feature_cols = [col for col in self.feature_engineer.get_feature_names() 
                       if col in enhanced_features.columns]
        
        X = enhanced_features[feature_cols].fillna(0)
        
        predictions = []
        for target_col in sorted(self.models.keys()):
            if target_col in self.models:
                pred = self.models[target_col]['model'].predict(X)
                predictions.append(pred[0])
        
        prediction = np.array(predictions) if predictions else np.array([0, 0, 0, 0, 0, 0])
        
        # 4. Оценка уверенности
        confidence = self._calculate_prediction_confidence(anomalies, len(features))
        
        return prediction, confidence
    
    def _calculate_prediction_confidence(self, anomalies: List[str], feature_count: int) -> float:
        """Расчет уверенности предсказания"""
        base_confidence = 0.8
        
        # Снижаем уверенность за аномалии
        anomaly_penalty = min(len(anomalies) * 0.1, 0.5)
        
        # Снижаем за пропущенные признаки
        missing_penalty = max(0, (36 - feature_count) * 0.02)  # ожидаем 36 признаков
        
        confidence = base_confidence - anomaly_penalty - missing_penalty
        return max(0.1, min(1.0, confidence))

if __name__ == "__main__":
    # Тестирование с непарными данными
    logging.basicConfig(level=logging.INFO)
    logger.info("Тестирование обучения с непарными данными")
    
    trainer = EnhancedModelTrainer()
    
    # Создание тестовых данных
    # Парные данные (с разметкой)
    paired_data = pd.DataFrame({
        'age': [25, 35, 45, 55, 65],
        'bmi': [22.5, 24.0, 26.5, 28.0, 30.5],
        'kidney_left_center_x_mm': [-45.2, -48.1, -42.3, -46.7, -44.5],
        'kidney_right_center_x_mm': [52.1, 49.8, 54.3, 51.2, 53.7],
        'delta_left_x': [5.2, 4.8, 5.5, 4.9, 5.1],
        'delta_left_y': [-3.1, -2.9, -3.3, -3.0, -3.2],
        'delta_left_z': [2.8, 2.6, 3.0, 2.7, 2.9],
        'delta_right_x': [5.0, 4.7, 5.3, 4.8, 5.2],
        'delta_right_y': [-3.0, -2.8, -3.1, -2.9, -3.0],
        'delta_right_z': [2.7, 2.5, 2.9, 2.6, 2.8]
    })
    
    # Непарные данные (только КТ)
    unpaired_data = pd.DataFrame({
        'age': [20, 30, 40, 50, 60, 70, 80],
        'bmi': [18.0, 20.0, 22.0, 25.0, 27.0, 29.0, 32.0],
        'kidney_left_center_x_mm': [-43.0, -46.0, -44.0, -47.0, -45.0, -48.0, -46.5],
        'kidney_right_center_x_mm': [50.0, 53.0, 51.0, 54.0, 52.0, 55.0, 53.5],
        'kidney_left_center_y_mm': [15.0, 18.0, 20.0, 22.0, 19.0, 21.0, 20.5],
        'kidney_right_center_y_mm': [16.0, 19.0, 21.0, 23.0, 20.0, 22.0, 21.5]
    })
    
    # Обучение
    target_cols = ['delta_left_x', 'delta_left_y', 'delta_left_z', 
                  'delta_right_x', 'delta_right_y', 'delta_right_z']
    
    results = trainer.train_with_unpaired_data(paired_data, unpaired_data, target_cols)
    
    print("Результаты обучения:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Тестовое предсказание
    test_features = {
        'age': 42,
        'bmi': 25.5,
        'kidney_left_center_x_mm': -45.0,
        'kidney_right_center_x_mm': 52.5
    }
    
    prediction, confidence = trainer.predict_with_confidence(test_features)
    
    print(f"\nТестовое предсказание:")
    print(f"  Предсказание: {prediction}")
    print(f"  Уверенность: {confidence:.3f}")
    
    logger.info("Обучение с непарными данными протестировано успешно")
