import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import joblib
import logging

logger = logging.getLogger(__name__)

class ConfidenceEstimator:
    """Оценка уверенности предсказания"""
    
    def __init__(self, models: List, train_data: np.ndarray):
        """
        Args:
            models: список обученных моделей ансамбля
            train_data: обучающие данные для оценки расстояния
        """
        self.models = models
        self.train_data = train_data
        self.feature_std = np.std(train_data, axis=0)
        
    def calculate_confidence(self, features: np.ndarray) -> float:
        """
        Расчет уверенности предсказания (0-1)
        
        Args:
            features: признаки пациента
            
        Returns:
            confidence_score: уверенность от 0 до 1
        """
        # 1. Distance to training data (40%)
        dist_score = self._distance_confidence(features)
        
        # 2. Model stability (40%)
        stability_score = self._stability_confidence(features)
        
        # 3. Data quality (20%)
        quality_score = self._quality_confidence(features)
        
        # Комбинированный score
        confidence = (dist_score * 0.4 + stability_score * 0.4 + quality_score * 0.2)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _distance_confidence(self, features: np.ndarray) -> float:
        """Уверенность на основе расстояния до train данных"""
        # Расстояние до ближайших train примеров
        distances = np.linalg.norm(self.train_data - features, axis=1)
        min_dist = np.min(distances)
        
        # Чем дальше, тем ниже уверенность
        # Используем экспоненциальное затухание
        avg_std = np.mean(self.feature_std)
        confidence = np.exp(-min_dist / (2 * avg_std))
        
        return confidence
    
    def _stability_confidence(self, features: np.ndarray) -> float:
        """Уверенность на основе стабильности моделей"""
        predictions = []
        
        # Предсказания от всех моделей ансамбля
        for model in self.models:
            pred = model.predict(features.reshape(1, -1))
            predictions.append(pred)
        
        # Дисперсия предсказаний
        variance = np.var(predictions, axis=0)
        avg_variance = np.mean(variance)
        
        # Чем меньше variance, тем выше уверенность
        confidence = 1.0 / (1.0 + avg_variance)
        
        return confidence
    
    def _quality_confidence(self, features: np.ndarray) -> float:
        """Уверенность на основе качества данных"""
        # Проверка пропусков
        missing_ratio = np.isnan(features).sum() / len(features)
        missing_score = 1.0 - missing_ratio
        
        # Проверка выбросов (простая эвристика)
        outlier_score = self._check_outliers(features)
        
        return (missing_score + outlier_score) / 2.0
    
    def _check_outliers(self, features: np.ndarray) -> float:
        """Проверка выбросов в признаках"""
        # Простая проверка на экстремальные значения
        # В реальной системе здесь была бы более сложная логика
        
        # Проверяем значения > 5 стандартных отклонений
        outliers = np.abs(features) > 5 * self.feature_std
        outlier_ratio = np.sum(outliers) / len(features)
        
        return 1.0 - outlier_ratio

class AnatomicalConstraints:
    """Анатомические ограничения"""
    
    def __init__(self, body_limits: Dict, spine_center: np.ndarray):
        """
        Args:
            body_limits: пределы тела {x_min, x_max, y_min, y_max, z_min, z_max}
            spine_center: центр позвоночника
        """
        self.body_limits = body_limits
        self.spine_center = spine_center
        self.max_displacement = 50.0  # мм по каждой оси
        self.max_total_displacement = 80.0  # мм общая длина
        
    def apply_constraints(self, original_pos: np.ndarray, predicted_pos: np.ndarray) -> np.ndarray:
        """
        Применение анатомических ограничений
        
        Args:
            original_pos: исходное положение почки
            predicted_pos: предсказанное положение
            
        Returns:
            constrained_pos: скорректированное положение
        """
        # 1. Ограничение смещения
        displacement = predicted_pos - original_pos
        displacement = self._clamp_displacement(displacement)
        constrained_pos = original_pos + displacement
        
        # 2. Проверка пересечения позвоночника
        if self._crosses_spine(constrained_pos):
            constrained_pos = self._resolve_spine_collision(original_pos, constrained_pos)
        
        # 3. Проверка выхода за пределы тела
        if not self._inside_body(constrained_pos):
            constrained_pos = self._clamp_to_body(constrained_pos)
        
        return constrained_pos
    
    def _clamp_displacement(self, displacement: np.ndarray) -> np.ndarray:
        """Ограничение величины смещения"""
        # Ограничение по каждой оси
        displacement = np.clip(displacement, -self.max_displacement, self.max_displacement)
        
        # Ограничение общей величины
        total_disp = np.linalg.norm(displacement)
        if total_disp > self.max_total_displacement:
            displacement = displacement * (self.max_total_displacement / total_disp)
        
        return displacement
    
    def _crosses_spine(self, position: np.ndarray) -> bool:
        """Проверка пересечения с позвоночником"""
        distance_to_spine = np.linalg.norm(position - self.spine_center)
        return distance_to_spine < 20.0  # минимальное расстояние 20 мм
    
    def _resolve_spine_collision(self, original_pos: np.ndarray, constrained_pos: np.ndarray) -> np.ndarray:
        """Разрешение столкновения с позвоночником"""
        # Простая логика: отодвигаем от позвоночника в направлении от него
        direction = constrained_pos - self.spine_center
        direction_norm = direction / np.linalg.norm(direction)
        
        # Устанавливаем минимальное расстояние
        min_distance = 25.0  # мм
        corrected_pos = self.spine_center + direction_norm * min_distance
        
        return corrected_pos
    
    def _inside_body(self, position: np.ndarray) -> bool:
        """Проверка нахождения внутри тела"""
        return (self.body_limits['x_min'] <= position[0] <= self.body_limits['x_max'] and
                self.body_limits['y_min'] <= position[1] <= self.body_limits['y_max'] and
                self.body_limits['z_min'] <= position[2] <= self.body_limits['z_max'])
    
    def _clamp_to_body(self, position: np.ndarray) -> np.ndarray:
        """Ограничение пределами тела"""
        clamped_pos = position.copy()
        clamped_pos[0] = np.clip(clamped_pos[0], self.body_limits['x_min'], self.body_limits['x_max'])
        clamped_pos[1] = np.clip(clamped_pos[1], self.body_limits['y_min'], self.body_limits['y_max'])
        clamped_pos[2] = np.clip(clamped_pos[2], self.body_limits['z_min'], self.body_limits['z_max'])
        
        return clamped_pos

class FallbackHandler:
    """Fallback и обработка ошибок"""
    
    def __init__(self, statistical_model, constraints: AnatomicalConstraints):
        """
        Args:
            statistical_model: fallback модель (статистическая)
            constraints: анатомические ограничения
        """
        self.statistical_model = statistical_model
        self.constraints = constraints
    
    def handle_prediction(self, features: np.ndarray, ml_prediction: np.ndarray, 
                         confidence: float, original_position: np.ndarray) -> np.ndarray:
        """
        Обработка предсказания с fallback
        
        Args:
            features: признаки пациента
            ml_prediction: предсказание ML модели
            confidence: уверенность предсказания
            original_position: исходное положение почки
            
        Returns:
            final_prediction: финальное предсказание
        """
        # 1. Проверка confidence
        if confidence < 0.3:
            logger.warning(f"Low confidence: {confidence:.3f}")
            return self._get_fallback_prediction(features)
        
        # 2. Проверка на аномалии
        if self._is_anomalous(ml_prediction):
            logger.warning("Anomalous prediction detected")
            return self._clamp_prediction(ml_prediction)
        
        # 3. Применение ограничений
        constrained_prediction = self.constraints.apply_constraints(original_position, ml_prediction)
        
        return constrained_prediction
    
    def _get_fallback_prediction(self, features: np.ndarray) -> np.ndarray:
        """Fallback предсказание на основе статистики"""
        # Простая статистическая модель: среднее смещение
        # В реальной системе здесь была бы более сложная логика
        mean_displacement = np.array([5.0, -3.0, 2.0])  # среднее смещение по осям
        return mean_displacement
    
    def _is_anomalous(self, prediction: np.ndarray) -> bool:
        """Проверка на аномальность предсказания"""
        # Проверка на экстремальные значения
        return np.any(np.abs(prediction) > 100.0)  # мм
    
    def _clamp_prediction(self, prediction: np.ndarray) -> np.ndarray:
        """Ограничение аномального предсказания"""
        return np.clip(prediction, -50.0, 50.0)

class TemporalSmoother:
    """Временное сглаживание"""
    
    def __init__(self, method: str = 'exponential', alpha: float = 0.7):
        """
        Args:
            method: метод сглаживания ('exponential' или 'kalman')
            alpha: параметр сглаживания для экспоненциального метода
        """
        self.method = method
        self.alpha = alpha
        self.history = []
        self.max_history = 10
        
        if method == 'kalman':
            self.kalman = self._init_kalman_filter()
    
    def smooth(self, current_prediction: np.ndarray) -> np.ndarray:
        """Сглаживание предсказания"""
        if self.method == 'exponential':
            return self._exponential_smoothing(current_prediction)
        elif self.method == 'kalman':
            return self._kalman_smoothing(current_prediction)
        else:
            return current_prediction
    
    def _exponential_smoothing(self, current: np.ndarray) -> np.ndarray:
        """Экспоненциальное сглаживание"""
        if len(self.history) == 0:
            smoothed = current
        else:
            smoothed = self.alpha * current + (1 - self.alpha) * self.history[-1]
        
        self.history.append(smoothed)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return smoothed
    
    def _kalman_smoothing(self, current: np.ndarray) -> np.ndarray:
        """Kalman filter сглаживание"""
        # Упрощенная реализация Kalman filter
        # В реальной системе здесь была бы полноценная реализация
        
        if len(self.history) == 0:
            # Инициализация
            self.state = current
            self.covariance = np.eye(len(current)) * 1.0
        else:
            # Предсказание
            predicted_state = self.state  # простая модель постоянного состояния
            predicted_covariance = self.covariance + np.eye(len(current)) * 0.1
            
            # Обновление измерением
            kalman_gain = predicted_covariance / (predicted_covariance + np.eye(len(current)) * 0.5)
            self.state = predicted_state + kalman_gain * (current - predicted_state)
            self.covariance = (np.eye(len(current)) - kalman_gain) * predicted_covariance
        
        self.history.append(self.state)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return self.state
    
    def _init_kalman_filter(self):
        """Инициализация Kalman filter"""
        # Заглушка для Kalman filter
        return None
    
    def reset(self):
        """Сброс истории сглаживания"""
        self.history = []
        if self.method == 'kalman':
            self.kalman = self._init_kalman_filter()

if __name__ == "__main__":
    # Тестирование компонентов надежности
    logger.info("Тестирование компонентов надежности")
    
    # Создание тестовых данных
    train_data = np.random.randn(100, 10)
    test_features = np.random.randn(10)
    
    # Тестирование Confidence Estimator
    models = [RandomForestRegressor(n_estimators=10) for _ in range(3)]
    for model in models:
        model.fit(train_data, np.random.randn(100, 3))
    
    confidence_estimator = ConfidenceEstimator(models, train_data)
    confidence = confidence_estimator.calculate_confidence(test_features)
    print(f"Confidence score: {confidence:.3f}")
    
    # Тестирование Anatomical Constraints
    body_limits = {'x_min': -150, 'x_max': 150, 'y_min': -100, 'y_max': 100, 'z_min': 50, 'z_max': 150}
    spine_center = np.array([0, 0, 100])
    
    constraints = AnatomicalConstraints(body_limits, spine_center)
    original_pos = np.array([-50, 20, 95])
    predicted_pos = np.array([-10, 30, 110])
    
    constrained_pos = constraints.apply_constraints(original_pos, predicted_pos)
    print(f"Original: {original_pos}")
    print(f"Predicted: {predicted_pos}")
    print(f"Constrained: {constrained_pos}")
    
    # Тестирование Temporal Smoothing
    smoother = TemporalSmoother(method='exponential', alpha=0.7)
    
    predictions = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.5, 2.5, 3.5]),
        np.array([1.2, 2.2, 3.2]),
        np.array([1.8, 2.8, 3.8])
    ]
    
    for i, pred in enumerate(predictions):
        smoothed = smoother.smooth(pred)
        print(f"Prediction {i+1}: {pred} -> Smoothed: {smoothed}")
    
    logger.info("Компоненты надежности протестированы успешно")
