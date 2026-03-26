#!/usr/bin/env python3
"""
Модуль для расчета confidence scores для предсказаний
Оценка уверенности модели в предсказанных координатах
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings


@dataclass
class ConfidenceScore:
    """Результат оценки уверенности"""
    overall_confidence: float
    per_axis_confidence: Dict[str, float]
    per_point_confidence: List[float]
    distance_to_train: float
    ensemble_variance: float
    feature_similarity: float
    constraint_violation_penalty: float
    breakdown: Dict[str, Any]


class ConfidenceEstimator:
    """
    Класс для оценки уверенности в предсказаниях
    """
    
    def __init__(self,
                 method: str = "ensemble",
                 n_neighbors: int = 5):
        """
        Инициализация оценщика уверенности
        
        Args:
            method: Метод оценки ('ensemble', 'distance', 'variance', 'hybrid')
            n_neighbors: Количество соседей для distance-based метода
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.train_features = None
        self.train_targets = None
        self.nn_model = None
        self.feature_importance = None
        
    def fit(self, train_features: np.ndarray, train_targets: np.ndarray):
        """
        Обучить оценщик на тренировочных данных
        
        Args:
            train_features: Признаки тренировочных данных
            train_targets: Целевые значения тренировочных данных
        """
        self.train_features = train_features
        self.train_targets = train_targets
        
        # Нормализация признаков
        self.scaler.fit(train_features)
        scaled_features = self.scaler.transform(train_features)
        
        # Обучение модели ближайших соседей
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        self.nn_model.fit(scaled_features)
        
        # Расчет важности признаков (если возможно)
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(scaled_features, train_targets)
            if train_targets.ndim == 1:
                self.feature_importance = rf.feature_importances_
            else:
                self.feature_importance = np.mean(rf.feature_importances_, axis=0)
        except Exception:
            self.feature_importance = np.ones(train_features.shape[1]) / train_features.shape[1]
    
    def predict_confidence(self,
                         test_features: np.ndarray,
                         model_predictions: Optional[np.ndarray] = None,
                         ensemble_predictions: Optional[Dict[str, np.ndarray]] = None,
                         constraint_violations: Optional[Dict[str, Any]] = None) -> List[ConfidenceScore]:
        """
        Предсказать уверенность для набора данных
        
        Args:
            test_features: Признаки тестовых данных
            model_predictions: Предсказания основной модели
            ensemble_predictions: Предсказания ансамбля {'model_name': predictions}
            constraint_violations: Результаты проверки ограничений
            
        Returns:
            Список ConfidenceScore для каждого примера
        """
        if self.train_features is None:
            raise ValueError("Модель не обучена. Вызовите fit()")
        
        confidence_scores = []
        
        # Нормализация тестовых признаков
        scaled_test_features = self.scaler.transform(test_features)
        
        for i, features in enumerate(scaled_test_features):
            confidence = self._calculate_single_confidence(
                features,
                test_features[i] if hasattr(test_features, '__getitem__') else None,
                model_predictions[i] if model_predictions is not None else None,
                {k: v[i] if v.ndim > 1 else v for k, v in (ensemble_predictions or {}).items()},
                constraint_violations[i] if constraint_violations and isinstance(constraint_violations, list) else constraint_violations
            )
            confidence_scores.append(confidence)
        
        return confidence_scores
    
    def _calculate_single_confidence(self,
                                   scaled_features: np.ndarray,
                                   original_features: Optional[np.ndarray] = None,
                                   model_prediction: Optional[np.ndarray] = None,
                                   ensemble_predictions: Optional[Dict[str, np.ndarray]] = None,
                                   constraint_violations: Optional[Dict[str, Any]] = None) -> ConfidenceScore:
        """Рассчитать уверенность для одного примера"""
        
        # Distance-based confidence
        distance_confidence = self._calculate_distance_confidence(scaled_features)
        
        # Variance-based confidence (для ансамбля)
        variance_confidence = self._calculate_variance_confidence(ensemble_predictions)
        
        # Feature similarity confidence
        feature_confidence = self._calculate_feature_similarity(scaled_features)
        
        # Constraint violation penalty
        constraint_penalty = self._calculate_constraint_penalty(constraint_violations)
        
        # Комбинирование оценок
        if self.method == "distance":
            overall_confidence = distance_confidence
        elif self.method == "variance":
            overall_confidence = variance_confidence
        elif self.method == "ensemble":
            # Взвешенное среднее
            overall_confidence = (
                0.4 * distance_confidence +
                0.3 * variance_confidence +
                0.3 * feature_confidence
            )
        else:  # hybrid
            overall_confidence = (
                0.3 * distance_confidence +
                0.3 * variance_confidence +
                0.2 * feature_confidence +
                0.2 * (1.0 - constraint_penalty)
            )
        
        # Учет штрафов за нарушения ограничений
        final_confidence = max(0.0, overall_confidence - constraint_penalty)
        
        # Разбивка по осям (если есть предсказания)
        per_axis_confidence = {}
        if model_prediction is not None:
            if model_prediction.ndim == 1:
                per_axis_confidence = {
                    'x': final_confidence,
                    'y': final_confidence,
                    'z': final_confidence
                }
            else:
                for i, axis in enumerate(['x', 'y', 'z']):
                    per_axis_confidence[axis] = final_confidence
        
        # Дополнительная информация
        breakdown = {
            'distance_confidence': distance_confidence,
            'variance_confidence': variance_confidence,
            'feature_confidence': feature_confidence,
            'constraint_penalty': constraint_penalty,
            'method': self.method
        }
        
        return ConfidenceScore(
            overall_confidence=final_confidence,
            per_axis_confidence=per_axis_confidence,
            per_point_confidence=[final_confidence],
            distance_to_train=distance_confidence,
            ensemble_variance=variance_confidence,
            feature_similarity=feature_confidence,
            constraint_violation_penalty=constraint_penalty,
            breakdown=breakdown
        )
    
    def _calculate_distance_confidence(self, scaled_features: np.ndarray) -> float:
        """Рассчитать уверенность на основе расстояния до тренировочных данных"""
        if self.nn_model is None:
            return 0.5  # Значение по умолчанию
        
        # Найти ближайших соседей
        distances, indices = self.nn_model.kneighbors(scaled_features.reshape(1, -1))
        avg_distance = np.mean(distances[0])
        
        # Нормализация расстояния в confidence (0-1)
        # Используем обратную функцию: confidence = 1 / (1 + distance)
        confidence = 1.0 / (1.0 + avg_distance)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_variance_confidence(self, ensemble_predictions: Optional[Dict[str, np.ndarray]]) -> float:
        """Рассчитать уверенность на основе дисперсии ансамбля"""
        if not ensemble_predictions or len(ensemble_predictions) < 2:
            return 0.5  # Значение по умолчанию
        
        # Собрать все предсказания
        predictions = []
        for model_name, preds in ensemble_predictions.items():
            if preds.ndim == 0:
                predictions.append(preds)
            else:
                predictions.extend(preds.flatten())
        
        if len(predictions) < 2:
            return 0.5
        
        predictions = np.array(predictions)
        variance = np.var(predictions)
        
        # Нормализация дисперсии в confidence
        # Чем меньше дисперсия, тем выше уверенность
        confidence = 1.0 / (1.0 + variance)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_feature_similarity(self, scaled_features: np.ndarray) -> float:
        """Рассчитать уверенность на основе схожести признаков"""
        if self.train_features is None:
            return 0.5
        
        # Расчет схожести с тренировочными данными
        similarities = []
        for train_feat in self.train_features:
            # Косинусная схожесть
            dot_product = np.dot(scaled_features, train_feat)
            norm_product = np.linalg.norm(scaled_features) * np.linalg.norm(train_feat)
            
            if norm_product > 0:
                similarity = dot_product / norm_product
                similarities.append(similarity)
        
        if not similarities:
            return 0.5
        
        # Усредненная схожесть
        avg_similarity = np.mean(similarities)
        
        # Нормализация в 0-1
        confidence = (avg_similarity + 1.0) / 2.0
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_constraint_penalty(self, constraint_violations: Optional[Dict[str, Any]]) -> float:
        """Рассчитать штраф за нарушения ограничений"""
        if not constraint_violations:
            return 0.0
        
        penalty = 0.0
        
        # Штраф за общую валидность
        if not constraint_violations.get('valid', True):
            penalty += 0.3
        
        # Штраф за количество нарушений
        violations = constraint_violations.get('violations', [])
        penalty += len(violations) * 0.1
        
        # Штраф за общий penalty score
        total_penalty = constraint_violations.get('total_penalty', 0.0)
        penalty += min(total_penalty / 100.0, 0.4)  # Нормализованный штраф
        
        return np.clip(penalty, 0.0, 1.0)
    
    def get_confidence_statistics(self, confidence_scores: List[ConfidenceScore]) -> Dict[str, Any]:
        """
        Получить статистику по confidence scores
        
        Args:
            confidence_scores: Список confidence scores
            
        Returns:
            Статистика по уверенности
        """
        if not confidence_scores:
            return {}
        
        overall_confidences = [cs.overall_confidence for cs in confidence_scores]
        
        stats = {
            'mean_confidence': np.mean(overall_confidences),
            'std_confidence': np.std(overall_confidences),
            'min_confidence': np.min(overall_confidences),
            'max_confidence': np.max(overall_confidences),
            'median_confidence': np.median(overall_confidences),
            'low_confidence_count': sum(1 for c in overall_confidences if c < 0.5),
            'high_confidence_count': sum(1 for c in overall_confidences if c > 0.8),
            'total_samples': len(confidence_scores)
        }
        
        # Статистика по осям
        axis_stats = {}
        if confidence_scores[0].per_axis_confidence:
            for axis in ['x', 'y', 'z']:
                axis_confidences = [cs.per_axis_confidence.get(axis, 0) for cs in confidence_scores]
                axis_stats[axis] = {
                    'mean': np.mean(axis_confidences),
                    'std': np.std(axis_confidences),
                    'min': np.min(axis_confidences),
                    'max': np.max(axis_confidences)
                }
        
        stats['per_axis_statistics'] = axis_stats
        
        return stats
    
    def set_confidence_threshold(self, threshold: float):
        """
        Установить порог уверенности
        
        Args:
            threshold: Порог уверенности (0-1)
        """
        self.confidence_threshold = np.clip(threshold, 0.0, 1.0)
    
    def filter_by_confidence(self, 
                           predictions: np.ndarray,
                           confidence_scores: List[ConfidenceScore],
                           threshold: Optional[float] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Отфильтровать предсказания по уверенности
        
        Args:
            predictions: Предсказания
            confidence_scores: Confidence scores
            threshold: Порог уверенности
            
        Returns:
            (отфильтрованные предсказания, индексы отобранных примеров)
        """
        if threshold is None:
            threshold = getattr(self, 'confidence_threshold', 0.5)
        
        high_confidence_indices = [
            i for i, cs in enumerate(confidence_scores)
            if cs.overall_confidence >= threshold
        ]
        
        filtered_predictions = predictions[high_confidence_indices]
        
        return filtered_predictions, high_confidence_indices


class AdaptiveConfidenceThreshold:
    """
    Адаптивный порог уверенности
    """
    
    def __init__(self, target_recall: float = 0.95):
        """
        Инициализация адаптивного порога
        
        Args:
            target_recall: Целевая полнота (0-1)
        """
        self.target_recall = target_recall
        self.threshold_history = []
    
    def calculate_optimal_threshold(self,
                                 confidence_scores: List[ConfidenceScore],
                                 true_errors: np.ndarray,
                                 error_threshold: float = 10.0) -> float:
        """
        Рассчитать оптимальный порог уверенности
        
        Args:
            confidence_scores: Confidence scores
            true_errors: Истинные ошибки
            error_threshold: Порог ошибки для классификации
            
        Returns:
            Оптимальный порог уверенности
        """
        confidences = [cs.overall_confidence for cs in confidence_scores]
        binary_errors = (true_errors > error_threshold).astype(int)
        
        # Перебор порогов для нахождения оптимального
        thresholds = np.linspace(0.0, 1.0, 101)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            predictions = (np.array(confidences) >= threshold).astype(int)
            
            # Расчет метрик
            tp = np.sum((predictions == 1) & (binary_errors == 1))
            fp = np.sum((predictions == 1) & (binary_errors == 0))
            fn = np.sum((predictions == 0) & (binary_errors == 1))
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
                if recall >= self.target_recall:
                    # Точность как дополнительная метрика
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    score = recall * precision  # F1-like score
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
        
        self.threshold_history.append(best_threshold)
        return best_threshold


if __name__ == "__main__":
    # Тестирование confidence scoring
    print("=== Тестирование Confidence Scoring ===")
    
    # Генерация тестовых данных
    np.random.seed(42)
    n_train = 100
    n_test = 10
    n_features = 20
    n_targets = 3
    
    train_features = np.random.randn(n_train, n_features)
    train_targets = np.random.randn(n_train, n_targets)
    test_features = np.random.randn(n_test, n_features)
    
    # Создание и обучение оценщика
    estimator = ConfidenceEstimator(method="ensemble", n_neighbors=5)
    estimator.fit(train_features, train_targets)
    
    # Тестовые предсказания
    model_predictions = np.random.randn(n_test, n_targets)
    ensemble_predictions = {
        'rf': np.random.randn(n_test, n_targets),
        'xgb': np.random.randn(n_test, n_targets),
        'lr': np.random.randn(n_test, n_targets)
    }
    
    # Расчет confidence scores
    confidence_scores = estimator.predict_confidence(
        test_features, model_predictions, ensemble_predictions
    )
    
    print(f"Confidence scores для {len(confidence_scores)} примеров:")
    for i, cs in enumerate(confidence_scores):
        print(f"  Пример {i}: {cs.overall_confidence:.3f}")
        print(f"    Per axis: {cs.per_axis_confidence}")
        print(f"    Distance: {cs.distance_to_train:.3f}")
        print(f"    Variance: {cs.ensemble_variance:.3f}")
        print()
    
    # Статистика
    stats = estimator.get_confidence_statistics(confidence_scores)
    print("Статистика confidence:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Фильтрация по уверенности
    filtered_indices = estimator.filter_by_confidence(
        model_predictions, confidence_scores, threshold=0.6
    )[1]
    
    print(f"\nОтобрано {len(filtered_indices)} примеров с confidence > 0.6")
    print(f"Индексы: {filtered_indices}")
