#!/usr/bin/env python3
"""
Модуль для сглаживания предсказаний координат
Различные методы smoothing для улучшения стабильности результатов
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from scipy import ndimage
from scipy.signal import savgol_filter
import warnings


class SmoothingMethod(Enum):
    """Методы сглаживания"""
    MOVING_AVERAGE = "moving_average"
    GAUSSIAN = "gaussian"
    SAVITZKY_GOLAY = "savitgky_golay"
    EXPONENTIAL = "exponential"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    ADAPTIVE = "adaptive"


@dataclass
class SmoothingConfig:
    """Конфигурация сглаживания"""
    method: SmoothingMethod
    window_size: int = 5
    sigma: float = 1.0
    poly_order: int = 2
    alpha: float = 0.3
    preserve_edges: bool = True
    min_samples: int = 3


class CoordinateSmoother:
    """
    Класс для сглаживания координат почек
    """
    
    def __init__(self, config: Optional[SmoothingConfig] = None):
        """
        Инициализация сглаживателя
        
        Args:
            config: Конфигурация сглаживания
        """
        self.config = config or SmoothingConfig(method=SmoothingMethod.MOVING_AVERAGE)
        self.history_buffer: List[np.ndarray] = []
        self.max_history = 50
        
    def smooth_single_prediction(self, 
                              prediction: np.ndarray,
                              reference_coords: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Сгладить одиночное предсказание
        
        Args:
            prediction: Предсказанные координаты (3,) или (N, 3)
            reference_coords: Референтные координаты для пространственного сглаживания
            
        Returns:
            Сглаженные координаты
        """
        if prediction.ndim == 1:
            prediction = prediction.reshape(1, -1)
        
        smoothed = prediction.copy()
        
        if self.config.method == SmoothingMethod.MOVING_AVERAGE:
            smoothed = self._moving_average_smoothing(prediction)
        elif self.config.method == SmoothingMethod.GAUSSIAN:
            smoothed = self._gaussian_smoothing(prediction)
        elif self.config.method == SmoothingMethod.SAVITZKY_GOLAY:
            smoothed = self._savitgky_golay_smoothing(prediction)
        elif self.config.method == SmoothingMethod.EXPONENTIAL:
            smoothed = self._exponential_smoothing(prediction)
        elif self.config.method == SmoothingMethod.SPATIAL and reference_coords is not None:
            smoothed = self._spatial_smoothing(prediction, reference_coords)
        elif self.config.method == SmoothingMethod.ADAPTIVE:
            smoothed = self._adaptive_smoothing(prediction)
        
        return smoothed.reshape(-1) if prediction.shape[0] == 1 else smoothed
    
    def smooth_temporal_sequence(self, 
                               predictions: np.ndarray,
                               timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Сгладить временную последовательность предсказаний
        
        Args:
            predictions: Последовательность предсказаний (T, N, 3)
            timestamps: Временные метки
            
        Returns:
            Сглаженная последовательность
        """
        if predictions.ndim < 2:
            predictions = predictions.reshape(1, -1, 3) if predictions.ndim == 1 else predictions.reshape(-1, 1, 3)
        
        smoothed = np.zeros_like(predictions)
        
        # Сглаживание для каждой точки independently
        for point_idx in range(predictions.shape[1]):
            for axis_idx in range(predictions.shape[2]):
                signal = predictions[:, point_idx, axis_idx]
                
                if self.config.method == SmoothingMethod.MOVING_AVERAGE:
                    smoothed[:, point_idx, axis_idx] = self._moving_average_1d(signal)
                elif self.config.method == SmoothingMethod.GAUSSIAN:
                    smoothed[:, point_idx, axis_idx] = self._gaussian_1d(signal)
                elif self.config.method == SmoothingMethod.SAVITZKY_GOLAY:
                    smoothed[:, point_idx, axis_idx] = self._savitgky_golay_1d(signal)
                elif self.config.method == SmoothingMethod.EXPONENTIAL:
                    smoothed[:, point_idx, axis_idx] = self._exponential_1d(signal)
                else:
                    # Fallback к moving average
                    smoothed[:, point_idx, axis_idx] = self._moving_average_1d(signal)
        
        return smoothed
    
    def smooth_with_history(self, 
                          new_prediction: np.ndarray,
                          confidence: Optional[float] = None) -> np.ndarray:
        """
        Сгладить с использованием исторических данных
        
        Args:
            new_prediction: Новое предсказание
            confidence: Уверенность в предсказании
            
        Returns:
            Сглаженное предсказание
        """
        if new_prediction.ndim == 1:
            new_prediction = new_prediction.reshape(1, -1)
        
        # Добавляем в историю
        self.history_buffer.append(new_prediction.copy())
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
        
        if len(self.history_buffer) < self.config.min_samples:
            return new_prediction.reshape(-1)
        
        # Взвешенное сглаживание с учетом confidence
        if confidence is not None:
            weights = np.array([self.config.alpha] ** i for i in range(len(self.history_buffer))])
            weights = weights / weights.sum()
            
            # Учитываем confidence нового предсказания
            weights[-1] *= confidence
            weights = weights / weights.sum()
            
            smoothed = np.average(np.vstack(self.history_buffer), axis=0, weights=weights)
        else:
            # Простое экспоненциальное сглаживание
            smoothed = self.history_buffer[-1].copy()
            for i in range(len(self.history_buffer) - 1):
                alpha = self.config.alpha ** (i + 1)
                smoothed = alpha * self.history_buffer[-(i + 2)] + (1 - alpha) * smoothed
        
        return smoothed.reshape(-1)
    
    def _moving_average_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Скользящее среднее"""
        if data.shape[0] < self.config.window_size:
            return data
        
        smoothed = np.zeros_like(data)
        half_window = self.config.window_size // 2
        
        for i in range(data.shape[0]):
            start = max(0, i - half_window)
            end = min(data.shape[0], i + half_window + 1)
            smoothed[i] = np.mean(data[start:end], axis=0)
        
        return smoothed
    
    def _gaussian_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Гауссово сглаживание"""
        if data.shape[0] < 3:
            return data
        
        # Для 1D сигнала
        if data.ndim == 1 or data.shape[1] == 1:
            return self._gaussian_1d(data.flatten())
        
        # Для 2D+ данных
        smoothed = np.zeros_like(data)
        for i in range(data.shape[1]):
            smoothed[:, i] = self._gaussian_1d(data[:, i])
        
        return smoothed
    
    def _savitgky_golay_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Сглаживание Савицкого-Голая"""
        if data.shape[0] < self.config.poly_order + 1:
            return data
        
        # Для 1D сигнала
        if data.ndim == 1 or data.shape[1] == 1:
            return self._savitgky_golay_1d(data.flatten())
        
        # Для 2D+ данных
        smoothed = np.zeros_like(data)
        for i in range(data.shape[1]):
            smoothed[:, i] = self._savitgky_golay_1d(data[:, i])
        
        return smoothed
    
    def _exponential_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Экспоненциальное сглаживание"""
        if data.shape[0] < 2:
            return data
        
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        
        for i in range(1, data.shape[0]):
            smoothed[i] = self.config.alpha * data[i] + (1 - self.config.alpha) * smoothed[i-1]
        
        return smoothed
    
    def _spatial_smoothing(self, 
                         prediction: np.ndarray, 
                         reference_coords: np.ndarray) -> np.ndarray:
        """Пространственное сглаживание с учетом анатомии"""
        if reference_coords is None or reference_coords.shape[0] < 2:
            return prediction
        
        # Расчет расстояний между точками
        distances = np.linalg.norm(reference_coords[:, np.newaxis, :] - reference_coords[np.newaxis, :, :], axis=2)
        
        # Веса на основе расстояний (ближайшие точки имеют больший вес)
        sigma = self.config.sigma * np.mean(distances[distances > 0])
        weights = np.exp(-distances**2 / (2 * sigma**2))
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Сглаживание как взвешенное среднее соседей
        if prediction.ndim == 1:
            prediction = prediction.reshape(1, -1)
        
        smoothed = np.zeros_like(prediction)
        for i in range(prediction.shape[0]):
            smoothed[i] = np.sum(weights[i] * prediction, axis=0)
        
        return smoothed
    
    def _adaptive_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Адаптивное сглаживание"""
        if data.shape[0] < 3:
            return data
        
        # Расчет локальной изменчивости
        local_var = np.zeros_like(data)
        window = min(3, data.shape[0])
        
        for i in range(data.shape[0]):
            start = max(0, i - window // 2)
            end = min(data.shape[0], i + window // 2 + 1)
            local_var[i] = np.var(data[start:end], axis=0)
        
        # Адаптивная сила сглаживания (меньше для высоких изменчивостей)
        adaptive_alpha = self.config.alpha / (1 + local_var)
        
        # Применение адаптивного сглаживания
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        
        for i in range(1, data.shape[0]):
            if data.ndim == 1:
                smoothed[i] = adaptive_alpha[i] * data[i] + (1 - adaptive_alpha[i]) * smoothed[i-1]
            else:
                for j in range(data.shape[1]):
                    smoothed[i, j] = adaptive_alpha[i, j] * data[i, j] + (1 - adaptive_alpha[i, j]) * smoothed[i-1, j]
        
        return smoothed
    
    def _moving_average_1d(self, signal: np.ndarray) -> np.ndarray:
        """1D скользящее среднее"""
        if len(signal) < self.config.window_size:
            return signal
        
        return np.convolve(signal, np.ones(self.config.window_size)/self.config.window_size, mode='same')
    
    def _gaussian_1d(self, signal: np.ndarray) -> np.ndarray:
        """1D гауссово сглаживание"""
        if len(signal) < 3:
            return signal
        
        return ndimage.gaussian_filter1d(signal, sigma=self.config.sigma)
    
    def _savitgky_golay_1d(self, signal: np.ndarray) -> np.ndarray:
        """1D сглаживание Савицкого-Голая"""
        if len(signal) < self.config.poly_order + 1:
            return signal
        
        window_size = min(self.config.window_size, len(signal))
        if window_size % 2 == 0:
            window_size -= 1
        
        try:
            return savgol_filter(signal, window_size, self.config.poly_order)
        except Exception:
            # Fallback к moving average
            return self._moving_average_1d(signal)
    
    def _exponential_1d(self, signal: np.ndarray) -> np.ndarray:
        """1D экспоненциальное сглаживание"""
        if len(signal) < 2:
            return signal
        
        smoothed = np.zeros_like(signal)
        smoothed[0] = signal[0]
        
        for i in range(1, len(signal)):
            smoothed[i] = self.config.alpha * signal[i] + (1 - self.config.alpha) * smoothed[i-1]
        
        return smoothed
    
    def clear_history(self):
        """Очистить историю сглаживания"""
        self.history_buffer = []
    
    def get_smoothing_statistics(self, 
                             original: np.ndarray, 
                             smoothed: np.ndarray) -> Dict[str, Any]:
        """
        Получить статистику сглаживания
        
        Args:
            original: Оригинальные данные
            smoothed: Сглаженные данные
            
        Returns:
            Статистика сглаживания
        """
        diff = smoothed - original
        
        return {
            'mean_change': np.mean(np.abs(diff)),
            'max_change': np.max(np.abs(diff)),
            'std_change': np.std(diff),
            'reduction_in_variance': np.var(original) - np.var(smoothed),
            'signal_to_noise_ratio_improvement': self._calculate_snr_improvement(original, smoothed)
        }
    
    def _calculate_snr_improvement(self, original: np.ndarray, smoothed: np.ndarray) -> float:
        """Рассчитать улучшение отношения сигнал/шум"""
        def snr(signal):
            return np.mean(signal) / np.std(signal) if np.std(signal) > 0 else 0
        
        return snr(smoothed) - snr(original)


class MultiPointSmoother:
    """
    Сглаживатель для нескольких точек почек с учетом их взаимного расположения
    """
    
    def __init__(self, config: Optional[SmoothingConfig] = None):
        """
        Инициализация
        
        Args:
            config: Конфигурация сглаживания
        """
        self.config = config or SmoothingConfig(method=SmoothingMethod.ADAPTIVE)
        self.point_smoothers = {}
        
    def add_point_smoother(self, point_name: str, smoother: CoordinateSmoother):
        """Добавить сглаживатель для конкретной точки"""
        self.point_smoothers[point_name] = smoother
    
    def smooth_kidney_points(self, 
                           predictions: Dict[str, np.ndarray],
                           anatomical_constraints: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """
        Сгладить координаты точек почки с учетом анатомических ограничений
        
        Args:
            predictions: Словарь с предсказаниями {'upper': array, 'middle': array, 'lower': array}
            anatomical_constraints: Анатомические ограничения
            
        Returns:
            Сглаженные предсказания
        """
        smoothed_predictions = {}
        
        # Сглаживание каждой точки независимо
        for point_name, coords in predictions.items():
            if point_name in self.point_smoothers:
                smoother = self.point_smoothers[point_name]
                smoothed_predictions[point_name] = smoother.smooth_single_prediction(coords)
            else:
                # Создаем временный сглаживатель
                temp_smoother = CoordinateSmoother(self.config)
                smoothed_predictions[point_name] = temp_smoother.smooth_single_prediction(coords)
        
        # Применение анатомических ограничений
        if anatomical_constraints:
            smoothed_predictions = self._apply_anatomical_constraints(smoothed_predictions, anatomical_constraints)
        
        return smoothed_predictions
    
    def _apply_anatomical_constraints(self, 
                                   predictions: Dict[str, np.ndarray],
                                   constraints: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Применить анатомические ограничения к сглаженным предсказаниям"""
        constrained = predictions.copy()
        
        # Ограничение на расстояния между точками
        if 'min_point_distance' in constraints:
            points = ['upper', 'middle', 'lower']
            for i in range(len(points) - 1):
                point1, point2 = points[i], points[i + 1]
                if point1 in constrained and point2 in constrained:
                    dist = np.linalg.norm(constrained[point2] - constrained[point1])
                    min_dist = constraints['min_point_distance']
                    
                    if dist < min_dist:
                        # Корректируем положение второй точки
                        direction = (constrained[point2] - constrained[point1]) / dist
                        constrained[point2] = constrained[point1] + direction * min_dist
        
        # Ограничение на общую длину почки
        if 'max_kidney_length' in constraints:
            if 'upper' in constrained and 'lower' in constrained:
                length = np.linalg.norm(constrained['lower'] - constrained['upper'])
                max_length = constraints['max_kidney_length']
                
                if length > max_length:
                    # Масштабируем почку
                    scale = max_length / length
                    center = (constrained['upper'] + constrained['lower']) / 2
                    constrained['upper'] = center + (constrained['upper'] - center) * scale
                    constrained['lower'] = center + (constrained['lower'] - center) * scale
                    
                    # Корректируем среднюю точку пропорционально
                    if 'middle' in constrained:
                        constrained['middle'] = center + (constrained['middle'] - center) * scale
        
        return constrained


def create_smoother(method: str = "adaptive", **kwargs) -> CoordinateSmoother:
    """
    Создать сглаживатель с указанными параметрами
    
    Args:
        method: Метод сглаживания
        **kwargs: Дополнительные параметры
        
    Returns:
        CoordinateSmoother
    """
    try:
        smoothing_method = SmoothingMethod(method)
    except ValueError:
        smoothing_method = SmoothingMethod.ADAPTIVE
    
    config = SmoothingConfig(method=smoothing_method, **kwargs)
    return CoordinateSmoother(config)


if __name__ == "__main__":
    # Тестирование сглаживания
    print("=== Тестирование Coordinate Smoother ===")
    
    # Генерация тестовых данных
    np.random.seed(42)
    n_points = 20
    
    # Создаем зашумленный сигнал
    true_signal = np.linspace(0, 10, n_points)
    noisy_signal = true_signal + np.random.normal(0, 0.5, n_points)
    
    print("Тестовый сигнал:")
    print(f"Original: {noisy_signal[:5]}")
    
    # Тестирование разных методов
    methods = ["moving_average", "gaussian", "savitgky_golay", "exponential", "adaptive"]
    
    for method in methods:
        smoother = create_smoother(method, window_size=5, sigma=1.0, alpha=0.3)
        smoothed = smoother.smooth_single_prediction(noisy_signal)
        
        stats = smoother.get_smoothing_statistics(noisy_signal, smoothed)
        
        print(f"\n{method.upper()}:")
        print(f"  Smoothed: {smoothed[:5]}")
        print(f"  Mean change: {stats['mean_change']:.3f}")
        print(f"  Variance reduction: {stats['reduction_in_variance']:.3f}")
    
    # Тестирование сглаживания координат почек
    print(f"\n=== Тестирование сглаживания координат почек ===")
    
    kidney_predictions = {
        'upper': np.array([-100, -30, 200]) + np.random.normal(0, 5, 3),
        'middle': np.array([-90, -25, 180]) + np.random.normal(0, 5, 3),
        'lower': np.array([-80, -20, 160]) + np.random.normal(0, 5, 3)
    }
    
    multi_smoother = MultiPointSmoother()
    smoothed_kidneys = multi_smoother.smooth_kidney_points(
        kidney_predictions,
        anatomical_constraints={
            'min_point_distance': 20,  # мм
            'max_kidney_length': 60  # мм
        }
    )
    
    print("Оригинальные координаты:")
    for point, coords in kidney_predictions.items():
        print(f"  {point}: {coords}")
    
    print("\nСглаженные координаты:")
    for point, coords in smoothed_kidneys.items():
        print(f"  {point}: {coords}")
