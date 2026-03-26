#!/usr/bin/env python3
"""
Интегрированный модуль улучшений для AR Laparoscopy ML Pipeline
Объединяет координатные системы, относительные координаты, ограничения и confidence scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

from .coordinate_system import MedicalCoordinateSystem, CoordinateSystem, validate_and_fix_coordinates
from .relative_coordinates import RelativeCoordinateConverter, create_relative_converter_from_dicom_features
from .constraints import AnatomicalConstraints, create_default_constraints, ConstraintOptimizer
from .confidence_scoring import ConfidenceEstimator, ConfidenceScore


@dataclass
class EnhancedPrediction:
    """Расширенный результат предсказания с дополнительной информацией"""
    predictions: np.ndarray
    confidence_scores: List[ConfidenceScore]
    relative_coordinates: Dict[str, np.ndarray]
    constraint_results: Dict[str, Any]
    coordinate_system_info: Dict[str, Any]
    metadata: Dict[str, Any]


class EnhancedKidneyPredictor:
    """
    Улучшенный предиктор с интеграцией всех модулей
    """
    
    def __init__(self,
                 coordinate_system: CoordinateSystem = CoordinateSystem.LPS,
                 use_relative_coordinates: bool = True,
                 apply_constraints: bool = True,
                 calculate_confidence: bool = True):
        """
        Инициализация улучшенного предиктора
        
        Args:
            coordinate_system: Система координат
            use_relative_coordinates: Использовать относительные координаты
            apply_constraints: Применять ограничения
            calculate_confidence: Рассчитывать уверенность
        """
        self.coordinate_system = MedicalCoordinateSystem(coordinate_system)
        self.use_relative_coordinates = use_relative_coordinates
        self.apply_constraints = apply_constraints
        self.calculate_confidence = calculate_confidence
        
        # Инициализация компонентов
        self.relative_converter = None
        self.constraints = create_default_constraints() if apply_constraints else None
        self.constraint_optimizer = None
        self.confidence_estimator = None
        
        # Обучающие данные для confidence scoring
        self.train_features = None
        self.train_targets = None
        
        # Метаданные
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False
    
    def fit(self,
            train_features: np.ndarray,
            train_targets: np.ndarray,
            feature_names: Optional[List[str]] = None,
            target_names: Optional[List[str]] = None,
            dicom_metadata: Optional[Dict[str, Any]] = None):
        """
        Обучить предиктор
        
        Args:
            train_features: Признаки тренировочных данных
            train_targets: Целевые значения
            feature_names: Имена признаков
            target_names: Имена целевых переменных
            dicom_metadata: Метаданные DICOM для настройки референтных точек
        """
        self.train_features = train_features
        self.train_targets = train_targets
        self.feature_names = feature_names or [f"feature_{i}" for i in range(train_features.shape[1])]
        self.target_names = target_names or [f"target_{i}" for i in range(train_targets.shape[1])]
        
        # Валидация и исправление координат
        if train_targets.shape[1] >= 3:  # Если есть пространственные координаты
            fixed_targets, validation_results = validate_and_fix_coordinates(
                train_targets, self.coordinate_system
            )
            if not validation_results['valid']:
                warnings.warn(f"Обнаружены проблемы с координатами: {validation_results['warnings']}")
            train_targets = fixed_targets
        
        # Настройка конвертера относительных координат
        if self.use_relative_coordinates and dicom_metadata:
            self.relative_converter = create_relative_converter_from_dicom_features(dicom_metadata)
        
        # Настройка confidence estimator
        if self.calculate_confidence:
            self.confidence_estimator = ConfidenceEstimator(method="ensemble")
            self.confidence_estimator.fit(train_features, train_targets)
        
        # Настройка constraint optimizer
        if self.apply_constraints and self.constraints:
            self.constraint_optimizer = ConstraintOptimizer(self.constraints)
        
        self.is_fitted = True
    
    def predict(self,
                test_features: np.ndarray,
                raw_predictions: Optional[np.ndarray] = None,
                ensemble_predictions: Optional[Dict[str, np.ndarray]] = None,
                reference_coordinates: Optional[np.ndarray] = None,
                dicom_metadata: Optional[Dict[str, Any]] = None) -> EnhancedPrediction:
        """
        Сделать предсказание с улучшениями
        
        Args:
            test_features: Признаки тестовых данных
            raw_predictions: "Сырые" предсказания модели
            ensemble_predictions: Предсказания ансамбля
            reference_coordinates: Референтные координаты
            dicom_metadata: Метаданные DICOM
            
        Returns:
            EnhancedPrediction с полной информацией
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit()")
        
        if raw_predictions is None:
            raise ValueError("Требуются raw_predictions")
        
        # Валидация входных данных
        validated_predictions, coord_validation = validate_and_fix_coordinates(
            raw_predictions, self.coordinate_system
        )
        
        # Применение ограничений
        constraint_results = {'valid': True, 'violations': [], 'penalty': 0}
        if self.apply_constraints and self.constraints:
            validated_predictions, constraint_results = self.constraints.apply_constraints_to_predictions(
                validated_predictions, reference_coordinates, mode="correct"
            )
        
        # Расчет относительных координат
        relative_coordinates = {}
        if self.use_relative_coordinates:
            if self.relative_converter is None and dicom_metadata is not None:
                self.relative_converter = create_relative_converter_from_dicom_features(dicom_metadata)
            
            if self.relative_converter is not None:
                # Предполагаем, что предсказания - это смещения
                if reference_coordinates is not None:
                    final_coords = reference_coordinates + validated_predictions
                    relative = self.relative_converter.calculate_relative_coordinates(final_coords)
                    relative_coordinates = {
                        'relative_to_spine': relative.relative_to_spine,
                        'relative_to_body_center': relative.relative_to_body_center,
                        'normalized': relative.normalized_by_body_size,
                        'distances': relative.distances
                    }
        
        # Расчет confidence scores
        confidence_scores = []
        if self.calculate_confidence and self.confidence_estimator is not None:
            confidence_scores = self.confidence_estimator.predict_confidence(
                test_features, validated_predictions, ensemble_predictions, constraint_results
            )
        
        # Информация о координатной системе
        coordinate_system_info = {
            'system': self.coordinate_system.current_system.value,
            'description': self.coordinate_system.system_def.description,
            'axis_directions': self.coordinate_system.get_anatomical_directions(),
            'validation': coord_validation
        }
        
        # Метаданные
        metadata = {
            'n_samples': len(test_features),
            'n_features': test_features.shape[1],
            'n_targets': validated_predictions.shape[1] if validated_predictions.ndim > 1 else 1,
            'features_used': self.use_relative_coordinates,
            'constraints_applied': self.apply_constraints,
            'confidence_calculated': self.calculate_confidence,
            'method': self.confidence_estimator.method if self.confidence_estimator else None
        }
        
        return EnhancedPrediction(
            predictions=validated_predictions,
            confidence_scores=confidence_scores,
            relative_coordinates=relative_coordinates,
            constraint_results=constraint_results,
            coordinate_system_info=coordinate_system_info,
            metadata=metadata
        )
    
    def predict_with_optimization(self,
                                test_features: np.ndarray,
                                raw_predictions: np.ndarray,
                                reference_coordinates: np.ndarray,
                                dicom_metadata: Optional[Dict[str, Any]] = None,
                                max_iterations: int = 100) -> EnhancedPrediction:
        """
        Предсказание с оптимизацией ограничений
        
        Args:
            test_features: Признаки тестовых данных
            raw_predictions: "Сырые" предсказания модели
            reference_coordinates: Референтные координаты
            dicom_metadata: Метаданные DICOM
            max_iterations: Максимальное количество итераций оптимизации
            
        Returns:
            EnhancedPrediction с оптимизированными предсказаниями
        """
        if not self.constraint_optimizer:
            raise ValueError("Constraint optimizer не инициализирован")
        
        # Оптимизация предсказаний
        optimized_predictions, optimization_stats = self.constraint_optimizer.optimize_predictions(
            raw_predictions, reference_coordinates, max_iterations=max_iterations
        )
        
        # Создание EnhancedPrediction с оптимизированными результатами
        result = self.predict(
            test_features=test_features,
            raw_predictions=optimized_predictions,
            reference_coordinates=reference_coordinates,
            dicom_metadata=dicom_metadata
        )
        
        # Добавление статистики оптимизации
        result.metadata['optimization'] = optimization_stats
        
        return result
    
    def evaluate_with_confidence(self,
                               test_features: np.ndarray,
                               test_targets: np.ndarray,
                               predictions: np.ndarray,
                               confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Оценить модель с учетом confidence scores
        
        Args:
            test_features: Признаки тестовых данных
            test_targets: Истинные значения
            predictions: Предсказания модели
            confidence_threshold: Порог уверенности
            
        Returns:
            Результаты оценки с confidence
        """
        # Расчет confidence scores
        confidence_scores = self.confidence_estimator.predict_confidence(
            test_features, predictions
        )
        
        # Фильтрация по уверенности
        high_confidence_mask = np.array([
            cs.overall_confidence >= confidence_threshold for cs in confidence_scores
        ])
        
        # Метрики на всех данных
        all_errors = np.abs(predictions - test_targets)
        all_mae = np.mean(all_errors)
        
        # Метрики на высокоуверенных предсказаниях
        if high_confidence_mask.any():
            high_conf_errors = all_errors[high_confidence_mask]
            high_conf_mae = np.mean(high_conf_errors)
            high_conf_count = np.sum(high_confidence_mask)
        else:
            high_conf_mae = float('nan')
            high_conf_count = 0
        
        # Статистика confidence
        confidences = [cs.overall_confidence for cs in confidence_scores]
        
        results = {
            'overall_mae': all_mae,
            'high_confidence_mae': high_conf_mae,
            'high_confidence_count': high_conf_count,
            'high_confidence_ratio': high_conf_count / len(confidence_scores),
            'mean_confidence': np.mean(confidences),
            'confidence_threshold': confidence_threshold,
            'n_samples': len(confidence_scores)
        }
        
        return results
    
    def get_feature_importance_with_confidence(self) -> Dict[str, float]:
        """
        Получить важность признаков с учетом confidence
        
        Returns:
            Словарь с важностью признаков
        """
        if not self.feature_names or not self.confidence_estimator:
            return {}
        
        if self.confidence_estimator.feature_importance is not None:
            importance_dict = dict(zip(self.feature_names, self.confidence_estimator.feature_importance))
            return importance_dict
        
        return {}
    
    def export_configuration(self) -> Dict[str, Any]:
        """
        Экспортировать конфигурацию предиктора
        
        Returns:
            Конфигурация в виде словаря
        """
        config = {
            'coordinate_system': self.coordinate_system.current_system.value,
            'use_relative_coordinates': self.use_relative_coordinates,
            'apply_constraints': self.apply_constraints,
            'calculate_confidence': self.calculate_confidence,
            'constraints': self.constraints.get_constraint_summary() if self.constraints else None,
            'confidence_method': self.confidence_estimator.method if self.confidence_estimator else None,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'is_fitted': self.is_fitted
        }
        
        return config


def create_enhanced_predictor_from_config(config: Dict[str, Any]) -> EnhancedKidneyPredictor:
    """
    Создать улучшенный предиктор из конфигурации
    
    Args:
        config: Конфигурация
        
    Returns:
        EnhancedKidneyPredictor
    """
    coordinate_system = CoordinateSystem(config.get('coordinate_system', 'LPS'))
    
    predictor = EnhancedKidneyPredictor(
        coordinate_system=coordinate_system,
        use_relative_coordinates=config.get('use_relative_coordinates', True),
        apply_constraints=config.get('apply_constraints', True),
        calculate_confidence=config.get('calculate_confidence', True)
    )
    
    return predictor


if __name__ == "__main__":
    # Тестирование интегрированного предиктора
    print("=== Тестирование Enhanced Kidney Predictor ===")
    
    # Генерация тестовых данных
    np.random.seed(42)
    n_train = 50
    n_test = 10
    n_features = 15
    n_targets = 9  # 3 точки * 3 координаты
    
    train_features = np.random.randn(n_train, n_features)
    train_targets = np.random.randn(n_train, n_targets) * 10  # Смещения в мм
    test_features = np.random.randn(n_test, n_features)
    raw_predictions = np.random.randn(n_test, n_targets) * 10
    reference_coords = np.random.randn(n_test, n_targets) * 100
    
    # DICOM метаданные для теста
    dicom_metadata = {
        'body_com_x_mm': 0, 'body_com_y_mm': 0, 'body_com_z_mm': 0,
        'spine_center_x_mm': 0, 'spine_center_y_mm': -50, 'spine_center_z_mm': 0,
        'body_width_mm_median': 400, 'body_depth_mm_median': 300
    }
    
    # Создание и обучение предиктора
    predictor = EnhancedKidneyPredictor(
        coordinate_system=CoordinateSystem.LPS,
        use_relative_coordinates=True,
        apply_constraints=True,
        calculate_confidence=True
    )
    
    predictor.fit(
        train_features=train_features,
        train_targets=train_targets,
        feature_names=[f"feature_{i}" for i in range(n_features)],
        target_names=[f"delta_{axis}_{point}" for point in ['upper', 'middle', 'lower'] for axis in ['X', 'Y', 'Z']],
        dicom_metadata=dicom_metadata
    )
    
    # Предсказание
    ensemble_preds = {
        'rf': raw_predictions + np.random.randn(*raw_predictions.shape) * 2,
        'xgb': raw_predictions + np.random.randn(*raw_predictions.shape) * 2
    }
    
    enhanced_result = predictor.predict(
        test_features=test_features,
        raw_predictions=raw_predictions,
        ensemble_predictions=ensemble_preds,
        reference_coordinates=reference_coords,
        dicom_metadata=dicom_metadata
    )
    
    print("Результаты предсказания:")
    print(f"Форма предсказаний: {enhanced_result.predictions.shape}")
    print(f"Количество confidence scores: {len(enhanced_result.confidence_scores)}")
    
    if enhanced_result.confidence_scores:
        print(f"Средняя уверенность: {np.mean([cs.overall_confidence for cs in enhanced_result.confidence_scores]):.3f}")
    
    print(f"Нарушений ограничений: {len(enhanced_result.constraint_results.get('violations', []))}")
    print(f"Система координат: {enhanced_result.coordinate_system_info['system']}")
    
    if enhanced_result.relative_coordinates:
        print("Относительные координаты рассчитаны")
    
    # Оценка с confidence
    test_targets = np.random.randn(n_test, n_targets) * 10
    evaluation = predictor.evaluate_with_confidence(
        test_features, test_targets, enhanced_result.predictions
    )
    
    print(f"\nОценка с confidence:")
    for key, value in evaluation.items():
        print(f"  {key}: {value}")
    
    # Экспорт конфигурации
    config = predictor.export_configuration()
    print(f"\nКонфигурация предиктора:")
    print(f"  Система координат: {config['coordinate_system']}")
    print(f"  Относительные координаты: {config['use_relative_coordinates']}")
    print(f"  Ограничения: {config['apply_constraints']}")
    print(f"  Confidence: {config['calculate_confidence']}")
