#!/usr/bin/env python3
"""
Модуль для наложения ограничений на предсказания координат
Жесткие constraints для обеспечения анатомической корректности
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class ConstraintType(Enum):
    """Типы ограничений"""
    HARD = "hard"      # Жесткие ограничения (не могут быть нарушены)
    SOFT = "soft"      # Мягкие ограничения (штраф за нарушение)
    BOUNDS = "bounds"  # Границы значений
    PHYSICS = "physics" # Физические ограничения


@dataclass
class Constraint:
    """Описание ограничения"""
    name: str
    constraint_type: ConstraintType
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    penalty_weight: float = 1.0
    apply_to_axes: Optional[List[str]] = None  # ['x', 'y', 'z'] или None для всех


class AnatomicalConstraints:
    """
    Класс для анатомических ограничений координат почек
    """
    
    def __init__(self):
        """Инициализация ограничений"""
        self.constraints = []
        self.body_bounds = None
        self.kidney_bounds = None
        self._setup_default_constraints()
    
    def _setup_default_constraints(self):
        """Установка ограничений по умолчанию"""
        
        # Максимальное смещение почек при изменении положения
        self.constraints.append(Constraint(
            name="max_shift_magnitude",
            constraint_type=ConstraintType.HARD,
            description="Максимальное смещение почки при переходе supine->lateral",
            max_value=150.0,  # 150 мм максимальное смещение
            apply_to_axes=['x', 'y', 'z']
        ))
        
        # Максимальное смещение по каждой оси
        self.constraints.append(Constraint(
            name="max_shift_x",
            constraint_type=ConstraintType.HARD,
            description="Максимальное смещение по оси X (медиально-латерально)",
            max_value=80.0,  # 80 мм
            apply_to_axes=['x']
        ))
        
        self.constraints.append(Constraint(
            name="max_shift_y",
            constraint_type=ConstraintType.HARD,
            description="Максимальное смещение по оси Y (антеро-постериально)",
            max_value=100.0,  # 100 мм
            apply_to_axes=['y']
        ))
        
        self.constraints.append(Constraint(
            name="max_shift_z",
            constraint_type=ConstraintType.HARD,
            description="Максимальное смещение по оси Z (кранио-каудально)",
            max_value=60.0,  # 60 мм
            apply_to_axes=['z']
        ))
        
        # Ограничения на положение почек относительно тела
        self.constraints.append(Constraint(
            name="kidney_body_bounds",
            constraint_type=ConstraintType.HARD,
            description="Почки должны находиться в пределах тела",
            min_value=-200,  # 200 мм от центра влево
            max_value=200,   # 200 мм от центра вправо
            apply_to_axes=['x', 'y']
        ))
        
        # Ограничения на расстояние между почками
        self.constraints.append(Constraint(
            name="min_kidney_separation",
            constraint_type=ConstraintType.HARD,
            description="Минимальное расстояние между почками",
            min_value=50.0,  # 50 мм минимальное расстояние
            apply_to_axes=None
        ))
        
        # Ограничения на асимметрию смещений
        self.constraints.append(Constraint(
            name="max_asymmetry",
            constraint_type=ConstraintType.SOFT,
            description="Максимальная асимметрия смещений левой и правой почки",
            max_value=40.0,  # 40 мм
            penalty_weight=2.0,
            apply_to_axes=['x', 'y', 'z']
        ))
    
    def set_body_bounds(self, bounds: Dict[str, Tuple[float, float]]):
        """
        Установить границы тела
        
        Args:
            bounds: Словарь с границами {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        """
        self.body_bounds = bounds
    
    def set_kidney_bounds(self, bounds: Dict[str, Tuple[float, float]]):
        """
        Установить физиологические границы почек
        
        Args:
            bounds: Словарь с границами {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        """
        self.kidney_bounds = bounds
    
    def validate_predictions(self, 
                          predictions: np.ndarray,
                          reference_coords: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Валидировать предсказания с учетом ограничений
        
        Args:
            predictions: Предсказанные смещения (N x 3)
            reference_coords: Исходные координаты (N x 3)
            
        Returns:
            Словарь с результатами валидации
        """
        results = {
            'valid': True,
            'violations': [],
            'penalties': {},
            'corrected_predictions': predictions.copy(),
            'total_penalty': 0.0
        }
        
        corrected = predictions.copy()
        
        for constraint in self.constraints:
            violation = self._check_constraint(predictions, constraint, reference_coords)
            
            if violation['violated']:
                results['violations'].append({
                    'constraint': constraint.name,
                    'description': constraint.description,
                    'details': violation['details']
                })
                
                if constraint.constraint_type == ConstraintType.HARD:
                    results['valid'] = False
                    corrected = self._apply_hard_correction(corrected, constraint, violation)
                elif constraint.constraint_type == ConstraintType.SOFT:
                    penalty = violation['penalty'] * constraint.penalty_weight
                    results['penalties'][constraint.name] = penalty
                    results['total_penalty'] += penalty
        
        results['corrected_predictions'] = corrected
        return results
    
    def _check_constraint(self, 
                         predictions: np.ndarray,
                         constraint: Constraint,
                         reference_coords: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Проверить конкретное ограничение"""
        violation = {'violated': False, 'details': {}, 'penalty': 0.0}
        
        if constraint.name == "max_shift_magnitude":
            # Проверка общей величины смещения
            shift_magnitudes = np.linalg.norm(predictions, axis=1)
            max_shift = np.max(shift_magnitudes)
            
            if max_shift > constraint.max_value:
                violation['violated'] = True
                violation['details'] = {
                    'max_shift': max_shift,
                    'allowed_max': constraint.max_value,
                    'excess': max_shift - constraint.max_value
                }
                violation['penalty'] = max_shift - constraint.max_value
        
        elif constraint.name.startswith("max_shift_"):
            # Проверка смещения по конкретной оси
            axis = constraint.name.split("_")[-1]
            axis_idx = ['x', 'y', 'z'].index(axis)
            
            if constraint.apply_to_axes and axis in constraint.apply_to_axes:
                axis_shifts = np.abs(predictions[:, axis_idx])
                max_axis_shift = np.max(axis_shifts)
                
                if max_axis_shift > constraint.max_value:
                    violation['violated'] = True
                    violation['details'] = {
                        'axis': axis,
                        'max_shift': max_axis_shift,
                        'allowed_max': constraint.max_value,
                        'excess': max_axis_shift - constraint.max_value
                    }
                    violation['penalty'] = max_axis_shift - constraint.max_value
        
        elif constraint.name == "kidney_body_bounds":
            # Проверка границ тела
            if self.body_bounds is not None:
                for i, axis in enumerate(['x', 'y', 'z']):
                    if constraint.apply_to_axes and axis in constraint.apply_to_axes:
                        min_bound, max_bound = self.body_bounds[axis]
                        axis_coords = predictions[:, i]
                        
                        out_of_bounds = (axis_coords < min_bound) | (axis_coords > max_bound)
                        if out_of_bounds.any():
                            violation['violated'] = True
                            violation['details'][axis] = {
                                'out_of_bounds_count': out_of_bounds.sum(),
                                'bounds': (min_bound, max_bound),
                                'min_value': np.min(axis_coords),
                                'max_value': np.max(axis_coords)
                            }
        
        elif constraint.name == "min_kidney_separation":
            # Проверка минимального расстояния между почками
            if reference_coords is not None and len(predictions) >= 2:
                # Предполагаем, что первые две точки - левая и правая почки
                left_kidney = reference_coords[0] + predictions[0]
                right_kidney = reference_coords[1] + predictions[1]
                
                separation = np.linalg.norm(left_kidney - right_kidney)
                
                if separation < constraint.min_value:
                    violation['violated'] = True
                    violation['details'] = {
                        'separation': separation,
                        'min_allowed': constraint.min_value,
                        'deficit': constraint.min_value - separation
                    }
                    violation['penalty'] = constraint.min_value - separation
        
        elif constraint.name == "max_asymmetry":
            # Проверка асимметрии смещений
            if len(predictions) >= 2:
                asymmetry = np.abs(predictions[0] - predictions[1])
                max_asymmetry = np.max(asymmetry)
                
                if max_asymmetry > constraint.max_value:
                    violation['violated'] = True
                    violation['details'] = {
                        'max_asymmetry': max_asymmetry,
                        'allowed_max': constraint.max_value,
                        'excess': max_asymmetry - constraint.max_value,
                        'asymmetry_by_axis': asymmetry.tolist()
                    }
                    violation['penalty'] = max_asymmetry - constraint.max_value
        
        return violation
    
    def _apply_hard_correction(self, 
                             predictions: np.ndarray,
                             constraint: Constraint,
                             violation: Dict[str, Any]) -> np.ndarray:
        """Применить жесткую коррекцию для нарушения"""
        corrected = predictions.copy()
        
        if constraint.name == "max_shift_magnitude":
            # Ограничение общей величины смещения
            shift_magnitudes = np.linalg.norm(corrected, axis=1)
            max_allowed = constraint.max_value
            
            for i in range(len(corrected)):
                if shift_magnitudes[i] > max_allowed:
                    scale = max_allowed / shift_magnitudes[i]
                    corrected[i] *= scale
        
        elif constraint.name.startswith("max_shift_"):
            # Ограничение смещения по оси
            axis = constraint.name.split("_")[-1]
            axis_idx = ['x', 'y', 'z'].index(axis)
            
            if constraint.apply_to_axes and axis in constraint.apply_to_axes:
                max_allowed = constraint.max_value
                corrected[:, axis_idx] = np.clip(corrected[:, axis_idx], -max_allowed, max_allowed)
        
        elif constraint.name == "kidney_body_bounds":
            # Ограничение границами тела
            if self.body_bounds is not None:
                for i, axis in enumerate(['x', 'y', 'z']):
                    if constraint.apply_to_axes and axis in constraint.apply_to_axes:
                        min_bound, max_bound = self.body_bounds[axis]
                        corrected[:, i] = np.clip(corrected[:, i], min_bound, max_bound)
        
        return corrected
    
    def apply_constraints_to_predictions(self,
                                     predictions: np.ndarray,
                                     reference_coords: Optional[np.ndarray] = None,
                                     mode: str = "correct") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Применить ограничения к предсказаниям
        
        Args:
            predictions: Предсказанные смещения
            reference_coords: Исходные координаты
            mode: Режим - 'correct' (исправить) или 'validate' (только проверить)
            
        Returns:
            (исправленные предсказания, результаты валидации)
        """
        validation = self.validate_predictions(predictions, reference_coords)
        
        if mode == "correct":
            return validation['corrected_predictions'], validation
        else:
            return predictions, validation
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Получить сводку всех ограничений"""
        return {
            'total_constraints': len(self.constraints),
            'hard_constraints': len([c for c in self.constraints if c.constraint_type == ConstraintType.HARD]),
            'soft_constraints': len([c for c in self.constraints if c.constraint_type == ConstraintType.SOFT]),
            'constraints': [
                {
                    'name': c.name,
                    'type': c.constraint_type.value,
                    'description': c.description,
                    'min_value': c.min_value,
                    'max_value': c.max_value,
                    'penalty_weight': c.penalty_weight,
                    'apply_to_axes': c.apply_to_axes
                }
                for c in self.constraints
            ]
        }


class ConstraintOptimizer:
    """
    Оптимизатор для коррекции предсказаний с учетом ограничений
    """
    
    def __init__(self, constraints: AnatomicalConstraints):
        """
        Инициализация оптимизатора
        
        Args:
            constraints: Объект ограничений
        """
        self.constraints = constraints
    
    def optimize_predictions(self,
                         original_predictions: np.ndarray,
                         reference_coords: np.ndarray,
                         learning_rate: float = 0.1,
                         max_iterations: int = 100) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Оптимизировать предсказания с учетом ограничений
        
        Args:
            original_predictions: Исходные предсказания
            reference_coords: Референтные координаты
            learning_rate: Скорость обучения
            max_iterations: Максимальное количество итераций
            
        Returns:
            (оптимизированные предсказания, статистика оптимизации)
        """
        current_predictions = original_predictions.copy()
        stats = {
            'iterations': 0,
            'initial_penalty': 0,
            'final_penalty': 0,
            'penalty_reduction': 0,
            'converged': False
        }
        
        # Начальная оценка
        _, validation = self.constraints.validate_predictions(current_predictions, reference_coords)
        stats['initial_penalty'] = validation['total_penalty']
        
        for iteration in range(max_iterations):
            # Текущая оценка
            corrected, validation = self.constraints.apply_constraints_to_predictions(
                current_predictions, reference_coords, mode="correct"
            )
            
            current_penalty = validation['total_penalty']
            
            # Проверка сходимости
            if current_penalty < 1e-6:
                stats['converged'] = True
                break
            
            # Градиентный шаг к исправленным значениям
            gradient = corrected - current_predictions
            current_predictions += learning_rate * gradient
            
            stats['iterations'] = iteration + 1
        
        # Финальная оценка
        _, final_validation = self.constraints.validate_predictions(current_predictions, reference_coords)
        stats['final_penalty'] = final_validation['total_penalty']
        stats['penalty_reduction'] = stats['initial_penalty'] - stats['final_penalty']
        
        return current_predictions, stats


def create_default_constraints() -> AnatomicalConstraints:
    """
    Создать ограничения по умолчанию для почек
    
    Returns:
        AnatomicalConstraints с настройками по умолчанию
    """
    constraints = AnatomicalConstraints()
    
    # Стандартные границы тела (в мм от центра)
    body_bounds = {
        'x': (-200, 200),   # Ширина 400 мм
        'y': (-150, 150),   # Глубина 300 мм
        'z': (-100, 300)    # Высота 400 мм (смещенная вверх)
    }
    
    # Физиологические границы почек
    kidney_bounds = {
        'x': (-150, 150),   # Почки не дальше 150 мм от центра
        'y': (-100, 50),    # Спереди ограничены
        'z': (50, 250)      # В верхней половине тела
    }
    
    constraints.set_body_bounds(body_bounds)
    constraints.set_kidney_bounds(kidney_bounds)
    
    return constraints


if __name__ == "__main__":
    # Тестирование ограничений
    print("=== Тестирование анатомических ограничений ===")
    
    # Создание ограничений
    constraints = create_default_constraints()
    
    # Тестовые предсказания (с нарушениями)
    test_predictions = np.array([
        [200, 150, 100],   # Слишком большое смещение по X и Y
        [-50, -120, 80],   # Слишком большое смещение по Y
        [30, 40, 20]       # Нормальное смещение
    ])
    
    print("Тестовые предсказания:")
    print(test_predictions)
    
    # Валидация
    corrected, validation = constraints.apply_constraints_to_predictions(
        test_predictions, mode="correct"
    )
    
    print(f"\nРезультаты валидации:")
    print(f"Валидны: {validation['valid']}")
    print(f"Нарушения: {len(validation['violations'])}")
    for violation in validation['violations']:
        print(f"  - {violation['constraint']}: {violation['description']}")
    print(f"Общий штраф: {validation['total_penalty']:.2f}")
    
    print(f"\nИсправленные предсказания:")
    print(corrected)
    
    # Оптимизация
    print("\n=== Оптимизация предсказаний ===")
    optimizer = ConstraintOptimizer(constraints)
    
    reference_coords = np.array([
        [-100, -30, 200],
        [100, -25, 190],
        [0, -20, 180]
    ])
    
    optimized, stats = optimizer.optimize_predictions(
        test_predictions, reference_coords
    )
    
    print(f"Статистика оптимизации:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nОптимизированные предсказания:")
    print(optimized)
