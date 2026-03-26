#!/usr/bin/env python3
"""
Координатная система для AR Laparoscopy проекта
Стандартизация согласно медицинским конвенциям DICOM/LPS
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from enum import Enum


class CoordinateSystem(Enum):
    """Типы координатных систем"""
    LPS = "LPS"  # Left, Posterior, Superior - DICOM стандарт
    RAS = "RAS"  # Right, Anterior, Superior - 3D Slicer стандарт
    PATIENT = "PATIENT"  # Пациент-центричная система
    ANATOMICAL = "ANATOMICAL"  # Анатомическая система


@dataclass
class CoordinateSystemDefinition:
    """Определение координатной системы"""
    name: str
    description: str
    axis_directions: Dict[str, str]  # {'x': 'left->right', 'y': 'posterior->anterior', 'z': 'inferior->superior'}
    origin_description: str
    units: str = "mm"


class MedicalCoordinateSystem:
    """
    Класс для работы с медицинскими координатными системами
    """
    
    # Стандартные определения систем
    SYSTEMS = {
        CoordinateSystem.LPS: CoordinateSystemDefinition(
            name="LPS (DICOM)",
            description="Left, Posterior, Superior - DICOM стандарт",
            axis_directions={
                'x': 'from right towards left',
                'y': 'from anterior towards posterior', 
                'z': 'from inferior towards superior'
            },
            origin_description="Произвольная точка, обычно центр изображения или геометрический центр сканера",
            units="mm"
        ),
        CoordinateSystem.RAS: CoordinateSystemDefinition(
            name="RAS (3D Slicer)",
            description="Right, Anterior, Superior - 3D Slicer стандарт",
            axis_directions={
                'x': 'from left towards right',
                'y': 'from posterior towards anterior',
                'z': 'from inferior towards superior'
            },
            origin_description="Произвольная точка, обычно центр изображения",
            units="mm"
        ),
        CoordinateSystem.PATIENT: CoordinateSystemDefinition(
            name="Patient-Centric",
            description="Пациент-центричная система с началом в центре тела",
            axis_directions={
                'x': 'from midline towards left (медиально-латерально)',
                'y': 'from posterior towards anterior (антеро-постериорно)',
                'z': 'from inferior towards superior (кранио-каудально)'
            },
            origin_description="Центр тела пациента (примерно пупок)",
            units="mm"
        )
    }
    
    def __init__(self, current_system: CoordinateSystem = CoordinateSystem.LPS):
        """
        Инициализация координатной системы
        
        Args:
            current_system: Текущая система координат
        """
        self.current_system = current_system
        self.system_def = self.SYSTEMS[current_system]
        
    def get_system_info(self) -> CoordinateSystemDefinition:
        """Получить информацию о текущей системе"""
        return self.system_def
    
    def transform_to_system(self, 
                          coordinates: np.ndarray,
                          target_system: CoordinateSystem,
                          transformation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Преобразование координат в другую систему
        
        Args:
            coordinates: Координаты в текущей системе (N x 3)
            target_system: Целевая система координат
            transformation_matrix: Матрица преобразования (4x4)
            
        Returns:
            Координаты в целевой системе
        """
        if self.current_system == target_system:
            return coordinates.copy()
            
        # Стандартные преобразования между LPS и RAS
        if self.current_system == CoordinateSystem.LPS and target_system == CoordinateSystem.RAS:
            # LPS -> RAS: инвертировать X и Y оси
            transformed = coordinates.copy()
            transformed[:, 0] *= -1  # X: Left->Right
            transformed[:, 1] *= -1  # Y: Posterior->Anterior
            return transformed
            
        elif self.current_system == CoordinateSystem.RAS and target_system == CoordinateSystem.LPS:
            # RAS -> LPS: инвертировать X и Y оси
            transformed = coordinates.copy()
            transformed[:, 0] *= -1  # X: Right->Left
            transformed[:, 1] *= -1  # Y: Anterior->Posterior
            return transformed
            
        # Если есть кастомная матрица преобразования
        elif transformation_matrix is not None:
            # Преобразование в однородные координаты
            homogeneous = np.column_stack([coordinates, np.ones(len(coordinates))])
            transformed_h = (transformation_matrix @ homogeneous.T).T
            return transformed_h[:, :3]
            
        else:
            raise ValueError(f"Неизвестное преобразование: {self.current_system} -> {target_system}")
    
    def create_transformation_matrix(self,
                                 origin_source: np.ndarray,
                                 origin_target: np.ndarray,
                                 rotation_angles: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Создать матрицу преобразования 4x4
        
        Args:
            origin_source: Начало координат в исходной системе
            origin_target: Начало координат в целевой системе
            rotation_angles: Углы поворота (rx, ry, rz) в градусах
            
        Returns:
            Матрица преобразования 4x4
        """
        matrix = np.eye(4)
        
        # Трансляция
        translation = origin_target - origin_source
        matrix[:3, 3] = translation
        
        # Вращение (если задано)
        if rotation_angles:
            rx, ry, rz = np.radians(rotation_angles)
            
            # Матрицы вращения
            Rx = np.array([[1, 0, 0],
                         [0, np.cos(rx), -np.sin(rx)],
                         [0, np.sin(rx), np.cos(rx)]])
            
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                         [0, 1, 0],
                         [-np.sin(ry), 0, np.cos(ry)]])
            
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                         [np.sin(rz), np.cos(rz), 0],
                         [0, 0, 1]])
            
            # Комбинированное вращение
            R = Rz @ Ry @ Rx
            matrix[:3, :3] = R
            
        return matrix
    
    def validate_coordinates(self, coordinates: np.ndarray) -> Dict[str, Any]:
        """
        Валидация координат в текущей системе
        
        Args:
            coordinates: Координаты для валидации (N x 3)
            
        Returns:
            Словарь с результатами валидации
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Проверка размерности
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            results['valid'] = False
            results['errors'].append(f"Неверная размерность: {coordinates.shape}, ожидается (N, 3)")
            return results
        
        # Проверка на NaN/Inf
        nan_mask = np.isnan(coordinates)
        inf_mask = np.isinf(coordinates)
        
        if nan_mask.any():
            results['warnings'].append(f"Обнаружены NaN значения: {nan_mask.sum()} шт.")
            
        if inf_mask.any():
            results['warnings'].append(f"Обнаружены Inf значения: {inf_mask.sum()} шт.")
        
        # Статистика
        results['statistics'] = {
            'num_points': len(coordinates),
            'x_range': (np.nanmin(coordinates[:, 0]), np.nanmax(coordinates[:, 0])),
            'y_range': (np.nanmin(coordinates[:, 1]), np.nanmax(coordinates[:, 1])),
            'z_range': (np.nanmin(coordinates[:, 2]), np.nanmax(coordinates[:, 2])),
            'mean_point': np.nanmean(coordinates, axis=0),
            'std_point': np.nanstd(coordinates, axis=0)
        }
        
        # Проверка на аномальные значения для LPS системы
        if self.current_system == CoordinateSystem.LPS:
            # В LPS: X должен быть отрицательным для левой стороны
            # Y должен быть отрицательным для передней части
            # Z должен быть положительным для верхней части
            
            x_positive = coordinates[:, 0] > 0
            y_positive = coordinates[:, 1] > 0
            z_negative = coordinates[:, 2] < 0
            
            if x_positive.any():
                results['warnings'].append(f"Точки с положительным X ({x_positive.sum()} шт.) - могут быть справа")
                
            if y_positive.any():
                results['warnings'].append(f"Точки с положительным Y ({y_positive.sum()} шт.) - могут быть сзади")
                
            if z_negative.any():
                results['warnings'].append(f"Точки с отрицательным Z ({z_negative.sum()} шт.) - могут быть снизу")
        
        return results
    
    def get_anatomical_directions(self) -> Dict[str, str]:
        """Получить направления осей для текущей системы"""
        return self.system_def.axis_directions
    
    def print_system_info(self):
        """Вывести информацию о текущей системе координат"""
        print(f"=== {self.system_def.name} ===")
        print(f"Описание: {self.system_def.description}")
        print(f"Направления осей:")
        for axis, direction in self.system_def.axis_directions.items():
            print(f"  {axis.upper()}: {direction}")
        print(f"Начало координат: {self.system_def.origin_description}")
        print(f"Единицы: {self.system_def.units}")
        print()


def create_standard_coordinate_system() -> MedicalCoordinateSystem:
    """
    Создать стандартную координатную систему для проекта
    
    Returns:
        MedicalCoordinateSystem с LPS системой (DICOM стандарт)
    """
    return MedicalCoordinateSystem(CoordinateSystem.LPS)


def validate_and_fix_coordinates(coordinates: np.ndarray,
                             system: MedicalCoordinateSystem) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Валидировать и исправить координаты
    
    Args:
        coordinates: Входные координаты
        system: Система координат
        
    Returns:
        (исправленные координаты, результаты валидации)
    """
    validation = system.validate_coordinates(coordinates)
    
    # Исправление NaN/Inf
    fixed_coords = coordinates.copy()
    nan_mask = np.isnan(fixed_coords)
    inf_mask = np.isinf(fixed_coords)
    
    # Замена NaN на медианные значения
    if nan_mask.any():
        median_vals = np.nanmedian(fixed_coords, axis=0)
        for i in range(3):
            fixed_coords[nan_mask[:, i], i] = median_vals[i]
        validation['warnings'].append("NaN значения заменены на медианные")
    
    # Замена Inf на большие значения
    if inf_mask.any():
        fixed_coords[inf_mask] = 1e6  # Большое значение
        validation['warnings'].append("Inf значения заменены на 1e6")
    
    return fixed_coords, validation


if __name__ == "__main__":
    # Тестирование координатной системы
    print("=== Тестирование координатной системы ===")
    
    # Создание системы
    lps_system = MedicalCoordinateSystem(CoordinateSystem.LPS)
    lps_system.print_system_info()
    
    # Тестовые координаты (почка в LPS системе)
    test_coords = np.array([
        [-50, -30, 100],  # Левая, передняя, верхняя
        [-60, -25, 95],   # Левее, переднее, ниже
        [-45, -35, 105]   # Правее, заднее, выше
    ])
    
    print("\n=== Валидация тестовых координат ===")
    validation = lps_system.validate_coordinates(test_coords)
    print(f"Валидны: {validation['valid']}")
    print(f"Предупреждения: {validation['warnings']}")
    print(f"Ошибки: {validation['errors']}")
    print(f"Статистика: {validation['statistics']}")
    
    # Преобразование в RAS
    print("\n=== Преобразование LPS -> RAS ===")
    ras_coords = lps_system.transform_to_system(test_coords, CoordinateSystem.RAS)
    print("LPS координаты:")
    print(test_coords)
    print("RAS координаты:")
    print(ras_coords)
    
    # Преобразование обратно
    print("\n=== Преобразование RAS -> LPS ===")
    ras_system = MedicalCoordinateSystem(CoordinateSystem.RAS)
    back_to_lps = ras_system.transform_to_system(ras_coords, CoordinateSystem.LPS)
    print("Обратно в LPS:")
    print(back_to_lps)
    print(f"Разница с оригиналом: {np.max(np.abs(back_to_lps - test_coords)):.6f}")
