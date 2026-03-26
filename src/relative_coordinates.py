#!/usr/bin/env python3
"""
Модуль для работы с относительными координатами
Перевод абсолютных координат в относительные относительно позвоночника и центра тела
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
from .coordinate_system import MedicalCoordinateSystem, CoordinateSystem


@dataclass
class RelativeCoordinates:
    """Относительные координаты точки"""
    relative_to_spine: np.ndarray  # Относительно позвоночника
    relative_to_body_center: np.ndarray  # Относительно центра тела
    normalized_by_body_size: np.ndarray  # Нормализованные на размер тела
    distances: Dict[str, float]  # Расстояния до референтных точек


class RelativeCoordinateConverter:
    """
    Класс для преобразования абсолютных координат в относительные
    """
    
    def __init__(self, coordinate_system: MedicalCoordinateSystem):
        """
        Инициализация конвертера
        
        Args:
            coordinate_system: Система координат
        """
        self.coord_system = coordinate_system
        self.body_center = None
        self.spine_center = None
        self.body_dimensions = None
        
    def set_reference_points(self,
                          body_center: np.ndarray,
                          spine_center: np.ndarray,
                          body_dimensions: Optional[np.ndarray] = None):
        """
        Установить референтные точки
        
        Args:
            body_center: Центр тела пациента (3 координаты)
            spine_center: Центр позвоночника (3 координаты)
            body_dimensions: Размеры тела (width, depth, height) в мм
        """
        self.body_center = np.array(body_center)
        self.spine_center = np.array(spine_center)
        
        if body_dimensions is not None:
            self.body_dimensions = np.array(body_dimensions)
        else:
            # Стандартные размеры тела если не заданы
            self.body_dimensions = np.array([400, 300, 1700])  # ширина, глубина, рост в мм
    
    def calculate_relative_coordinates(self, 
                                  absolute_coords: np.ndarray) -> RelativeCoordinates:
        """
        Рассчитать относительные координаты для точки или набора точек
        
        Args:
            absolute_coords: Абсолютные координаты (N x 3)
            
        Returns:
            RelativeCoordinates объект
        """
        if self.body_center is None or self.spine_center is None:
            raise ValueError("Референтные точки не установлены. Вызовите set_reference_points()")
        
        # Относительные координаты относительно позвоночника
        relative_to_spine = absolute_coords - self.spine_center
        
        # Относительные координаты относительно центра тела
        relative_to_body_center = absolute_coords - self.body_center
        
        # Нормализованные координаты на размер тела
        normalized_by_body_size = absolute_coords / self.body_dimensions
        
        # Расстояния до референтных точек
        distances = {
            'to_spine_center': np.linalg.norm(relative_to_spine, axis=-1) if absolute_coords.ndim > 1 else np.linalg.norm(relative_to_spine),
            'to_body_center': np.linalg.norm(relative_to_body_center, axis=-1) if absolute_coords.ndim > 1 else np.linalg.norm(relative_to_body_center)
        }
        
        return RelativeCoordinates(
            relative_to_spine=relative_to_spine,
            relative_to_body_center=relative_to_body_center,
            normalized_by_body_size=normalized_by_body_size,
            distances=distances
        )
    
    def calculate_body_center_from_landmarks(self, landmarks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Рассчитать центр тела из анатомических ориентиров
        
        Args:
            landmarks: Словарь с анатомическими точками
            
        Returns:
            Центр тела (3 координаты)
        """
        points = []
        
        # Используем различные анатомические точки для расчета центра
        if 'left_hip' in landmarks and 'right_hip' in landmarks:
            hip_center = (landmarks['left_hip'] + landmarks['right_hip']) / 2
            points.append(hip_center)
            
        if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
            shoulder_center = (landmarks['left_shoulder'] + landmarks['right_shoulder']) / 2
            points.append(shoulder_center)
            
        if 'spine_top' in landmarks and 'spine_bottom' in landmarks:
            spine_center = (landmarks['spine_top'] + landmarks['spine_bottom']) / 2
            points.append(spine_center)
        
        if 'kidney_left_center' in landmarks:
            points.append(landmarks['kidney_left_center'])
            
        if 'kidney_right_center' in landmarks:
            points.append(landmarks['kidney_right_center'])
        
        if not points:
            raise ValueError("Недостаточно анатомических ориентиров для расчета центра тела")
        
        return np.mean(points, axis=0)
    
    def estimate_body_dimensions(self, landmarks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Оценить размеры тела из анатомических ориентиров
        
        Args:
            landmarks: Словарь с анатомическими точками
            
        Returns:
            Размеры тела (width, depth, height) в мм
        """
        width = depth = height = 0
        
        # Ширина тела (медиально-латерально)
        if 'left_hip' in landmarks and 'right_hip' in landmarks:
            width = np.linalg.norm(landmarks['left_hip'] - landmarks['right_hip'])
        elif 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
            width = np.linalg.norm(landmarks['left_shoulder'] - landmarks['right_shoulder'])
        else:
            width = 400  # стандартное значение
        
        # Глубина тела (антеро-постериорно)
        if 'anterior_body' in landmarks and 'posterior_body' in landmarks:
            depth = np.linalg.norm(landmarks['anterior_body'] - landmarks['posterior_body'])
        else:
            depth = 300  # стандартное значение
        
        # Рост (кранио-каудально)
        if 'head_top' in landmarks and 'feet_bottom' in landmarks:
            height = np.linalg.norm(landmarks['head_top'] - landmarks['feet_bottom'])
        elif 'spine_top' in landmarks and 'spine_bottom' in landmarks:
            height = np.linalg.norm(landmarks['spine_top'] - landmarks['spine_bottom']) * 1.2  # с запасом
        else:
            height = 1700  # стандартное значение
        
        return np.array([width, depth, height])
    
    def create_relative_features(self, 
                             absolute_coords: np.ndarray,
                             feature_names: Optional[list] = None) -> Dict[str, np.ndarray]:
        """
        Создать признаки на основе относительных координат
        
        Args:
            absolute_coords: Абсолютные координаты (N x 3)
            feature_names: Имена признаков для генерации
            
        Returns:
            Словарь с относительными признаками
        """
        relative = self.calculate_relative_coordinates(absolute_coords)
        
        features = {}
        
        # Базовые относительные координаты
        features['rel_x_spine'] = relative.relative_to_spine[:, 0] if absolute_coords.ndim > 1 else relative.relative_to_spine[0]
        features['rel_y_spine'] = relative.relative_to_spine[:, 1] if absolute_coords.ndim > 1 else relative.relative_to_spine[1]
        features['rel_z_spine'] = relative.relative_to_spine[:, 2] if absolute_coords.ndim > 1 else relative.relative_to_spine[2]
        
        features['rel_x_body'] = relative.relative_to_body_center[:, 0] if absolute_coords.ndim > 1 else relative.relative_to_body_center[0]
        features['rel_y_body'] = relative.relative_to_body_center[:, 1] if absolute_coords.ndim > 1 else relative.relative_to_body_center[1]
        features['rel_z_body'] = relative.relative_to_body_center[:, 2] if absolute_coords.ndim > 1 else relative.relative_to_body_center[2]
        
        # Нормализованные координаты
        features['norm_x'] = relative.normalized_by_body_size[:, 0] if absolute_coords.ndim > 1 else relative.normalized_by_body_size[0]
        features['norm_y'] = relative.normalized_by_body_size[:, 1] if absolute_coords.ndim > 1 else relative.normalized_by_body_size[1]
        features['norm_z'] = relative.normalized_by_body_size[:, 2] if absolute_coords.ndim > 1 else relative.normalized_by_body_size[2]
        
        # Расстояния
        features['dist_to_spine'] = relative.distances['to_spine_center']
        features['dist_to_body_center'] = relative.distances['to_body_center']
        
        # Дополнительные признаки
        if absolute_coords.ndim > 1:
            # Углы относительно позвоночника
            vectors_to_spine = relative.relative_to_spine
            angles_spine = np.arctan2(vectors_to_spine[:, 1], vectors_to_spine[:, 0])  # угол в XY плоскости
            features['angle_to_spine_xy'] = np.degrees(angles_spine)
            
            # Радиальное расстояние от позвоночника
            features['radial_dist_spine'] = np.sqrt(vectors_to_spine[:, 0]**2 + vectors_to_spine[:, 1]**2)
            
            # Высота относительно позвоночника
            features['height_above_spine'] = vectors_to_spine[:, 2]
        
        return features
    
    def convert_dataset_to_relative(self, 
                                 df,
                                 coord_columns: Dict[str, list],
                                 reference_columns: Optional[Dict[str, str]] = None) -> Dict[str, np.ndarray]:
        """
        Конвертировать весь датасет в относительные координаты
        
        Args:
            df: DataFrame с данными
            coord_columns: Словарь с колонками координат {'point_name': ['x_col', 'y_col', 'z_col']}
            reference_columns: Словарь с колонками референтных точек
            
        Returns:
            Словарь с относительными признаками
        """
        if reference_columns is None:
            # Стандартные имена колонок для референтных точек
            reference_columns = {
                'body_center': ['body_com_x_mm', 'body_com_y_mm', 'body_com_z_mm'],
                'spine_center': ['spine_center_x_mm', 'spine_center_y_mm', 'spine_center_z_mm'],
                'body_dimensions': ['body_width_mm_median', 'body_depth_mm_median', 'body_area_mm2_median']
            }
        
        # Установка референтных точек из первой строки датасета
        first_row = df.iloc[0]
        
        body_center = np.array([first_row[reference_columns['body_center'][i]] for i in range(3)])
        spine_center = np.array([first_row[reference_columns['spine_center'][i]] for i in range(3)])
        
        # Размеры тела
        if all(col in df.columns for col in reference_columns['body_dimensions']):
            body_dimensions = np.array([first_row[reference_columns['body_dimensions'][i]] for i in range(3)])
        else:
            body_dimensions = None
        
        self.set_reference_points(body_center, spine_center, body_dimensions)
        
        # Конвертация всех точек
        relative_features = {}
        
        for point_name, cols in coord_columns.items():
            if all(col in df.columns for col in cols):
                coords = df[cols].values
                relative = self.calculate_relative_coordinates(coords)
                
                # Сохранение относительных координат
                relative_features[f'{point_name}_rel_spine_x'] = relative.relative_to_spine[:, 0]
                relative_features[f'{point_name}_rel_spine_y'] = relative.relative_to_spine[:, 1]
                relative_features[f'{point_name}_rel_spine_z'] = relative.relative_to_spine[:, 2]
                
                relative_features[f'{point_name}_rel_body_x'] = relative.relative_to_body_center[:, 0]
                relative_features[f'{point_name}_rel_body_y'] = relative.relative_to_body_center[:, 1]
                relative_features[f'{point_name}_rel_body_z'] = relative.relative_to_body_center[:, 2]
                
                relative_features[f'{point_name}_dist_to_spine'] = relative.distances['to_spine_center']
                relative_features[f'{point_name}_dist_to_body'] = relative.distances['to_body_center']
        
        return relative_features


def create_relative_converter_from_dicom_features(dicom_features: Dict[str, Any]) -> RelativeCoordinateConverter:
    """
    Создать конвертер относительных координат из DICOM признаков
    
    Args:
        dicom_features: Признаки извлеченные из DICOM
        
    Returns:
        RelativeCoordinateConverter с установленными референтными точками
    """
    coord_system = MedicalCoordinateSystem(CoordinateSystem.LPS)
    converter = RelativeCoordinateConverter(coord_system)
    
    # Извлечение референтных точек из DICOM признаков
    body_center = np.array([
        dicom_features.get('body_com_x_mm', 0),
        dicom_features.get('body_com_y_mm', 0),
        dicom_features.get('body_com_z_mm', 0)
    ])
    
    spine_center = np.array([
        dicom_features.get('spine_center_x_mm', 0),
        dicom_features.get('spine_center_y_mm', 0),
        dicom_features.get('spine_center_z_mm', 0)
    ])
    
    body_dimensions = np.array([
        dicom_features.get('body_width_mm_median', 400),
        dicom_features.get('body_depth_mm_median', 300),
        dicom_features.get('body_area_mm2_median', 120000) ** 0.5  # Квадратный корень из площади
    ])
    
    converter.set_reference_points(body_center, spine_center, body_dimensions)
    
    return converter


if __name__ == "__main__":
    # Тестирование модуля
    print("=== Тестирование относительных координат ===")
    
    # Создание конвертера
    coord_system = MedicalCoordinateSystem(CoordinateSystem.LPS)
    converter = RelativeCoordinateConverter(coord_system)
    
    # Установка референтных точек
    body_center = np.array([0, 0, 0])  # Центр тела
    spine_center = np.array([0, -50, 0])  # Позвоночник на 50мм сзади
    body_dimensions = np.array([400, 300, 1700])  # Ширина, глубина, рост
    
    converter.set_reference_points(body_center, spine_center, body_dimensions)
    
    # Тестовые координаты почки
    kidney_coords = np.array([
        [-100, -30, 200],  # Левая почка
        [100, -25, 190],   # Правая почка
    ])
    
    print("Абсолютные координаты:")
    print(kidney_coords)
    
    # Расчет относительных координат
    relative = converter.calculate_relative_coordinates(kidney_coords)
    
    print(f"\nОтносительно позвоночника:")
    print(relative.relative_to_spine)
    
    print(f"\nОтносительно центра тела:")
    print(relative.relative_to_body_center)
    
    print(f"\nНормализованные на размер тела:")
    print(relative.normalized_by_body_size)
    
    print(f"\nРасстояния:")
    print(relative.distances)
    
    # Создание признаков
    features = converter.create_relative_features(kidney_coords)
    print(f"\nСгенерированные признаки:")
    for name, value in features.items():
        print(f"{name}: {value}")
