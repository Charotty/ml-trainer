import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

class KidneyGeometryModel:
    """Параметрическая модель почки (capsule/ellipsoid)"""
    
    def __init__(self, center: np.ndarray, length: float, width: float, depth: float, 
                 orientation: np.ndarray = None):
        """
        Args:
            center: [x, y, z] центр почки в мм
            length: длина по оси Z в мм
            width: ширина по оси X в мм  
            depth: глубина по оси Y в мм
            orientation: вектор ориентации [x, y, z]
        """
        self.center = np.array(center)
        self.length = length
        self.width = width
        self.depth = depth
        self.orientation = orientation if orientation is not None else np.array([0, 0, 1])
        
    def get_capsule_points(self, n_points: int = 50) -> np.ndarray:
        """Генерирует точки на поверхности capsule"""
        # Capsule = цилиндр + полусферы на концах
        points = []
        
        # Генерируем точки по окружности и длине
        theta = np.linspace(0, 2*np.pi, n_points//2, endpoint=False)
        
        # Боковая поверхность (цилиндр)
        for t in theta:
            # Точки на окружности
            x = self.width/2 * np.cos(t)
            y = self.depth/2 * np.sin(t)
            
            # Вдоль длины почки
            for z_frac in np.linspace(-0.4, 0.4, 5):  # 80% длины для цилиндра
                z = self.center[2] + z_frac * self.length
                point = self.center + np.array([x, y, 0]) + z_frac * self.length * self.orientation
                points.append(point)
        
        # Верхняя и нижняя полусферы
        for end_z in [-self.length/2, self.length/2]:
            for t in theta:
                x = self.width/2 * np.cos(t)
                y = self.depth/2 * np.sin(t)
                point = self.center + np.array([x, y, end_z])
                points.append(point)
        
        return np.array(points)
    
    def get_ellipsoid_points(self, n_points: int = 50) -> np.ndarray:
        """Генерирует точки на поверхности эллипсоида"""
        u = np.linspace(0, 2 * np.pi, n_points//2)
        v = np.linspace(0, np.pi, n_points//2)
        
        points = []
        for i in range(len(u)):
            for j in range(len(v)):
                x = self.width/2 * np.sin(v[j]) * np.cos(u[i])
                y = self.depth/2 * np.sin(v[j]) * np.sin(u[i])
                z = self.length/2 * np.cos(v[j])
                
                point = self.center + np.array([x, y, z])
                points.append(point)
        
        return np.array(points[:n_points])
    
    def apply_displacement(self, delta: np.ndarray) -> 'KidneyGeometryModel':
        """Применяет смещение к модели"""
        new_center = self.center + delta
        return KidneyGeometryModel(
            center=new_center,
            length=self.length,
            width=self.width,
            depth=self.depth,
            orientation=self.orientation
        )
    
    def transform_to_world(self, patient_transform: np.ndarray) -> np.ndarray:
        """Преобразует точки в мировые координаты"""
        points = self.get_capsule_points()
        # patient_transform: 4x4 матрица трансформации
        homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
        world_points = (patient_transform @ homogeneous_points.T).T
        return world_points[:, :3]

class StatisticalKidneyModel:
    """Статистическая средняя модель почки"""
    
    def __init__(self):
        # Средние параметры из обучающих данных
        self.avg_params = {
            'left': {
                'center': np.array([-50.0, 20.0, 100.0]),  # относительно позвоночника
                'length': 80.0,
                'width': 40.0,
                'depth': 30.0
            },
            'right': {
                'center': np.array([50.0, 20.0, 100.0]),
                'length': 80.0,
                'width': 40.0,
                'depth': 30.0
            }
        }
        
    def get_model(self, kidney_type: str, patient_scale: float = 1.0) -> KidneyGeometryModel:
        """Возвращает модель для почки с масштабированием"""
        params = self.avg_params[kidney_type]
        return KidneyGeometryModel(
            center=params['center'] * patient_scale,
            length=params['length'] * patient_scale,
            width=params['width'] * patient_scale,
            depth=params['depth'] * patient_scale
        )

def create_personal_kidney_model(ct_features: Dict) -> Tuple[KidneyGeometryModel, KidneyGeometryModel]:
    """Создает персональную модель из КТ признаков"""
    
    # Левая почка
    left_model = KidneyGeometryModel(
        center=np.array([
            ct_features.get('kidney_left_center_x_mm', 0),
            ct_features.get('kidney_left_center_y_mm', 0),
            ct_features.get('kidney_left_center_z_mm', 0)
        ]),
        length=ct_features.get('kidney_left_length_mm', 80.0),
        width=ct_features.get('kidney_left_bbox_width_mm', 40.0),
        depth=ct_features.get('kidney_left_bbox_height_mm', 30.0)
    )
    
    # Правая почка
    right_model = KidneyGeometryModel(
        center=np.array([
            ct_features.get('kidney_right_center_x_mm', 0),
            ct_features.get('kidney_right_center_y_mm', 0),
            ct_features.get('kidney_right_center_z_mm', 0)
        ]),
        length=ct_features.get('kidney_right_length_mm', 80.0),
        width=ct_features.get('kidney_right_bbox_width_mm', 40.0),
        depth=ct_features.get('kidney_right_bbox_height_mm', 30.0)
    )
    
    return left_model, right_model

def get_fallback_model(patient_bmi: float = None) -> Tuple[KidneyGeometryModel, KidneyGeometryModel]:
    """Возвращает fallback модель на основе статистики"""
    stat_model = StatisticalKidneyModel()
    
    # Масштабирование на основе BMI если доступно
    scale = 1.0
    if patient_bmi:
        # Простая эвристика: чем выше BMI, тем крупнее почки
        scale = 0.8 + (patient_bmi / 30.0) * 0.4
        scale = np.clip(scale, 0.7, 1.3)
    
    left_model = stat_model.get_model('left', scale)
    right_model = stat_model.get_model('right', scale)
    
    return left_model, right_model

if __name__ == "__main__":
    # Тестирование
    logger.info("Тестирование параметрической модели почки")
    
    # Персональная модель
    ct_features = {
        'kidney_left_center_x_mm': -45.2,
        'kidney_left_center_y_mm': 18.5,
        'kidney_left_center_z_mm': 95.3,
        'kidney_left_length_mm': 82.1,
        'kidney_left_bbox_width_mm': 38.5,
        'kidney_left_bbox_height_mm': 31.2
    }
    
    left_personal, right_personal = create_personal_kidney_model(ct_features)
    logger.info(f"Персональная модель создана: центр левой почки {left_personal.center}")
    
    # Fallback модель
    left_fallback, right_fallback = get_fallback_model(patient_bmi=22.5)
    logger.info(f"Fallback модель создана: центр левой почки {left_fallback.center}")
    
    # Применение смещения
    delta = np.array([5.2, -3.1, 2.8])
    left_displaced = left_personal.apply_displacement(delta)
    logger.info(f"Смещение применено: новый центр {left_displaced.center}")
    
    # Генерация polygon точек
    polygon_points = left_displaced.get_capsule_points(n_points=50)
    logger.info(f"Сгенерировано {len(polygon_points)} точек для polygon")
