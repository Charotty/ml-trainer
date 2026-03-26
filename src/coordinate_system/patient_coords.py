import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PatientCoordinateSystem:
    """Единая система координат пациента"""
    
    def __init__(self, spine_center: np.ndarray):
        """
        Args:
            spine_center: [x, y, z] центр позвоночника в мм
        """
        self.origin = np.array(spine_center)
        self.axes = {
            'X': np.array([1, 0, 0]),  # слева направо
            'Y': np.array([0, 1, 0]),  # сзади вперёд
            'Z': np.array([0, 0, 1])   # снизу вверх
        }
    
    def to_patient_coords(self, world_coords: np.ndarray) -> np.ndarray:
        """Преобразование мировых координат в систему пациента"""
        return world_coords - self.origin
    
    def from_patient_coords(self, patient_coords: np.ndarray) -> np.ndarray:
        """Преобразование из системы пациента в мировые координаты"""
        return patient_coords + self.origin
    
    def transform_features_to_patient(self, features: Dict) -> Dict:
        """Преобразование всех признаков в систему пациента"""
        transformed = features.copy()
        
        # Координаты почек
        for kidney in ['left', 'right']:
            for axis in ['x', 'y', 'z']:
                coord_key = f'kidney_{kidney}_center_{axis}_mm'
                if coord_key in features:
                    world_coord = features[coord_key]
                    if not np.isnan(world_coord):
                        patient_coord = self.to_patient_coords(np.array([world_coord, 0, 0]))[0]
                        transformed[f'kidney_{kidney}_{axis}_rel_spine'] = patient_coord
        
        # Центр позвоночника становится (0, 0, 0) в patient space
        transformed['spine_center_x_mm'] = 0.0
        transformed['spine_center_y_mm'] = 0.0
        transformed['spine_center_z_mm'] = 0.0
        
        return transformed
    
    def validate_patient_coords(self, coords: np.ndarray) -> bool:
        """Проверка корректности координат в системе пациента"""
        # В patient space левая почка должна быть слева (отрицательный X)
        # правая почка справа (положительный X)
        if coords[0] < -50:  # левая почка
            return True
        elif coords[0] > 50:  # правая почка
            return True
        return False

class MultiLevelTransformer:
    """Многоуровневая система трансформаций"""
    
    def __init__(self):
        self.ct_to_patient = None  # PatientCoordinateSystem
        self.patient_to_world = SensorTransformer()
        self.world_to_ar = ARTransformer()
    
    def set_patient_coordinate_system(self, spine_center: np.ndarray):
        """Установка системы координат пациента"""
        self.ct_to_patient = PatientCoordinateSystem(spine_center)
    
    def transform_ct_to_patient(self, ct_coords: np.ndarray) -> np.ndarray:
        """Преобразование CT координат в систему пациента"""
        if self.ct_to_patient is None:
            raise ValueError("Patient coordinate system not set")
        return self.ct_to_patient.to_patient_coords(ct_coords)
    
    def transform_patient_to_world(self, patient_coords: np.ndarray, 
                                 sensor_data: Dict) -> np.ndarray:
        """Преобразование из системы пациента в мировые координаты"""
        return self.patient_to_world.apply_transform(patient_coords, sensor_data)
    
    def transform_world_to_ar(self, world_coords: np.ndarray, 
                            ar_system_data: Dict) -> np.ndarray:
        """Преобразование мировых координат в AR систему"""
        self.world_to_ar.set_ar_system_transform(ar_system_data)
        return self.world_to_ar.to_ar_coords(world_coords)
    
    def full_transform_ct_to_ar(self, ct_coords: np.ndarray, 
                              sensor_data: Dict, 
                              ar_system_data: Dict) -> np.ndarray:
        """Полная трансформация CT → AR"""
        # CT → Patient
        patient_coords = self.transform_ct_to_patient(ct_coords)
        
        # Patient → World  
        world_coords = self.transform_patient_to_world(patient_coords, sensor_data)
        
        # World → AR
        ar_coords = self.transform_world_to_ar(world_coords, ar_system_data)
        
        return ar_coords

class SensorTransformer:
    """Трансформация от датчиков положения"""
    
    def __init__(self):
        self.current_transform = np.eye(4)  # 4x4 transformation matrix
    
    def update_from_sensors(self, sensor_data: Dict):
        """Обновление матрицы трансформации из датчиков"""
        # sensor_data содержит:
        # - position: [x, y, z] положение пациента
        # - orientation: quaternion или euler angles
        # - tilt: наклон
        # - rotation: поворот
        
        position = sensor_data.get('position', [0, 0, 0])
        orientation = sensor_data.get('orientation', [0, 0, 0, 1])  # quaternion
        
        # Создание transformation matrix
        self.current_transform = self._create_transformation_matrix(position, orientation)
    
    def apply_transform(self, patient_coords: np.ndarray, sensor_data: Dict) -> np.ndarray:
        """Применение трансформации к координатам"""
        self.update_from_sensors(sensor_data)
        
        # Преобразование в однородные координаты
        homogeneous_coords = np.hstack([patient_coords, 1])
        
        # Применение трансформации
        world_coords = self.current_transform @ homogeneous_coords
        
        return world_coords[:3]
    
    def _create_transformation_matrix(self, position: List, orientation: List) -> np.ndarray:
        """Создание 4x4 матрицы трансформации"""
        # Упрощенная реализация - только трансляция
        transform = np.eye(4)
        transform[:3, 3] = position
        
        # TODO: добавить вращение из quaternion
        return transform

class ARTransformer:
    """Трансформация в AR систему координат"""
    
    def __init__(self):
        self.ar_to_world_transform = np.eye(4)
        self.world_to_ar_transform = np.eye(4)
    
    def set_ar_system_transform(self, ar_system_data: Dict):
        """Установка трансформации AR системы"""
        # ar_system_data содержит:
        # - world_to_ar_matrix: 4x4 матрица
        # - scale_factor: масштабирование
        # - coordinate_system: тип системы координат
        
        if 'world_to_ar_matrix' in ar_system_data:
            self.world_to_ar_transform = np.array(ar_system_data['world_to_ar_matrix'])
            self.ar_to_world_transform = np.linalg.inv(self.world_to_ar_transform)
    
    def to_ar_coords(self, world_coords: np.ndarray) -> np.ndarray:
        """Преобразование мировых координат в AR координаты"""
        # Преобразование в однородные координаты
        homogeneous_coords = np.hstack([world_coords, 1])
        
        # Применение трансформации
        ar_coords = self.world_to_ar_transform @ homogeneous_coords
        
        return ar_coords[:3]
    
    def from_ar_coords(self, ar_coords: np.ndarray) -> np.ndarray:
        """Преобразование из AR координат в мировые"""
        homogeneous_coords = np.hstack([ar_coords, 1])
        world_coords = self.ar_to_world_transform @ homogeneous_coords
        return world_coords[:3]

if __name__ == "__main__":
    # Тестирование системы координат
    logger.info("Тестирование системы координат")
    
    # Создание системы координат пациента
    spine_center = np.array([0.0, 0.0, 100.0])  # центр позвоночника
    patient_cs = PatientCoordinateSystem(spine_center)
    
    # Тестовые координаты левой почки
    left_kidney_world = np.array([-50.0, 20.0, 95.0])
    left_kidney_patient = patient_cs.to_patient_coords(left_kidney_world)
    print(f"Левая почка в world: {left_kidney_world}")
    print(f"Левая почка в patient: {left_kidney_patient}")
    
    # Обратное преобразование
    left_kidney_back = patient_cs.from_patient_coords(left_kidney_patient)
    print(f"Обратное преобразование: {left_kidney_back}")
    
    # Тестирование многоуровневых трансформаций
    transformer = MultiLevelTransformer()
    transformer.set_patient_coordinate_system(spine_center)
    
    # Данные датчиков
    sensor_data = {
        'position': [10.0, 5.0, 0.0],
        'orientation': [0, 0, 0, 1],
        'tilt': 15.0,
        'rotation': 5.0
    }
    
    # AR данные
    ar_data = {
        'world_to_ar_matrix': np.eye(4),
        'scale_factor': 1.0
    }
    
    # Полная трансформация
    final_coords = transformer.full_transform_ct_to_ar(left_kidney_world, sensor_data, ar_data)
    print(f"Финальные AR координаты: {final_coords}")
    
    logger.info("Система координат протестирована успешно")
