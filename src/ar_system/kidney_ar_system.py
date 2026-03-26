import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from pathlib import Path
import sys

# Добавляем src в Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Импорты наших компонентов
from geometry.kidney_model import KidneyGeometryModel, create_personal_kidney_model, get_fallback_model
from coordinate_system.patient_coords import PatientCoordinateSystem, MultiLevelTransformer
from preprocessing.unified_pipeline import UnifiedPreprocessingPipeline
from reliability.confidence_constraints import (
    ConfidenceEstimator, AnatomicalConstraints, 
    FallbackHandler, TemporalSmoother
)

logger = logging.getLogger(__name__)

class KidneyARSystem:
    """Основная система предсказания смещения почек для AR"""
    
    def __init__(self, model_path: str = None, pipeline_path: str = None):
        """
        Args:
            model_path: путь к обученной ML модели
            pipeline_path: путь к preprocessing pipeline
        """
        self.models = []
        self.pipeline = None
        self.confidence_estimator = None
        self.constraints = None
        self.fallback_handler = None
        self.temporal_smoother = None
        self.coordinate_transformer = MultiLevelTransformer()
        
        # Загрузка компонентов
        self._load_components(model_path, pipeline_path)
        
        # Инициализация систем
        self._initialize_systems()
    
    def _load_components(self, model_path: str, pipeline_path: str):
        """Загрузка обученных компонентов"""
        try:
            # Загрузка ML моделей
            if model_path and Path(model_path).exists():
                model_data = joblib.load(model_path)
                self.models = model_data['models']
                self.train_data = model_data['train_data']
                logger.info(f"Загружено {len(self.models)} ML моделей")
            
            # Загрузка pipeline
            if pipeline_path and Path(pipeline_path).exists():
                self.pipeline = UnifiedPreprocessingPipeline.load_pipeline(pipeline_path)
                logger.info("Preprocessing pipeline загружен")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки компонентов: {e}")
    
    def _initialize_systems(self):
        """Инициализация системных компонентов"""
        # Настройка анатомических ограничений
        body_limits = {
            'x_min': -150, 'x_max': 150,
            'y_min': -100, 'y_max': 100, 
            'z_min': 50, 'z_max': 150
        }
        spine_center = np.array([0, 0, 100])
        self.constraints = AnatomicalConstraints(body_limits, spine_center)
        
        # Настройка fallback handler
        self.fallback_handler = FallbackHandler(None, self.constraints)
        
        # Настройка temporal smoother
        self.temporal_smoother = TemporalSmoother(method='exponential', alpha=0.7)
        
        # Настройка confidence estimator
        if self.models and hasattr(self, 'train_data'):
            self.confidence_estimator = ConfidenceEstimator(self.models, self.train_data)
        
        logger.info("Системные компоненты инициализированы")
    
    def predict_kidney_displacement(self, patient_data: Dict, sensor_data: Dict, 
                                  ar_system_data: Dict) -> Dict:
        """
        Основной метод предсказания смещения почек
        
        Args:
            patient_data: данные пациента (демография, КТ признаки)
            sensor_data: данные датчиков (положение, ориентация)
            ar_system_data: данные AR системы
            
        Returns:
            Dict с результатами предсказания
        """
        logger.info("Начало предсказания смещения почек")
        
        try:
            # 1. Валидация входных данных
            validation_result = self._validate_input_data(patient_data)
            if not validation_result['is_valid']:
                return self._create_error_response(validation_result['errors'])
            
            # 2. Создание персональной модели почек
            left_kidney_model, right_kidney_model = self._create_kidney_models(patient_data)
            
            # 3. Предобработка признаков
            features = self._preprocess_features(patient_data)
            
            # 4. ML предсказание смещения
            ml_prediction = self._predict_displacement(features)
            
            # 5. Оценка уверенности
            confidence = self._estimate_confidence(features)
            
            # 6. Применение ограничений и fallback
            final_displacement = self._apply_constraints_and_fallback(
                features, ml_prediction, confidence, patient_data
            )
            
            # 7. Временное сглаживание
            smoothed_displacement = self.temporal_smoother.smooth(final_displacement)
            
            # 8. Применение смещения к моделям почек
            left_displaced = left_kidney_model.apply_displacement(smoothed_displacement[:3])
            right_displaced = right_kidney_model.apply_displacement(smoothed_displacement[3:6])
            
            # 9. Трансформация в AR координаты
            left_ar_coords = self._transform_to_ar(left_displaced, sensor_data, ar_system_data)
            right_ar_coords = self._transform_to_ar(right_displaced, sensor_data, ar_system_data)
            
            # 10. Генерация polygon точек
            left_polygon = left_displaced.get_capsule_points(n_points=50)
            right_polygon = right_displaced.get_capsule_points(n_points=50)
            
            # 11. Трансформация polygon в AR
            left_polygon_ar = self._transform_polygon_to_ar(left_polygon, sensor_data, ar_system_data)
            right_polygon_ar = self._transform_polygon_to_ar(right_polygon, sensor_data, ar_system_data)
            
            result = {
                'success': True,
                'left_kidney': {
                    'center': left_ar_coords.tolist(),
                    'polygon': left_polygon_ar.tolist(),
                    'displacement': smoothed_displacement[:3].tolist()
                },
                'right_kidney': {
                    'center': right_ar_coords.tolist(), 
                    'polygon': right_polygon_ar.tolist(),
                    'displacement': smoothed_displacement[3:6].tolist()
                },
                'confidence': confidence,
                'processing_time': 0.0,  # TODO: добавить замер времени
                'warnings': validation_result['warnings']
            }
            
            logger.info(f"Предсказание завершено успешно, confidence: {confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return self._create_error_response([str(e)])
    
    def _validate_input_data(self, patient_data: Dict) -> Dict:
        """Валидация входных данных"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Проверка обязательных полей
        required_fields = ['age', 'bmi', 'kidney_left_center_x_mm', 'kidney_right_center_x_mm']
        
        for field in required_fields:
            if field not in patient_data:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['is_valid'] = False
        
        # Проверка диапазонов
        if 'age' in patient_data:
            age = patient_data['age']
            if not (0 <= age <= 100):
                validation_result['errors'].append(f"Invalid age: {age}")
                validation_result['is_valid'] = False
        
        if 'bmi' in patient_data:
            bmi = patient_data['bmi']
            if not (10 <= bmi <= 60):
                validation_result['warnings'].append(f"Unusual BMI: {bmi}")
        
        return validation_result
    
    def _create_kidney_models(self, patient_data: Dict) -> Tuple[KidneyGeometryModel, KidneyGeometryModel]:
        """Создание моделей почек"""
        try:
            # Персональная модель
            left_model, right_model = create_personal_kidney_model(patient_data)
            logger.info("Создана персональная модель почек")
        except Exception as e:
            logger.warning(f"Ошибка создания персональной модели: {e}")
            # Fallback модель
            bmi = patient_data.get('bmi', 22.5)
            left_model, right_model = get_fallback_model(bmi)
            logger.info("Использована fallback модель почек")
        
        return left_model, right_model
    
    def _preprocess_features(self, patient_data: Dict) -> np.ndarray:
        """Предобработка признаков"""
        # Преобразование в DataFrame
        df = pd.DataFrame([patient_data])
        
        # Применение pipeline
        if self.pipeline:
            features = self.pipeline.transform(df)
        else:
            # Упрощенная обработка если pipeline не загружен
            features = self._simple_preprocessing(df)
        
        return features[0]  # возвращаем один образец
    
    def _simple_preprocessing(self, df: pd.DataFrame) -> np.ndarray:
        """Упрощенная предобработка если pipeline не доступен"""
        # Базовые признаки
        basic_features = ['age', 'bmi', 'kidney_left_center_x_mm', 'kidney_right_center_x_mm']
        
        features_array = []
        for feature in basic_features:
            if feature in df.columns:
                features_array.append(df[feature].iloc[0])
            else:
                features_array.append(0.0)
        
        return np.array([features_array])
    
    def _predict_displacement(self, features: np.ndarray) -> np.ndarray:
        """ML предсказание смещения"""
        if not self.models:
            # Простейшее предсказание если модели не загружены
            return np.array([5.0, -3.0, 2.0, 5.0, -3.0, 2.0])
        
        # Ансамбль предсказаний
        predictions = []
        for model in self.models:
            pred = model.predict(features.reshape(1, -1))
            predictions.append(pred[0])
        
        # Усреднение предсказаний
        final_prediction = np.mean(predictions, axis=0)
        
        return final_prediction
    
    def _estimate_confidence(self, features: np.ndarray) -> float:
        """Оценка уверенности предсказания"""
        if self.confidence_estimator:
            return self.confidence_estimator.calculate_confidence(features)
        else:
            return 0.7  # значение по умолчанию
    
    def _apply_constraints_and_fallback(self, features: np.ndarray, prediction: np.ndarray, 
                                      confidence: float, patient_data: Dict) -> np.ndarray:
        """Применение ограничений и fallback логики"""
        # Получаем исходное положение
        original_left = np.array([
            patient_data.get('kidney_left_center_x_mm', 0),
            patient_data.get('kidney_left_center_y_mm', 0),
            patient_data.get('kidney_left_center_z_mm', 0)
        ])
        
        original_right = np.array([
            patient_data.get('kidney_right_center_x_mm', 0),
            patient_data.get('kidney_right_center_y_mm', 0),
            patient_data.get('kidney_right_center_z_mm', 0)
        ])
        
        # Проверяем размерность prediction
        if len(prediction) == 6:
            # Предсказание для обеих почек
            final_prediction = prediction
        elif len(prediction) == 3:
            # Предсказание для одной почки, дублируем для обеих
            final_prediction = np.concatenate([prediction, prediction])
        else:
            # Некорректная размерность, используем значения по умолчанию
            final_prediction = np.array([5.0, -3.0, 2.0, 5.0, -3.0, 2.0])
        
        # Применение fallback handler
        if self.fallback_handler:
            # Обрабатываем каждую почку отдельно
            left_final = self.fallback_handler.handle_prediction(
                features, final_prediction[:3], confidence, original_left
            )
            right_final = self.fallback_handler.handle_prediction(
                features, final_prediction[3:6], confidence, original_right
            )
            final_prediction = np.concatenate([left_final, right_final])
        
        return final_prediction
    
    def _transform_to_ar(self, kidney_model: KidneyGeometryModel, 
                        sensor_data: Dict, ar_system_data: Dict) -> np.ndarray:
        """Трансформация координат почки в AR систему"""
        # Установка системы координат пациента
        spine_center = np.array([0, 0, 100])  # TODO: получить из данных
        self.coordinate_transformer.set_patient_coordinate_system(spine_center)
        
        # Трансформация
        ar_coords = self.coordinate_transformer.full_transform_ct_to_ar(
            kidney_model.center, sensor_data, ar_system_data
        )
        
        return ar_coords
    
    def _transform_polygon_to_ar(self, polygon_points: np.ndarray, 
                                sensor_data: Dict, ar_system_data: Dict) -> np.ndarray:
        """Трансформация polygon точек в AR систему"""
        # Установка системы координат пациента
        spine_center = np.array([0, 0, 100])
        self.coordinate_transformer.set_patient_coordinate_system(spine_center)
        
        # Трансформация каждой точки
        ar_polygon = []
        for point in polygon_points:
            ar_point = self.coordinate_transformer.full_transform_ct_to_ar(
                point, sensor_data, ar_system_data
            )
            ar_polygon.append(ar_point)
        
        return np.array(ar_polygon)
    
    def _create_error_response(self, errors: List[str]) -> Dict:
        """Создание ответа с ошибкой"""
        return {
            'success': False,
            'error': 'Prediction failed',
            'details': errors,
            'left_kidney': None,
            'right_kidney': None,
            'confidence': 0.0
        }
    
    def reset_smoothing(self):
        """Сброс временного сглаживания"""
        if self.temporal_smoother:
            self.temporal_smoother.reset()
        logger.info("Временное сглаживание сброшено")

if __name__ == "__main__":
    # Тестирование интегрированной системы
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Тестирование интегрированной системы KidneyAR")
    
    # Создание системы
    system = KidneyARSystem()
    
    # Тестовые данные пациента
    patient_data = {
        'age': 45,
        'bmi': 24.5,
        'sex_encoded': 1,
        'kidney_left_center_x_mm': -45.2,
        'kidney_left_center_y_mm': 18.5,
        'kidney_left_center_z_mm': 95.3,
        'kidney_right_center_x_mm': 52.1,
        'kidney_right_center_y_mm': 19.8,
        'kidney_right_center_z_mm': 96.7
    }
    
    # Данные датчиков
    sensor_data = {
        'position': [10.0, 5.0, 0.0],
        'orientation': [0, 0, 0, 1],
        'tilt': 15.0,
        'rotation': 5.0
    }
    
    # AR данные
    ar_system_data = {
        'world_to_ar_matrix': np.eye(4),
        'scale_factor': 1.0
    }
    
    # Предсказание
    result = system.predict_kidney_displacement(patient_data, sensor_data, ar_system_data)
    
    print("Результат предсказания:")
    print(f"Success: {result['success']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if result['success']:
        print(f"Left kidney center: {result['left_kidney']['center']}")
        print(f"Right kidney center: {result['right_kidney']['center']}")
        print(f"Left polygon points: {len(result['left_kidney']['polygon'])}")
        print(f"Right polygon points: {len(result['right_kidney']['polygon'])}")
    else:
        print(f"Error: {result['error']}")
        print(f"Details: {result['details']}")
    
    logger.info("Тестирование интегрированной системы завершено")
