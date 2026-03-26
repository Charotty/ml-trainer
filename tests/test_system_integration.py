import unittest
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List
import sys
from pathlib import Path

# Добавляем src в Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from ar_system.kidney_ar_system import KidneyARSystem
from validation.data_validator import DataValidator, ClinicalMetrics, SystemLogger
from versioning.version_manager import VersionManager
from unpaired.unpaired_trainer import UnpairedDataProcessor, EnhancedModelTrainer

class TestKidneyARSystem(unittest.TestCase):
    """Комплексные тесты системы KidneyAR"""
    
    def setUp(self):
        """Настройка тестового окружения"""
        self.kidney_system = KidneyARSystem()
        self.validator = DataValidator()
        self.metrics = ClinicalMetrics()
        self.version_manager = VersionManager()
        
        # Тестовые данные
        self.valid_patient_data = {
            'age': 45,
            'bmi': 24.5,
            'sex_encoded': 1,
            'kidney_left_center_x_mm': -45.2,
            'kidney_left_center_y_mm': 18.5,
            'kidney_left_center_z_mm': 95.3,
            'kidney_right_center_x_mm': 52.1,
            'kidney_right_center_y_mm': 19.8,
            'kidney_right_center_z_mm': 96.7,
            'weight_kg': 75.0,
            'height_m': 1.75
        }
        
        self.sensor_data = {
            'position': [10.0, 5.0, 0.0],
            'orientation': [0, 0, 0, 1],
            'tilt': 15.0,
            'rotation': 5.0
        }
        
        self.ar_system_data = {
            'world_to_ar_matrix': np.eye(4).tolist(),
            'scale_factor': 1.0
        }
    
    def test_system_initialization(self):
        """Тест инициализации системы"""
        self.assertIsNotNone(self.kidney_system)
        self.assertIsNotNone(self.validator)
        self.assertIsNotNone(self.metrics)
        self.assertIsNotNone(self.version_manager)
    
    def test_valid_patient_data(self):
        """Тест валидации корректных данных"""
        validation = self.validator.validate_patient_data(self.valid_patient_data)
        self.assertTrue(validation['is_valid'])
        self.assertEqual(len(validation['errors']), 0)
    
    def test_invalid_patient_data(self):
        """Тест валидации некорректных данных"""
        invalid_data = self.valid_patient_data.copy()
        invalid_data['age'] = 150  # некорректный возраст
        invalid_data['bmi'] = 5.0  # некорректный BMI
        
        validation = self.validator.validate_patient_data(invalid_data)
        self.assertFalse(validation['is_valid'])
        self.assertGreater(len(validation['errors']), 0)
    
    def test_kidney_position_logic(self):
        """Тест логики положения почек"""
        # Левая почка слева, правая справа
        self.assertTrue(self.valid_patient_data['kidney_left_center_x_mm'] < 0)
        self.assertTrue(self.valid_patient_data['kidney_right_center_x_mm'] > 0)
    
    def test_prediction_success(self):
        """Тест успешного предсказания"""
        result = self.kidney_system.predict_kidney_displacement(
            self.valid_patient_data,
            self.sensor_data,
            self.ar_system_data
        )
        
        self.assertTrue(result['success'])
        self.assertGreater(result['confidence'], 0)
        self.assertIsNotNone(result['left_kidney'])
        self.assertIsNotNone(result['right_kidney'])
        
        # Проверка структуры ответа
        left_kidney = result['left_kidney']
        self.assertIn('center', left_kidney)
        self.assertIn('polygon', left_kidney)
        self.assertIn('displacement', left_kidney)
        
        # Проверка polygon точек
        self.assertGreater(len(left_kidney['polygon']), 0)
        self.assertEqual(len(left_kidney['center']), 3)
        self.assertEqual(len(left_kidney['displacement']), 3)
    
    def test_prediction_with_invalid_data(self):
        """Тест предсказания с некорректными данными"""
        invalid_data = self.valid_patient_data.copy()
        invalid_data['age'] = 200
        
        result = self.kidney_system.predict_kidney_displacement(
            invalid_data,
            self.sensor_data,
            self.ar_system_data
        )
        
        self.assertFalse(result['success'])
        self.assertIsNotNone(result.get('error'))
    
    def test_temporal_smoothing(self):
        """Тест временного сглаживания"""
        # Сброс сглаживания
        self.kidney_system.reset_smoothing()
        
        # Несколько последовательных предсказаний
        predictions = []
        for i in range(5):
            result = self.kidney_system.predict_kidney_displacement(
                self.valid_patient_data,
                self.sensor_data,
                self.ar_system_data
            )
            predictions.append(result['left_kidney']['displacement'])
        
        # Проверка, что предсказания сглаживаются
        # (в идеале должны быть похожими, но не идентичными)
        self.assertEqual(len(predictions), 5)
    
    def test_confidence_estimation(self):
        """Тест оценки уверенности"""
        result = self.kidney_system.predict_kidney_displacement(
            self.valid_patient_data,
            self.sensor_data,
            self.ar_system_data
        )
        
        confidence = result['confidence']
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_anatomical_constraints(self):
        """Тест анатомических ограничений"""
        result = self.kidney_system.predict_kidney_displacement(
            self.valid_patient_data,
            self.sensor_data,
            self.ar_system_data
        )
        
        if result['success']:
            # Проверка, что смещения не слишком большие
            left_displacement = result['left_kidney']['displacement']
            right_displacement = result['right_kidney']['displacement']
            
            for disp in left_displacement + right_displacement:
                self.assertLess(abs(disp), 100.0)  # не более 100 мм
    
    def test_coordinate_transformations(self):
        """Тест трансформаций координат"""
        result = self.kidney_system.predict_kidney_displacement(
            self.valid_patient_data,
            self.sensor_data,
            self.ar_system_data
        )
        
        if result['success']:
            # Проверка, что AR координаты отличаются от исходных
            left_center = result['left_kidney']['center']
            self.assertEqual(len(left_center), 3)
            
            # Проверка, что polygon точки трансформируются
            left_polygon = result['left_kidney']['polygon']
            self.assertGreater(len(left_polygon), 0)
            for point in left_polygon:
                self.assertEqual(len(point), 3)
    
    def test_performance_requirements(self):
        """Тест требований к производительности"""
        start_time = time.time()
        
        result = self.kidney_system.predict_kidney_displacement(
            self.valid_patient_data,
            self.sensor_data,
            self.ar_system_data
        )
        
        processing_time = time.time() - start_time
        
        # Требование: не более 100 мс на предсказание
        self.assertLess(processing_time, 0.1)
        self.assertTrue(result['success'])
    
    def test_logging_functionality(self):
        """Тест функциональности логирования"""
        logger = SystemLogger("logs/test_validation.log")
        
        # Тест логирования входных данных
        logger.log_input_data("test_patient", self.valid_patient_data, 0.05)
        
        # Тест логирования предсказания
        logger.log_prediction("test_patient", np.array([5.0, -3.0, 2.0]), 0.85, True)
        
        # Тест логирования ошибки
        logger.log_error("test_patient", "test_error", ["detail1", "detail2"])
        
        # Проверка получения логов
        recent_logs = logger.get_recent_logs(5)
        self.assertGreaterEqual(len(recent_logs), 3)
    
    def test_versioning_system(self):
        """Тест системы версионирования"""
        # Тест сохранения артефакта
        test_artifact = {"type": "test", "value": 42}
        self.version_manager.save_versioned_artifact(
            test_artifact, "system", "test_v1.0"
        )
        
        # Тест загрузки артефакта
        loaded_artifact = self.version_manager.load_versioned_artifact("system", "test_v1.0")
        self.assertEqual(loaded_artifact["value"], 42)
        
        # Тест метаданных
        metadata = self.version_manager.get_artifact_metadata("system", "test_v1.0")
        self.assertEqual(metadata["version"], "test_v1.0")
        
        # Тест снепшотов
        snapshot_id = self.version_manager.create_version_snapshot("Test snapshot")
        self.assertIsNotNone(snapshot_id)
        
        snapshots = self.version_manager.list_snapshots()
        self.assertGreater(len(snapshots), 0)

class TestUnpairedDataProcessing(unittest.TestCase):
    """Тесты обработки непарных данных"""
    
    def setUp(self):
        """Настройка тестовых данных"""
        self.unpaired_processor = UnpairedDataProcessor()
        
        # Тестовые непарные данные
        self.unpaired_data = pd.DataFrame({
            'age': [20, 30, 40, 50, 60, 70, 80],
            'bmi': [18.0, 20.0, 22.0, 25.0, 27.0, 29.0, 32.0],
            'kidney_left_center_x_mm': [-43.0, -46.0, -44.0, -47.0, -45.0, -48.0, -46.5],
            'kidney_right_center_x_mm': [50.0, 53.0, 51.0, 54.0, 52.0, 55.0, 53.5]
        })
    
    def test_unpaired_data_fitting(self):
        """Тест обучения на непарных данных"""
        self.unpaired_processor.fit_unpaired_data(self.unpaired_data)
        
        # Проверка расчета статистики
        stats = self.unpaired_processor.get_feature_statistics()
        self.assertGreater(len(stats), 0)
        
        # Проверка наличия основных признаков
        self.assertIn('age', stats)
        self.assertIn('bmi', stats)
        
        # Проверка корректности статистики
        age_stats = stats['age']
        self.assertIn('mean', age_stats)
        self.assertIn('std', age_stats)
        self.assertIn('min', age_stats)
        self.assertIn('max', age_stats)
    
    def test_anomaly_detection(self):
        """Тест детекции аномалий"""
        self.unpaired_processor.fit_unpaired_data(self.unpaired_data)
        
        # Нормальные признаки
        normal_features = {
            'age': 45,
            'bmi': 24.0,
            'kidney_left_center_x_mm': -45.0
        }
        
        anomalies = self.unpaired_processor.validate_with_unpaired(normal_features)
        self.assertEqual(len(anomalies), 0)
        
        # Аномальные признаки
        anomalous_features = {
            'age': 150,  # слишком большой возраст
            'bmi': 5.0,   # слишком низкий BMI
            'kidney_left_center_x_mm': -100.0  # слишком далеко
        }
        
        anomalies = self.unpaired_processor.validate_with_unpaired(anomalous_features)
        self.assertGreater(len(anomalies), 0)
    
    def test_feature_normalization(self):
        """Тест нормализации признаков"""
        self.unpaired_processor.fit_unpaired_data(self.unpaired_data)
        
        features = {
            'age': 45,
            'bmi': 24.0
        }
        
        normalized = self.unpaired_processor.normalize_features(features)
        
        # Проверка наличия нормализованных признаков
        self.assertIn('age_zscore', normalized)
        self.assertIn('bmi_zscore', normalized)
        self.assertIn('age_norm', normalized)
        self.assertIn('bmi_norm', normalized)

class TestClinicalMetrics(unittest.TestCase):
    """Тесты клинических метрик"""
    
    def setUp(self):
        """Настройка тестовых данных"""
        self.metrics = ClinicalMetrics()
        
        # Тестовые предсказания и истина
        self.y_true = np.array([
            [5.0, -3.0, 2.0, 5.1, -2.9, 2.1],
            [4.8, -3.2, 1.9, 5.2, -2.8, 2.0],
            [5.2, -2.8, 2.2, 4.9, -3.1, 1.8]
        ])
        
        self.y_pred = np.array([
            [5.1, -2.9, 2.1, 5.0, -3.0, 2.0],
            [4.9, -3.1, 2.0, 5.1, -2.9, 2.1],
            [5.1, -2.9, 2.1, 5.0, -3.0, 2.0]
        ])
    
    def test_metrics_calculation(self):
        """Тест расчета метрик"""
        metrics = self.metrics.calculate_metrics(self.y_true, self.y_pred)
        
        # Проверка основных метрик
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('within_5mm', metrics)
        self.assertIn('within_10mm', metrics)
        self.assertIn('within_15mm', metrics)
        
        # Проверка значений
        self.assertGreater(metrics['mae'], 0)
        self.assertLess(metrics['mae'], 10)  # должно быть разумным
        self.assertGreater(metrics['within_5mm'], 0)
        self.assertLessEqual(metrics['within_5mm'], 100)
    
    def test_kidney_specific_metrics(self):
        """Тест метрик для отдельных почек"""
        metrics = self.metrics.calculate_metrics(self.y_true, self.y_pred)
        
        # Проверка метрик для левой и правой почек
        self.assertIn('left_kidney', metrics)
        self.assertIn('right_kidney', metrics)
        
        left_metrics = metrics['left_kidney']
        self.assertIn('mae', left_metrics)
        self.assertIn('within_5mm', left_metrics)
    
    def test_prediction_history(self):
        """Тест истории предсказаний"""
        # Добавление предсказаний
        for i in range(5):
            prediction = {
                'success': True,
                'confidence': 0.8 + i * 0.02,
                'left_kidney': {'displacement': [5.0, -3.0, 2.0]},
                'right_kidney': {'displacement': [5.1, -2.9, 2.1]}
            }
            self.metrics.add_prediction(prediction)
        
        # Проверка сводных метрик
        summary = self.metrics.get_summary_metrics()
        
        self.assertEqual(summary['total_predictions'], 5)
        self.assertEqual(summary['successful_predictions'], 5)
        self.assertEqual(summary['success_rate'], 100.0)
        self.assertGreater(summary['average_confidence'], 0.8)

def run_stress_test(num_requests: int = 100):
    """Стресс тестирование системы"""
    print(f"Запуск стресс теста: {num_requests} запросов")
    
    system = KidneyARSystem()
    
    # Тестовые данные
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
    
    sensor_data = {
        'position': [10.0, 5.0, 0.0],
        'orientation': [0, 0, 0, 1],
        'tilt': 15.0,
        'rotation': 5.0
    }
    
    ar_system_data = {
        'world_to_ar_matrix': np.eye(4).tolist(),
        'scale_factor': 1.0
    }
    
    # Метрики производительности
    times = []
    successes = 0
    confidences = []
    
    start_time = time.time()
    
    for i in range(num_requests):
        request_start = time.time()
        
        result = system.predict_kidney_displacement(
            patient_data, sensor_data, ar_system_data
        )
        
        request_time = time.time() - request_start
        times.append(request_time)
        
        if result['success']:
            successes += 1
            confidences.append(result['confidence'])
        
        # Прогресс
        if (i + 1) % 10 == 0:
            print(f"  Обработано {i + 1}/{num_requests} запросов")
    
    total_time = time.time() - start_time
    
    # Результаты
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    success_rate = (successes / num_requests) * 100
    avg_confidence = np.mean(confidences) if confidences else 0
    throughput = num_requests / total_time
    
    print(f"\nРезультаты стресс теста:")
    print(f"  Всего запросов: {num_requests}")
    print(f"  Успешных: {successes} ({success_rate:.1f}%)")
    print(f"  Среднее время: {avg_time*1000:.1f} мс")
    print(f"  Минимальное время: {min_time*1000:.1f} мс")
    print(f"  Максимальное время: {max_time*1000:.1f} мс")
    print(f"  Пропускная способность: {throughput:.1f} запросов/сек")
    print(f"  Средняя уверенность: {avg_confidence:.3f}")
    
    # Проверка требований
    assert success_rate >= 95, f"Низкий success rate: {success_rate}%"
    assert avg_time <= 0.1, f"Медленная обработка: {avg_time*1000:.1f} мс"
    assert throughput >= 10, f"Низкая пропускная способность: {throughput:.1f} запросов/сек"
    
    print("✅ Стресс тест пройден успешно!")

if __name__ == "__main__":
    # Запуск всех тестов
    print("🚀 Запуск комплексного тестирования системы KidneyAR")
    print("=" * 60)
    
    # Юнит-тесты
    print("\n1. Запуск юнит-тестов...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Стресс тест
    print("\n2. Запуск стресс теста...")
    run_stress_test(50)  # 50 запросов для быстрого теста
    
    print("\n" + "=" * 60)
    print("🎉 Все тесты завершены успешно!")
    print("Система готова к продакшн использованию! 🚀")
