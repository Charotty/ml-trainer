import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import logging

# Добавляем src в Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from ar_system.kidney_ar_system import KidneyARSystem
from validation.data_validator import DataValidator, ClinicalMetrics, SystemLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_test():
    """Простой тест системы"""
    print("🚀 Простой тест системы KidneyAR")
    print("=" * 50)
    
    # 1. Инициализация системы
    print("\n1. Инициализация системы...")
    system = KidneyARSystem()
    validator = DataValidator()
    metrics = ClinicalMetrics()
    
    print("✅ Система инициализирована")
    
    # 2. Тестовые данные
    print("\n2. Подготовка тестовых данных...")
    patient_data = {
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
    
    print("✅ Данные подготовлены")
    
    # 3. Валидация данных
    print("\n3. Валидация данных...")
    validation = validator.validate_patient_data(patient_data)
    if validation['is_valid']:
        print("✅ Данные валидны")
    else:
        print(f"❌ Ошибки валидации: {validation['errors']}")
        return
    
    # 4. Предсказание
    print("\n4. Выполнение предсказания...")
    start_time = time.time()
    
    result = system.predict_kidney_displacement(
        patient_data, sensor_data, ar_system_data
    )
    
    processing_time = time.time() - start_time
    
    print(f"✅ Предсказание выполнено за {processing_time*1000:.1f} мс")
    
    # 5. Результаты
    print("\n5. Результаты предсказания:")
    print(f"   Успех: {result['success']}")
    print(f"   Уверенность: {result['confidence']:.3f}")
    
    if result['success']:
        left = result['left_kidney']
        right = result['right_kidney']
        
        print(f"   Левая почка:")
        print(f"     Центр: [{left['center'][0]:.1f}, {left['center'][1]:.1f}, {left['center'][2]:.1f}]")
        print(f"     Смещение: [{left['displacement'][0]:.2f}, {left['displacement'][1]:.2f}, {left['displacement'][2]:.2f}]")
        print(f"     Polygon точек: {len(left['polygon'])}")
        
        print(f"   Правая почка:")
        print(f"     Центр: [{right['center'][0]:.1f}, {right['center'][1]:.1f}, {right['center'][2]:.1f}]")
        print(f"     Смещение: [{right['displacement'][0]:.2f}, {right['displacement'][1]:.2f}, {right['displacement'][2]:.2f}]")
        print(f"     Polygon точек: {len(right['polygon'])}")
        
        # 6. Проверка требований
        print("\n6. Проверка требований:")
        
        # Производительность
        if processing_time <= 0.1:
            print("✅ Производительность: ≤ 100 мс")
        else:
            print(f"❌ Производительность: {processing_time*1000:.1f} мс > 100 мс")
        
        # Уверенность
        if result['confidence'] >= 0.5:
            print("✅ Уверенность: ≥ 0.5")
        else:
            print(f"❌ Уверенность: {result['confidence']:.3f} < 0.5")
        
        # Polygon точки
        if len(left['polygon']) >= 50 and len(right['polygon']) >= 50:
            print("✅ Polygon: ≥ 50 точек на почку")
        else:
            print(f"❌ Polygon: {len(left['polygon'])}/{len(right['polygon'])} точек")
        
        # 7. Метрики
        print("\n7. Клинические метрики:")
        
        # Добавляем предсказание в историю
        metrics.add_prediction(result)
        
        # Сводные метрики
        summary = metrics.get_summary_metrics()
        print(f"   Всего предсказаний: {summary['total_predictions']}")
        print(f"   Успешных: {summary['successful_predictions']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Средняя уверенность: {summary['average_confidence']:.3f}")
        
        print("\n" + "=" * 50)
        print("🎉 Простой тест завершен успешно!")
        print("Система готова к использованию! 🚀")
        
    else:
        print(f"❌ Ошибка предсказания: {result.get('error', 'Unknown error')}")
        if result.get('details'):
            print(f"   Детали: {result['details']}")

def stress_test(num_requests=20):
    """Упрощенный стресс тест"""
    print(f"\n🔥 Стресс тест: {num_requests} запросов")
    print("-" * 40)
    
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
    
    times = []
    successes = 0
    confidences = []
    
    for i in range(num_requests):
        start_time = time.time()
        
        result = system.predict_kidney_displacement(
            patient_data, sensor_data, ar_system_data
        )
        
        request_time = time.time() - start_time
        times.append(request_time)
        
        if result['success']:
            successes += 1
            confidences.append(result['confidence'])
        
        if (i + 1) % 5 == 0:
            print(f"  Обработано {i + 1}/{num_requests}")
    
    # Результаты
    avg_time = np.mean(times)
    success_rate = (successes / num_requests) * 100
    avg_confidence = np.mean(confidences) if confidences else 0
    throughput = num_requests / sum(times)
    
    print(f"\nРезультаты стресс теста:")
    print(f"  Успешных: {successes}/{num_requests} ({success_rate:.1f}%)")
    print(f"  Среднее время: {avg_time*1000:.1f} мс")
    print(f"  Пропускная способность: {throughput:.1f} запросов/сек")
    print(f"  Средняя уверенность: {avg_confidence:.3f}")
    
    # Проверка требований
    if success_rate >= 95 and avg_time <= 0.1:
        print("✅ Стресс тест пройден!")
    else:
        print("❌ Стресс тест не пройден")

if __name__ == "__main__":
    # Запуск простого теста
    simple_test()
    
    # Запуск стресс теста
    stress_test(20)
