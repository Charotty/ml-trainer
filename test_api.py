#!/usr/bin/env python3
"""
Тестирование API предсказания смещения почек
"""

import requests
import json
import time

def test_api():
    """Полное тестирование API"""
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Тестирование API предсказания смещения почек")
    print("=" * 50)
    
    # 1. Тест здоровья
    print("\n1. Проверка здоровья сервера...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Сервер здоров: {health_data['status']}")
            print(f"   Версия модели: {health_data['model_version']}")
            print(f"   Признаков: {health_data['features_count']}")
            print(f"   Целей: {health_data['targets_count']}")
        else:
            print(f"❌ Ошибка здоровья: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Не удалось подключиться к серверу: {e}")
        print("💡 Убедитесь, что API сервер запущен: python src/api/kidney_displacement_api.py")
        return False
    
    # 2. Тест информации о модели
    print("\n2. Получение информации о модели...")
    try:
        response = requests.get(f"{base_url}/model_info")
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Модель: {model_info['model_info']['name']}")
            print(f"   Производительность: MAE={model_info['model_info']['performance']['average_mae_mm']} мм")
            print(f"   Точность <5мм: {model_info['model_info']['performance']['accuracy_5mm']}%")
        else:
            print(f"❌ Ошибка получения информации: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # 3. Тест предсказания
    print("\n3. Тест предсказания смещения...")
    test_data = {
        "patient_data": {
            "kidney_left_center_x_rel": 85.5,
            "kidney_left_center_y_rel": 142.3,
            "kidney_left_center_z_rel": -745.2,
            "kidney_right_center_x_rel": 95.8,
            "kidney_right_center_y_rel": 148.7,
            "kidney_right_center_z_rel": -752.1,
            "kidney_left_length_mm": 98.5,
            "kidney_left_volume_cm3": 125.3,
            "kidney_right_length_mm": 102.1,
            "kidney_right_volume_cm3": 132.7,
            "body_width_mm": 385.2,
            "body_depth_mm": 285.6,
            "body_area_mm2": 110000.0,
            "kidney_left_to_spine_distance": 45.3,
            "kidney_right_to_spine_distance": 48.7,
            "kidney_left_to_body_center_distance": 92.1,
            "kidney_right_to_body_center_distance": 96.4,
            "spine_center_x": 0.0,
            "spine_center_y": 0.0,
            "spine_center_z": 0.0,
            "body_com_x": 0.0,
            "body_com_y": 0.0,
            "body_com_z": 0.0
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/predict", json=test_data)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Предсказание выполнено за {response_time:.3f} сек")
            print("   Результаты:")
            for target, value in result['predictions'].items():
                print(f"     {target}: {value:.3f} мм")
            
            # Проверка адекватности
            max_pred = max(abs(v) for v in result['predictions'].values())
            if max_pred > 30:
                print(f"⚠️  Предупреждение: высокое предсказание ({max_pred:.1f} мм)")
            else:
                print(f"✅ Предсказания адекватны (макс: {max_pred:.1f} мм)")
                
        else:
            print(f"❌ Ошибка предсказания: {response.status_code}")
            print(f"   Ответ: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False
    
    # 4. Тест пакетного предсказания
    print("\n4. Тест пакетного предсказания...")
    batch_data = {
        "patients": [
            {
                "patient_id": "test_patient_1",
                "patient_data": test_data["patient_data"]
            },
            {
                "patient_id": "test_patient_2",
                "patient_data": {
                    **test_data["patient_data"],
                    "kidney_left_volume_cm3": 130.0,
                    "kidney_right_volume_cm3": 135.0,
                    "body_width_mm": 390.0
                }
            }
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/predict_batch", json=batch_data)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Пакетное предсказание выполнено за {response_time:.3f} сек")
            print(f"   Успешных предсказаний: {result['metadata']['successful_predictions']}/{result['metadata']['total_patients']}")
            
            for patient_result in result['results']:
                if 'error' in patient_result:
                    print(f"   ❌ {patient_result['patient_id']}: {patient_result['error']}")
                else:
                    print(f"   ✅ {patient_result['patient_id']}: предсказано успешно")
        else:
            print(f"❌ Ошибка пакетного предсказания: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # 5. Тест производительности
    print("\n5. Тест производительности...")
    try:
        times = []
        for i in range(5):
            start_time = time.time()
            response = requests.post(f"{base_url}/predict", json=test_data)
            response_time = time.time() - start_time
            times.append(response_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"✅ Производительность (5 запросов):")
        print(f"   Среднее время: {avg_time:.3f} сек")
        print(f"   Минимальное: {min_time:.3f} сек")
        print(f"   Максимальное: {max_time:.3f} сек")
        
        if avg_time < 0.5:
            print("🚀 Отличная производительность!")
        elif avg_time < 1.0:
            print("✅ Хорошая производительность")
        else:
            print("⚠️  Производительность можно улучшить")
            
    except Exception as e:
        print(f"❌ Ошибка теста производительности: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Все тесты API завершены!")
    return True

if __name__ == "__main__":
    success = test_api()
    if success:
        print("\n✅ API готов к использованию!")
    else:
        print("\n❌ API требует исправлений")
