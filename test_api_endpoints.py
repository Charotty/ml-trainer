#!/usr/bin/env python3
"""Тестирование API эндпоинтов"""

import requests
import json
import time

def test_api_endpoints():
    """Проверка основных эндпоинтов API"""
    base_url = "http://localhost:8000"
    
    print('🧪 Тестирование API эндпоинтов...')
    
    # Тест 1: Корневой эндпоинт
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print('✅ Корневой эндпоинт (/): работает')
            print(f'   Ответ: {response.json()}')
        else:
            print(f'❌ Корневой эндпоинт: статус {response.status_code}')
            return False
    except Exception as e:
        print(f'❌ Корневой эндпоинт: {e}')
        return False
    
    # Тест 2: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print('✅ Health check (/health): работает')
            print(f'   Ответ: {response.json()}')
        else:
            print(f'❌ Health check: статус {response.status_code}')
            return False
    except Exception as e:
        print(f'❌ Health check: {e}')
        return False
    
    # Тест 3: Документация API
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print('✅ Документация (/docs): доступна')
        else:
            print(f'❌ Документация: статус {response.status_code}')
            return False
    except Exception as e:
        print(f'❌ Документация: {e}')
        return False
    
    # Тест 4: Prediciton endpoint (с тестовыми данными)
    try:
        test_data = {
            "age": 45.0,
            "bmi": 25.0,
            "sex_encoded": 1.0,
            "kidney_left_center_x_rel": 100.0,
            "kidney_left_center_y_rel": 150.0,
            "kidney_left_center_z_rel": -800.0,
            "kidney_right_center_x_rel": 120.0,
            "kidney_right_center_y_rel": 160.0,
            "kidney_right_center_z_rel": -820.0,
            "kidney_left_volume_cm3": 150.0,
            "kidney_right_volume_cm3": 160.0,
            "body_width_mm": 400.0,
            "body_depth_mm": 300.0,
            "body_area_mm2": 120000.0
        }
        
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=10)
        if response.status_code == 200:
            print('✅ Предсказание (/predict): работает')
            result = response.json()
            print(f'   Предсказания: {result.get("predictions", {})}')
            print(f'   Уверенность: {result.get("confidence", "N/A")}')
        else:
            print(f'❌ Предсказание: статус {response.status_code}')
            print(f'   Ошибка: {response.text}')
            return False
    except Exception as e:
        print(f'❌ Предсказание: {e}')
        return False
    
    print('\n🎉 Все API эндпоинты работают корректно!')
    return True

if __name__ == "__main__":
    # Даем серверу время на запуск
    time.sleep(2)
    
    success = test_api_endpoints()
    if success:
        print('\n✅ API сервер полностью функционален!')
        print('🌐 Доступен по адресу: http://localhost:8000')
        print('📚 Документация: http://localhost:8000/docs')
    else:
        print('\n❌ Проблемы с API сервером!')
