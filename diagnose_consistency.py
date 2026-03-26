#!/usr/bin/env python3
"""Диагностика консистентности предсказаний"""

import joblib
import numpy as np

def diagnose_consistency():
    """Диагностика проблемы консистентности"""
    
    # Загрузка модели
    model_data = joblib.load('models/adaptive_ensemble.pkl')
    
    # Создание идентичных тестовых данных
    base_features = {
        'kidney_left_center_x_rel': 100.0,
        'kidney_left_center_y_rel': 150.0,
        'kidney_left_center_z_rel': -800.0,
        'kidney_left_center_x_norm': 0.5,
        'kidney_left_center_y_norm': 0.6,
        'kidney_left_center_z_norm': -0.4,
        'kidney_right_center_x_rel': 120.0,
        'kidney_right_center_y_rel': 160.0,
        'kidney_right_center_z_rel': -820.0,
        'kidney_right_center_x_norm': 0.55,
        'kidney_right_center_y_norm': 0.65,
        'kidney_right_center_z_norm': -0.41,
        'kidney_left_length_mm': 110.0,
        'kidney_left_volume_cm3': 150.0,
        'kidney_right_length_mm': 115.0,
        'kidney_right_volume_cm3': 160.0,
        'body_width_mm': 400.0,
        'body_depth_mm': 300.0,
        'body_area_mm2': 120000.0,
        'kidney_left_to_spine_distance': 50.0,
        'kidney_right_to_spine_distance': 55.0,
        'kidney_left_to_body_center_distance': 100.0,
        'kidney_right_to_body_center_distance': 105.0,
        'spine_center_x': 0.0,
        'spine_center_y': 0.0,
        'spine_center_z': 0.0,
        'body_com_x': 0.0,
        'body_com_y': 0.0,
        'body_com_z': 0.0,
        'patient_position_encoded': 1.0
    }
    
    print('🔍 Диагностика консистентности предсказаний')
    print('=' * 60)
    
    # Тест 1: Идентичные входные данные
    print('\n📊 Тест 1: Идентичные входные данные')
    predictions_identical = []
    
    for i in range(5):
        # Упорядочивание признаков
        feature_names = model_data['feature_names']
        X_test = np.array([[base_features[feature] for feature in feature_names]])
        X_test_scaled = model_data['scaler'].transform(X_test)
        
        # Предсказание
        predictions = {}
        for target_name, model in model_data['models'].items():
            pred = model.predict(X_test_scaled)[0]
            predictions[target_name] = pred
        
        predictions_identical.append(predictions)
        print(f'  Запуск {i+1}: {predictions}')
    
    # Проверка идентичности
    all_values = []
    for pred in predictions_identical:
        all_values.extend(list(pred.values()))
    
    std_identical = np.std(all_values)
    print(f'  Стандартное отклонение: {std_identical:.6f}')
    
    if std_identical < 1e-10:
        print('  ✅ Предсказания идентичны (детерминированность)')
    else:
        print('  ❌ Предсказания различаются (проблема с детерминированностью)')
    
    # Тест 2: Проверка random_state в моделях
    print('\n🎲 Тест 2: Проверка random_state в моделях')
    for target_name, model in model_data['models'].items():
        if hasattr(model, 'random_state'):
            print(f'  {target_name}: random_state = {getattr(model, "random_state", "не установлен")}')
        else:
            print(f'  {target_name}: random_state не найден')
    
    # Тест 3: Проверка scaler
    print('\n📏 Тест 3: Проверка scaler')
    scaler = model_data['scaler']
    print(f'  Тип scaler: {type(scaler).__name__}')
    print(f'  Параметры scaler: {scaler.get_params()}')
    
    # Тест 4: Проверка feature order
    print('\n📋 Тест 4: Проверка порядка признаков')
    print(f'  Количество признаков: {len(model_data["feature_names"])}')
    print(f'  Первые 5 признаков: {model_data["feature_names"][:5]}')
    
    return std_identical < 1e-10

if __name__ == "__main__":
    is_consistent = diagnose_consistency()
    
    if is_consistent:
        print('\n🎉 Модель детерминирована!')
    else:
        print('\n⚠️  Модель недетерминирована - нужно исправить!')
