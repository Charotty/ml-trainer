#!/usr/bin/env python3
"""Тест загрузки модели"""

import joblib
import sys

def test_model_loading():
    """Проверка загрузки обученной модели"""
    try:
        # Загрузка модели
        model_data = joblib.load('models/adaptive_ensemble.pkl')
        
        print('✅ Модель загружена успешно')
        print(f'📊 Моделей в ансамбле: {len(model_data["models"])}')
        print(f'🎯 Признаков: {len(model_data["feature_names"])}')
        print(f'🎯 Таргетов: {len(model_data["target_names"])}')
        
        # Проверка наличия необходимых компонентов
        required_keys = ['models', 'scaler', 'imputer', 'feature_names', 'target_names']
        for key in required_keys:
            if key in model_data:
                print(f'✅ {key}: присутствует')
            else:
                print(f'❌ {key}: отсутствует')
                
        return True
        
    except Exception as e:
        print(f'❌ Ошибка загрузки модели: {e}')
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print('\n🎉 Тест загрузки модели пройден!')
    else:
        print('\n💥 Тест загрузки модели провален!')
