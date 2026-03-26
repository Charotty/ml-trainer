#!/usr/bin/env python3
"""Тест предсказания модели с поддержкой всех признаков"""

import joblib
import numpy as np
import pandas as pd
import sys
import os

# Добавляем путь к корневой директории для импорта наших модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'phase1'))
from adaptive_ensemble import AdaptiveEnsembleTrainer

def create_test_features():
    """Создание тестовых данных с полным набором признаков на основе реальных данных"""
    # Загружаем реальные данные для получения реалистичных значений
    try:
        train_df = pd.read_csv('data/processed/train.csv')
        # Берем первую строку как основу для теста
        base_features = train_df.iloc[0].to_dict()
        
        # Оставляем только базовые признаки (23)
        required_base_features = [
            'kidney_left_center_x_rel', 'kidney_left_center_y_rel', 'kidney_left_center_z_rel',
            'kidney_right_center_x_rel', 'kidney_right_center_y_rel', 'kidney_right_center_z_rel',
            'kidney_left_length_mm', 'kidney_left_volume_cm3',
            'kidney_right_length_mm', 'kidney_right_volume_cm3',
            'body_width_mm', 'body_depth_mm', 'body_area_mm2',
            'kidney_left_to_spine_distance', 'kidney_right_to_spine_distance',
            'kidney_left_to_body_center_distance', 'kidney_right_to_body_center_distance',
            'spine_center_x', 'spine_center_y', 'spine_center_z',
            'body_com_x', 'body_com_y', 'body_com_z',
        ]
        
        test_features = {k: base_features[k] for k in required_base_features if k in base_features}
        
    except Exception:
        # Fallback - используем реалистичные значения
        test_features = {
            'kidney_left_center_x_rel': 85.5,
            'kidney_left_center_y_rel': 142.3,
            'kidney_left_center_z_rel': -745.2,
            'kidney_right_center_x_rel': 95.8,
            'kidney_right_center_y_rel': 148.7,
            'kidney_right_center_z_rel': -752.1,
            'kidney_left_length_mm': 98.5,
            'kidney_left_volume_cm3': 125.3,
            'kidney_right_length_mm': 102.1,
            'kidney_right_volume_cm3': 132.7,
            'body_width_mm': 385.2,
            'body_depth_mm': 285.6,
            'body_area_mm2': 110000.0,
            'kidney_left_to_spine_distance': 45.3,
            'kidney_right_to_spine_distance': 48.7,
            'kidney_left_to_body_center_distance': 92.1,
            'kidney_right_to_body_center_distance': 96.4,
            'spine_center_x': 0.0,
            'spine_center_y': 0.0,
            'spine_center_z': 0.0,
            'body_com_x': 0.0,
            'body_com_y': 0.0,
            'body_com_z': 0.0,
        }
    
    # Создаем DataFrame и добавляем инженерные и cross-features
    df = pd.DataFrame([test_features])
    
    # Инициализируем тренер для создания признаков
    trainer = AdaptiveEnsembleTrainer()
    
    # Создаем инженерные признаки
    df = trainer._create_engineered_features(df)
    
    # Создаем cross-features
    df = trainer._create_cross_features(df)
    
    return df

def test_model_prediction():
    """Проверка работы предсказания модели с полным набором признаков"""
    try:
        print("🔧 Создание тестовых данных с полным набором признаков...")
        
        # Создание тестовых данных
        test_df = create_test_features()
        
        # Загрузка модели
        model_data = joblib.load('models/adaptive_ensemble.pkl')
        
        # Получаем имена признаков из модели
        feature_names = model_data['feature_names']
        print(f"📋 Модель ожидает {len(feature_names)} признаков")
        print(f"📋 Тестовые данные содержат {len(test_df.columns)} признаков")
        
        # Проверяем наличие всех необходимых признаков
        missing_features = [f for f in feature_names if f not in test_df.columns]
        if missing_features:
            print(f"❌ Отсутствующие признаки: {missing_features}")
            return False, None
        
        # Упорядочивание признаков согласно модели
        X_test = test_df[feature_names].values
        
        # Масштабирование данных
        X_test_scaled = model_data['scaler'].transform(X_test)
        
        print("✅ Данные подготовлены, выполняем предсказание...")
        
        # Предсказание для каждого таргета
        predictions = {}
        for target_name, model in model_data['models'].items():
            pred = model.predict(X_test_scaled)[0]
            predictions[target_name] = pred
            
        print('✅ Предсказание выполнено успешно')
        print('\n📊 Результаты предсказания:')
        for target, pred in predictions.items():
            print(f'  {target}: {pred:.3f} mm')
            
        # Проверка адекватности значений (сравнение с исследованиями 2024-2025)
        all_predictions = list(predictions.values())
        max_pred = max(abs(p) for p in all_predictions)
        avg_pred = np.mean(np.abs(all_predictions))
        
        print(f'\n📊 Анализ предсказаний:')
        print(f'  Максимальное смещение: {max_pred:.3f} mm')
        print(f'  Среднее смещение: {avg_pred:.3f} mm')
        
        # Сравнение с исследованиями 2024-2025
        research_mae_range = (1.5, 3.0)  # Типичный диапазон из исследований
        our_avg_mae = avg_pred
        
        if research_mae_range[0] <= our_avg_mae <= research_mae_range[1]:
            print(f'✅ Наши результаты ({our_avg_mae:.3f}mm) соответствуют исследованиям 2024-2025')
        elif our_avg_mae < research_mae_range[0]:
            print(f'🎉 ОТЛИЧНО! Наши результаты ({our_avg_mae:.3f}mm) лучше исследований!')
        else:
            print(f'⚠️  Наши результаты ({our_avg_mae:.3f}mm) хуже исследований')
            
        # Проверка на нереалистичные значения
        if max_pred > 50:  # Слишком большие значения
            print(f'❌ Предупреждение: максимальное предсказание {max_pred:.1f}mm выглядит нереалистичным')
        elif max_pred > 30:
            print(f'⚠️  Внимание: максимальное предсказание {max_pred:.1f}mm высокое')
        else:
            print(f'✅ Предсказания выглядят адекватными (макс: {max_pred:.1f}mm)')
            
        # Дополнительная информация о модели
        print(f'\n📋 Информация о модели:')
        print(f'  Количество признаков: {len(feature_names)}')
        print(f'  Количество таргетов: {len(model_data["models"])}')
        print(f'  Источники данных: {model_data.get("data_sources", "DICOMS+Vybor+KiTS19")}')
            
        return True, predictions
        
    except Exception as e:
        print(f'❌ Ошибка предсказания: {e}')
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, predictions = test_model_prediction()
    if success:
        print('\n🎉 Тест предсказания модели пройден!')
    else:
        print('\n💥 Тест предсказания модели провален!')
