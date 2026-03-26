#!/usr/bin/env python3
"""Тест предсказания модели на нескольких реальных примерах"""

import joblib
import numpy as np
import pandas as pd
import sys
import os

# Добавляем путь к корневой директории для импорта наших модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'phase1'))
from adaptive_ensemble import AdaptiveEnsembleTrainer

def test_multiple_predictions():
    """Тест предсказания на нескольких реальных примерах"""
    try:
        print("🔧 Загрузка реальных данных для тестирования...")
        
        # Загружаем реальные данные
        train_df = pd.read_csv('data/processed/train.csv')
        
        # Берем первые 3 строки для теста
        test_samples = train_df.head(3)
        
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
        
        # Фильтруем только нужные признаки
        test_data = test_samples[[col for col in required_base_features if col in test_samples.columns]]
        
        print(f"📋 Тестирование на {len(test_data)} реальных примерах")
        
        # Создаем признаки
        trainer = AdaptiveEnsembleTrainer()
        test_enhanced = trainer._create_engineered_features(test_data.copy())
        test_enhanced = trainer._create_cross_features(test_enhanced)
        
        # Загрузка модели
        model_data = joblib.load('models/adaptive_ensemble.pkl')
        feature_names = model_data['feature_names']
        
        # Проверяем наличие всех необходимых признаков
        missing_features = [f for f in feature_names if f not in test_enhanced.columns]
        if missing_features:
            print(f"❌ Отсутствующие признаки: {missing_features}")
            return False, None
        
        # Подготовка данных
        X_test = test_enhanced[feature_names].values
        X_test_scaled = model_data['scaler'].transform(X_test)
        
        print("✅ Данные подготовлены, выполняем предсказания...")
        
        # Предсказания для каждого примера
        all_predictions = {}
        for i, row in enumerate(test_enhanced.itertuples()):
            print(f"\n📊 Пример {i+1}:")
            print(f"  Базовые данные: kidney_left_volume={row.kidney_left_volume_cm3:.1f}, kidney_right_volume={row.kidney_right_volume_cm3:.1f}")
            print(f"  Body: width={row.body_width_mm:.1f}, depth={row.body_depth_mm:.1f}")
            
            predictions = {}
            for target_name, model in model_data['models'].items():
                pred = model.predict(X_test_scaled[i:i+1])[0]
                predictions[target_name] = pred
            
            all_predictions[f'sample_{i+1}'] = predictions
            
            # Вывод предсказаний
            for target, pred in predictions.items():
                print(f"    {target}: {pred:.3f} mm")
            
            # Анализ предсказаний для примера
            preds = list(predictions.values())
            max_pred = max(abs(p) for p in preds)
            avg_pred = np.mean(np.abs(preds))
            print(f"    Макс: {max_pred:.3f}mm, Среднее: {avg_pred:.3f}mm")
        
        # Общий анализ
        print(f"\n📊 Общий анализ по всем {len(test_data)} примерам:")
        all_preds = []
        for sample_preds in all_predictions.values():
            all_preds.extend(sample_preds.values())
        
        max_overall = max(abs(p) for p in all_preds)
        avg_overall = np.mean(np.abs(all_preds))
        std_overall = np.std(all_preds)
        
        print(f"  Общий максимум: {max_overall:.3f} mm")
        print(f"  Общее среднее: {avg_overall:.3f} mm")
        print(f"  Стандартное отклонение: {std_overall:.3f} mm")
        
        # Сравнение с ожидаемыми значениями из исследований
        research_mae_range = (1.5, 3.0)
        
        if research_mae_range[0] <= avg_overall <= research_mae_range[1]:
            print(f"✅ Результаты ({avg_overall:.3f}mm) соответствуют исследованиям 2024-2025")
        elif avg_overall < research_mae_range[0]:
            print(f"🎉 ОТЛИЧНО! Результаты ({avg_overall:.3f}mm) лучше исследований!")
        else:
            print(f"⚠️  Результаты ({avg_overall:.3f}mm) хуже исследований")
        
        # Проверка на адекватность
        if max_overall > 30:
            print(f"❌ Предупреждение: некоторые предсказания слишком высокие (макс: {max_overall:.1f}mm)")
        else:
            print(f"✅ Все предсказания выглядят адекватными")
        
        return True, all_predictions
        
    except Exception as e:
        print(f'❌ Ошибка тестирования: {e}')
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, predictions = test_multiple_predictions()
    if success:
        print('\n🎉 Множественный тест предсказания пройден!')
    else:
        print('\n💥 Множественный тест предсказания провален!')
