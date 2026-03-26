#!/usr/bin/env python3
"""Комплексное тестирование модуля предсказания"""

import joblib
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple

class PredictionTester:
    """Класс для тестирования модуля предсказания"""
    
    def __init__(self, model_path: str = "models/adaptive_ensemble.pkl"):
        self.model_path = model_path
        self.model_data = None
        self.load_model()
    
    def load_model(self) -> bool:
        """Загрузка модели"""
        try:
            self.model_data = joblib.load(self.model_path)
            print('✅ Модель успешно загружена')
            return True
        except Exception as e:
            print(f'❌ Ошибка загрузки модели: {e}')
            return False
    
    def test_single_prediction(self) -> bool:
        """Тест одиночного предсказания"""
        try:
            # Тестовые данные (средние значения)
            test_features = self._create_test_features()
            
            # Предсказание
            start_time = time.time()
            predictions = self._predict(test_features)
            prediction_time = time.time() - start_time
            
            # Проверка результатов
            print(f'✅ Одиночное предсказание выполнено за {prediction_time*1000:.1f}ms')
            print(f'📊 Результаты: {predictions}')
            
            # Проверка адекватности
            if self._validate_predictions(predictions):
                print('✅ Предсказания адекватны')
                return True
            else:
                print('❌ Предсказания неадекватны')
                return False
                
        except Exception as e:
            print(f'❌ Ошибка одиночного предсказания: {e}')
            return False
    
    def test_batch_prediction(self, batch_size: int = 10) -> bool:
        """Тест пакетного предсказания"""
        try:
            print(f'🔄 Тестирование пакетного предсказания (batch_size={batch_size})')
            
            predictions_list = []
            times = []
            
            for i in range(batch_size):
                test_features = self._create_test_features(variation=i)
                
                start_time = time.time()
                predictions = self._predict(test_features)
                prediction_time = time.time() - start_time
                
                predictions_list.append(predictions)
                times.append(prediction_time)
            
            avg_time = np.mean(times) * 1000
            print(f'✅ Пакетное предсказание: среднее время {avg_time:.1f}ms')
            
            # Проверка консистентности
            if self._check_consistency(predictions_list):
                print('✅ Предсказания консистентны')
                return True
            else:
                print('❌ Предсказания неконсистентны')
                return False
                
        except Exception as e:
            print(f'❌ Ошибка пакетного предсказания: {e}')
            return False
    
    def test_edge_cases(self) -> bool:
        """Тест граничных случаев"""
        try:
            print('🧪 Тестирование граничных случаев')
            
            edge_cases = [
                ("Минимальные значения", self._create_minimal_features()),
                ("Максимальные значения", self._create_maximal_features()),
                ("Нулевые значения", self._create_zero_features()),
                ("Случайные значения", self._create_random_features())
            ]
            
            all_passed = True
            for case_name, features in edge_cases:
                try:
                    predictions = self._predict(features)
                    print(f'✅ {case_name}: {predictions}')
                    
                    if not self._validate_predictions(predictions):
                        print(f'❌ {case_name}: предсказания неадекватны')
                        all_passed = False
                        
                except Exception as e:
                    print(f'❌ {case_name}: ошибка {e}')
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            print(f'❌ Ошибка тестирования граничных случаев: {e}')
            return False
    
    def _create_test_features(self, variation: int = 0) -> Dict:
        """Создание тестовых признаков"""
        base_features = {
            'kidney_left_center_x_rel': 100.0 + variation,
            'kidney_left_center_y_rel': 150.0 + variation,
            'kidney_left_center_z_rel': -800.0 + variation,
            'kidney_left_center_x_norm': 0.5 + variation * 0.01,
            'kidney_left_center_y_norm': 0.6 + variation * 0.01,
            'kidney_left_center_z_norm': -0.4 + variation * 0.01,
            'kidney_right_center_x_rel': 120.0 + variation,
            'kidney_right_center_y_rel': 160.0 + variation,
            'kidney_right_center_z_rel': -820.0 + variation,
            'kidney_right_center_x_norm': 0.55 + variation * 0.01,
            'kidney_right_center_y_norm': 0.65 + variation * 0.01,
            'kidney_right_center_z_norm': -0.41 + variation * 0.01,
            'kidney_left_length_mm': 110.0 + variation,
            'kidney_left_volume_cm3': 150.0 + variation * 5,
            'kidney_right_length_mm': 115.0 + variation,
            'kidney_right_volume_cm3': 160.0 + variation * 5,
            'body_width_mm': 400.0 + variation * 10,
            'body_depth_mm': 300.0 + variation * 10,
            'body_area_mm2': 120000.0 + variation * 1000,
            'kidney_left_to_spine_distance': 50.0 + variation,
            'kidney_right_to_spine_distance': 55.0 + variation,
            'kidney_left_to_body_center_distance': 100.0 + variation,
            'kidney_right_to_body_center_distance': 105.0 + variation,
            'spine_center_x': 0.0,
            'spine_center_y': 0.0,
            'spine_center_z': 0.0,
            'body_com_x': 0.0,
            'body_com_y': 0.0,
            'body_com_z': 0.0,
            'patient_position_encoded': 1.0
        }
        return base_features
    
    def _create_minimal_features(self) -> Dict:
        """Минимальные значения признаков"""
        features = self._create_test_features()
        for key in features:
            if isinstance(features[key], (int, float)) and features[key] > 0:
                features[key] = features[key] * 0.1
        return features
    
    def _create_maximal_features(self) -> Dict:
        """Максимальные значения признаков"""
        features = self._create_test_features()
        for key in features:
            if isinstance(features[key], (int, float)) and features[key] > 0:
                features[key] = features[key] * 2.0
        return features
    
    def _create_zero_features(self) -> Dict:
        """Нулевые значения признаков"""
        features = self._create_test_features()
        for key in features:
            if isinstance(features[key], (int, float)):
                features[key] = 0.0
        return features
    
    def _create_random_features(self) -> Dict:
        """Случайные значения признаков"""
        features = self._create_test_features()
        for key in features:
            if isinstance(features[key], (int, float)):
                features[key] = np.random.uniform(0.1, 2.0) * features[key]
        return features
    
    def _predict(self, features: Dict) -> Dict:
        """Выполнение предсказания"""
        # Упорядочивание признаков
        feature_names = self.model_data['feature_names']
        X_test = np.array([[features[feature] for feature in feature_names]])
        
        # Масштабирование
        X_test_scaled = self.model_data['scaler'].transform(X_test)
        
        # Предсказание для каждого таргета
        predictions = {}
        for target_name, model in self.model_data['models'].items():
            pred = model.predict(X_test_scaled)[0]
            predictions[target_name] = pred
            
        return predictions
    
    def _validate_predictions(self, predictions: Dict) -> bool:
        """Валидация предсказаний"""
        values = list(predictions.values())
        
        # Проверка на NaN
        if any(np.isnan(v) for v in values):
            return False
        
        # Проверка на бесконечность
        if any(np.isinf(v) for v in values):
            return False
        
        # Проверка адекватности диапазона
        max_abs_pred = max(abs(v) for v in values)
        if max_abs_pred > 100:  # Слишком большие значения
            return False
        
        return True
    
    def _check_consistency(self, predictions_list: List[Dict]) -> bool:
        """Проверка консистентности предсказаний"""
        if len(predictions_list) < 2:
            return True
        
        # Проверка что предсказания не слишком сильно различаются
        all_values = []
        for predictions in predictions_list:
            all_values.extend(list(predictions.values()))
        
        std_dev = np.std(all_values)
        mean_val = np.mean(np.abs(all_values))
        
        # Коэффициент вариации
        cv = (std_dev / mean_val) if mean_val > 0 else 0
        
        return cv < 0.5  # Коэффициент вариации < 50%
    
    def run_all_tests(self) -> bool:
        """Запуск всех тестов"""
        print('🧪 Запуск комплексного тестирования модуля предсказания')
        print('=' * 60)
        
        tests = [
            ("Загрузка модели", self.load_model),
            ("Одиночное предсказание", self.test_single_prediction),
            ("Пакетное предсказание", self.test_batch_prediction),
            ("Граничные случаи", self.test_edge_cases)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f'\n🔍 {test_name}:')
            try:
                result = test_func()
                results.append((test_name, result))
                status = '✅ ПРОЙДЕН' if result else '❌ ПРОВАЛЕН'
                print(f'   Результат: {status}')
            except Exception as e:
                print(f'   Ошибка: {e}')
                results.append((test_name, False))
        
        # Итоги
        print('\n' + '=' * 60)
        print('📊 ИТОГИ ТЕСТИРОВАНИЯ:')
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = '✅' if result else '❌'
            print(f'   {status} {test_name}')
        
        print(f'\n🎯 Пройдено тестов: {passed}/{total}')
        print(f'📈 Успешность: {passed/total*100:.1f}%')
        
        if passed == total:
            print('\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Модуль предсказания готов к продакшену.')
            return True
        else:
            print('\n⚠️  Некоторые тесты провалены. Нужна отладка.')
            return False

def main():
    """Основная функция тестирования"""
    tester = PredictionTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
