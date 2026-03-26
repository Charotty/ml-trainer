#!/usr/bin/env python3
"""
Модуль для валидации preprocessing данных
Проверка качества данных на всех этапах обработки
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import warnings


class ValidationLevel(Enum):
    """Уровни валидации"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Результат валидации"""
    level: ValidationLevel
    message: str
    feature_name: Optional[str] = None
    value: Optional[Any] = None
    expected_range: Optional[Tuple[float, float]] = None
    suggestion: Optional[str] = None


class DataValidator:
    """
    Класс для валидации данных на всех этапах preprocessing
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Инициализация валидатора
        
        Args:
            strict_mode: Строгий режим - ошибки прерывают выполнение
        """
        self.strict_mode = strict_mode
        self.validation_results: List[ValidationResult] = []
        self.medical_ranges = self._get_medical_ranges()
    
    def _get_medical_ranges(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Получить медицинские диапазоны для признаков"""
        return {
            'demographics': {
                'age': (18, 100),  # Возраст пациентов
                'bmi': (15, 50),   # ИМТ
                'weight_kg': (40, 200),  # Вес
                'height_m': (1.4, 2.2)  # Рост
            },
            'coordinates': {
                'x_range': (-300, 300),  # Координаты X в мм
                'y_range': (-200, 200),  # Координаты Y в мм
                'z_range': (-100, 400)   # Координаты Z в мм
            },
            'displacements': {
                'delta_range': (-150, 150),  # Смещения в мм
                'max_magnitude': 150  # Максимальная величина смещения
            },
            'anatomical': {
                'kidney_length': (80, 150),  # Длина почки в мм
                'kidney_volume': (100, 300),  # Объем почки в см³
                'body_fat_ratio': (0.1, 0.5)  # Процент жира
            }
        }
    
    def validate_raw_data(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Валидировать сырые данные
        
        Args:
            df: DataFrame с сырыми данными
            
        Returns:
            Список результатов валидации
        """
        results = []
        
        # Проверка размера датасета
        n_samples, n_features = df.shape
        if n_samples < 10:
            results.append(ValidationResult(
                level=ValidationLevel.CRITICAL,
                message=f"Слишком маленький датасет: {n_samples}样本. Минимум 10样本.",
                value=n_samples,
                suggestion="Соберите больше данных"
            ))
        elif n_samples < 50:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Маленький датасет: {n_samples}样本. Рекомендуется > 50样本.",
                value=n_samples,
                suggestion="Соберите больше данных для улучшения обобщения"
            ))
        
        # Проверка пропусков
        missing_stats = df.isnull().sum()
        high_missing_cols = missing_stats[missing_stats > len(df) * 0.5]
        
        if not high_missing_cols.empty:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Колонки с >50% пропусков: {list(high_missing_cols.index)}",
                suggestion="Удалите или заполните эти колонки"
            ))
        
        # Проверка дубликатов
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Обнаружены дубликаты: {n_duplicates} шт.",
                value=n_duplicates,
                suggestion="Удалите дубликаты"
            ))
        
        # Валидация конкретных признаков
        results.extend(self._validate_demographics(df))
        results.extend(self._validate_coordinates(df))
        results.extend(self._validate_medical_features(df))
        
        self.validation_results.extend(results)
        
        if self.strict_mode and any(r.level == ValidationLevel.CRITICAL for r in results):
            raise ValueError(f"Критические ошибки в данных: {[r.message for r in results if r.level == ValidationLevel.CRITICAL]}")
        
        return results
    
    def _validate_demographics(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Валидировать демографические данные"""
        results = []
        ranges = self.medical_ranges['demographics']
        
        # Возраст
        if 'age' in df.columns or 'age_x' in df.columns:
            age_col = 'age' if 'age' in df.columns else 'age_x'
            age_values = df[age_col].dropna()
            
            if not age_values.empty:
                out_of_range = age_values[(age_values < ranges['age'][0]) | (age_values > ranges['age'][1])]
                if not out_of_range.empty:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Возраст вне диапазона {ranges['age']}: {len(out_of_range)} значений",
                        feature_name=age_col,
                        value=out_of_range.tolist(),
                        expected_range=ranges['age']
                    ))
        
        # ИМТ
        if 'bmi' in df.columns or 'bmi_x' in df.columns:
            bmi_col = 'bmi' if 'bmi' in df.columns else 'bmi_x'
            bmi_values = df[bmi_col].dropna()
            
            if not bmi_values.empty:
                out_of_range = bmi_values[(bmi_values < ranges['bmi'][0]) | (bmi_values > ranges['bmi'][1])]
                if not out_of_range.empty:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"ИМТ вне диапазона {ranges['bmi']}: {len(out_of_range)} значений",
                        feature_name=bmi_col,
                        value=out_of_range.tolist(),
                        expected_range=ranges['bmi']
                    ))
        
        return results
    
    def _validate_coordinates(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Валидировать координаты"""
        results = []
        coord_ranges = self.medical_ranges['coordinates']
        
        # Поиск координатных колонок
        coord_patterns = ['_x_', '_y_', '_z_', 'X_', 'Y_', 'Z_']
        coord_cols = [col for col in df.columns if any(pattern in col for pattern in coord_patterns)]
        
        for col in coord_cols:
            if df[col].dtype in ['int64', 'float64']:
                values = df[col].dropna()
                
                if not values.empty:
                    # Проверка диапазона
                    min_val, max_val = values.min(), values.max()
                    
                    # Определяем ось
                    if any(pattern in col for pattern in ['_x_', 'X_']):
                        range_check = coord_ranges['x_range']
                        axis = 'X'
                    elif any(pattern in col for pattern in ['_y_', 'Y_']):
                        range_check = coord_ranges['y_range']
                        axis = 'Y'
                    else:
                        range_check = coord_ranges['z_range']
                        axis = 'Z'
                    
                    if min_val < range_check[0] or max_val > range_check[1]:
                        results.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"Координаты {axis} вне диапазона {range_check}: [{min_val:.1f}, {max_val:.1f}]",
                            feature_name=col,
                            value=(min_val, max_val),
                            expected_range=range_check,
                            suggestion="Проверьте систему координат"
                        ))
                    
                    # Проверка на аномальные значения
                    if values.std() > 200:  # Слишком большой разброс
                        results.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"Большой разброс координат {col}: std={values.std():.1f}",
                            feature_name=col,
                            value=values.std(),
                            suggestion="Проверьте на выбросы"
                        ))
        
        return results
    
    def _validate_medical_features(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Валидировать медицинские признаки"""
        results = []
        medical_ranges = self.medical_ranges['anatomical']
        
        # Длина почки
        kidney_length_cols = [col for col in df.columns if 'kidney_length' in col and 'mm' in col]
        for col in kidney_length_cols:
            values = df[col].dropna()
            if not values.empty:
                out_of_range = values[(values < medical_ranges['kidney_length'][0]) | 
                                   (values > medical_ranges['kidney_length'][1])]
                if not out_of_range.empty:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Длина почки вне диапазона {medical_ranges['kidney_length']}: {len(out_of_range)} значений",
                        feature_name=col,
                        value=out_of_range.tolist(),
                        expected_range=medical_ranges['kidney_length']
                    ))
        
        # Объем почки
        kidney_volume_cols = [col for col in df.columns if 'kidney_volume' in col and 'cm3' in col]
        for col in kidney_volume_cols:
            values = df[col].dropna()
            if not values.empty:
                out_of_range = values[(values < medical_ranges['kidney_volume'][0]) | 
                                   (values > medical_ranges['kidney_volume'][1])]
                if not out_of_range.empty:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Объем почки вне диапазона {medical_ranges['kidney_volume']}: {len(out_of_range)} значений",
                        feature_name=col,
                        value=out_of_range.tolist(),
                        expected_range=medical_ranges['kidney_volume']
                    ))
        
        return results
    
    def validate_processed_data(self, 
                             X: np.ndarray, 
                             y: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             target_names: Optional[List[str]] = None) -> List[ValidationResult]:
        """
        Валидировать обработанные данные для ML
        
        Args:
            X: Признаки
            y: Целевые переменные
            feature_names: Имена признаков
            target_names: Имена целевых переменных
            
        Returns:
            Список результатов валидации
        """
        results = []
        
        # Проверка размерностей
        if X.shape[0] != y.shape[0]:
            results.append(ValidationResult(
                level=ValidationLevel.CRITICAL,
                message=f"Несоответствие размерностей: X={X.shape}, y={y.shape}",
                suggestion="Проверьте обработку данных"
            ))
        
        # Проверка пропусков после обработки
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Обнаружены NaN в обработанных данных: {nan_count} значений",
                value=nan_count,
                suggestion="Проверьте imputer"
            ))
        
        if np.isnan(y).any():
            nan_count = np.isnan(y).sum()
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Обнаружены NaN в целевых переменных: {nan_count} значений",
                value=nan_count,
                suggestion="Проверьте обработку целевых переменных"
            ))
        
        # Проверка бесконечных значений
        if np.isinf(X).any():
            inf_count = np.isinf(X).sum()
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Обнаружены Inf в признаках: {inf_count} значений",
                value=inf_count,
                suggestion="Проверьте на выбросы"
            ))
        
        # Валидация целевых переменных (смещения)
        if target_names:
            delta_ranges = self.medical_ranges['displacements']
            
            for i, target in enumerate(target_names):
                if 'delta' in target.lower():
                    target_values = y[:, i]
                    out_of_range = np.abs(target_values[target_values > delta_ranges['delta_range'][1]]) + \
                                  np.abs(target_values[target_values < delta_ranges['delta_range'][0]])
                    
                    if len(out_of_range) > 0:
                        results.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"Смещение {target} вне диапазона {delta_ranges['delta_range']}: {len(out_of_range)} значений",
                            feature_name=target,
                            value=out_of_range.tolist(),
                            expected_range=delta_ranges['delta_range']
                        ))
        
        # Проверка мультиколлинеарности
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X.T)
            high_corr_pairs = []
            
            for i in range(min(10, corr_matrix.shape[0])):  # Ограничиваем проверку
                for j in range(i+1, min(10, corr_matrix.shape[1])):
                    if abs(corr_matrix[i, j]) > 0.95:
                        feature_i = feature_names[i] if feature_names else f"feature_{i}"
                        feature_j = feature_names[j] if feature_names else f"feature_{j}"
                        high_corr_pairs.append((feature_i, feature_j, corr_matrix[i, j]))
            
            if high_corr_pairs:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Высокая корреляция (>0.95): {len(high_corr_pairs)} пар признаков",
                    value=high_corr_pairs[:5],  # Первые 5 пар
                    suggestion="Рассмотрите удаление одного из признаков в каждой паре"
                ))
        
        self.validation_results.extend(results)
        return results
    
    def validate_scaling(self, 
                       X_original: np.ndarray,
                       X_scaled: np.ndarray,
                       scaler_name: str = "StandardScaler") -> List[ValidationResult]:
        """
        Валидировать результаты масштабирования
        
        Args:
            X_original: Оригинальные признаки
            X_scaled: Масштабированные признаки
            scaler_name: Имя скейлера
            
        Returns:
            Список результатов валидации
        """
        results = []
        
        # Проверка сохранения ранга
        if np.linalg.matrix_rank(X_original) != np.linalg.matrix_rank(X_scaled):
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Изменился ранг матрицы после {scaler_name}",
                suggestion="Проверьте параметры скейлера"
            ))
        
        # Проверка среднего значения (для StandardScaler)
        if scaler_name == "StandardScaler":
            scaled_means = np.mean(X_scaled, axis=0)
            
            # Средние должны быть близки к 0
            max_mean_deviation = np.max(np.abs(scaled_means))
            if max_mean_deviation > 1e-10:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Средние значения после StandardScaler не равны 0: max={max_mean_deviation:.2e}",
                    value=max_mean_deviation,
                    suggestion="Проверьте корректность работы StandardScaler"
                ))
            
            # Стандартные отклонения должны быть близки к 1
            scaled_stds = np.std(X_scaled, axis=0)
            max_std_deviation = np.max(np.abs(scaled_stds - 1.0))
            if max_std_deviation > 1e-10:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Стандартные отклонения после StandardScaler не равны 1: max deviation={max_std_deviation:.2e}",
                    value=max_std_deviation,
                    suggestion="Проверьте корректность работы StandardScaler"
                ))
        
        self.validation_results.extend(results)
        return results
    
    def validate_feature_consistency(self,
                                 train_features: List[str],
                                 test_features: List[str]) -> List[ValidationResult]:
        """
        Валидировать консистентность признаков между train и test
        
        Args:
            train_features: Признаки тренировочных данных
            test_features: Признаки тестовых данных
            
        Returns:
            Список результатов валидации
        """
        results = []
        
        train_set = set(train_features)
        test_set = set(test_features)
        
        # Признаки в train, но не в test
        missing_in_test = train_set - test_set
        if missing_in_test:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Признаки в train, но не в test: {list(missing_in_test)}",
                value=list(missing_in_test),
                suggestion="Добавьте недостающие признаки в test данные"
            ))
        
        # Признаки в test, но не в train
        extra_in_test = test_set - train_set
        if extra_in_test:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"Признаки в test, но не в train: {list(extra_in_test)}",
                value=list(extra_in_test),
                suggestion="Удалите лишние признаки или добавьте их в train данные"
            ))
        
        self.validation_results.extend(results)
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Получить сводку результатов валидации
        
        Returns:
            Сводка валидации
        """
        if not self.validation_results:
            return {"total": 0, "by_level": {}, "by_feature": {}}
        
        # Группировка по уровням
        by_level = {}
        for level in ValidationLevel:
            level_results = [r for r in self.validation_results if r.level == level]
            by_level[level.value] = {
                'count': len(level_results),
                'messages': [r.message for r in level_results]
            }
        
        # Группировка по признакам
        by_feature = {}
        for result in self.validation_results:
            if result.feature_name:
                if result.feature_name not in by_feature:
                    by_feature[result.feature_name] = []
                by_feature[result.feature_name].append({
                    'level': result.level.value,
                    'message': result.message
                })
        
        return {
            'total': len(self.validation_results),
            'by_level': by_level,
            'by_feature': by_feature,
            'has_critical': any(r.level == ValidationLevel.CRITICAL for r in self.validation_results),
            'has_errors': any(r.level == ValidationLevel.ERROR for r in self.validation_results)
        }
    
    def print_validation_report(self):
        """Вывести отчет валидации"""
        summary = self.get_validation_summary()
        
        print("=== ОТЧЕТ ВАЛИДАЦИИ ДАННЫХ ===")
        print(f"Всего проверок: {summary['total']}")
        
        for level, info in summary['by_level'].items():
            if info['count'] > 0:
                print(f"\n{level.upper()} ({info['count']}):")
                for msg in info['messages']:
                    print(f"  - {msg}")
        
        if summary['by_feature']:
            print(f"\nПО ПРИЗНАКАМ:")
            for feature, issues in summary['by_feature'].items():
                print(f"\n{feature}:")
                for issue in issues:
                    print(f"  [{issue['level'].upper()}] {issue['message']}")
    
    def clear_results(self):
        """Очистить результаты валидации"""
        self.validation_results = []


def create_data_validator(strict_mode: bool = False) -> DataValidator:
    """
    Создать валидатор данных с настройками по умолчанию
    
    Args:
        strict_mode: Строгий режим
        
    Returns:
        DataValidator
    """
    return DataValidator(strict_mode=strict_mode)


if __name__ == "__main__":
    # Тестирование валидатора
    print("=== Тестирование Data Validator ===")
    
    # Создание тестовых данных
    np.random.seed(42)
    n_samples = 100
    
    # Сырые данные
    raw_data = {
        'age_x': np.random.normal(50, 15, n_samples),
        'bmi_x': np.random.normal(25, 5, n_samples),
        'X_upper_supine': np.random.normal(-50, 30, n_samples),
        'Y_upper_supine': np.random.normal(-30, 20, n_samples),
        'Z_upper_supine': np.random.normal(200, 50, n_samples),
        'delta_X_upper': np.random.normal(5, 15, n_samples),
        'kidney_length_mm': np.random.normal(120, 20, n_samples)
    }
    
    # Добавление проблемных данных
    raw_data['age_x'][0] = 150  # Неверный возраст
    raw_data['X_upper_supine'][1] = 500  # Неверная координата
    raw_data['bmi_x'][2:5] = np.nan  # Пропуски
    
    df = pd.DataFrame(raw_data)
    
    # Валидация
    validator = create_data_validator(strict_mode=False)
    
    print("Валидация сырых данных:")
    raw_results = validator.validate_raw_data(df)
    print(f"Найдено проблем: {len(raw_results)}")
    
    # Валидация обработанных данных
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples, 3) * 10
    
    print("\nВалидация обработанных данных:")
    processed_results = validator.validate_processed_data(X, y)
    print(f"Найдено проблем: {len(processed_results)}")
    
    # Вывод полного отчета
    validator.print_validation_report()
