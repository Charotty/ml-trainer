#!/usr/bin/env python3
"""
Модуль для работы с непарными данными
Использование unpaired данных для регуляризации и улучшения обобщения
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import warnings


class UnpairedDataType(Enum):
    """Типы непарных данных"""
    SUPINE_ONLY = "supine_only"
    LATERAL_ONLY = "lateral_only"
    MIXED = "mixed"


@dataclass
class UnpairedDataInfo:
    """Информация о непарных данных"""
    data_type: UnpairedDataType
    n_samples: int
    feature_names: List[str]
    source_files: List[str]
    quality_score: float
    preprocessing_applied: List[str]


class UnpairedDataManager:
    """
    Класс для управления непарными данными
    """
    
    def __init__(self):
        """Инициализация менеджера непарных данных"""
        self.supine_data: Optional[pd.DataFrame] = None
        self.lateral_data: Optional[pd.DataFrame] = None
        self.supine_features: Optional[np.ndarray] = None
        self.lateral_features: Optional[np.ndarray] = None
        self.feature_distributions: Dict[str, Dict[str, Any]] = {}
        self.scaler: Optional[StandardScaler] = None
        self.pca_models: Dict[str, PCA] = {}
        self.gmm_models: Dict[str, GaussianMixture] = {}
        
    def load_unpaired_data(self, 
                          supine_path: Optional[str] = None,
                          lateral_path: Optional[str] = None,
                          mixed_path: Optional[str] = None) -> Dict[str, UnpairedDataInfo]:
        """
        Загрузить непарные данные
        
        Args:
            supine_path: Путь к данным в положении supine
            lateral_path: Путь к данным в положении lateral
            mixed_path: Путь к смешанным данным
            
        Returns:
            Словарь с информацией о загруженных данных
        """
        data_info = {}
        
        if supine_path:
            try:
                self.supine_data = pd.read_csv(supine_path)
                data_info['supine'] = UnpairedDataInfo(
                    data_type=UnpairedDataType.SUPINE_ONLY,
                    n_samples=len(self.supine_data),
                    feature_names=list(self.supine_data.columns),
                    source_files=[supine_path],
                    quality_score=self._assess_data_quality(self.supine_data),
                    preprocessing_applied=[]
                )
                print(f"Загружено {len(self.supine_data)} записей supine данных")
            except Exception as e:
                print(f"Ошибка загрузки supine данных: {e}")
        
        if lateral_path:
            try:
                self.lateral_data = pd.read_csv(lateral_path)
                data_info['lateral'] = UnpairedDataInfo(
                    data_type=UnpairedDataType.LATERAL_ONLY,
                    n_samples=len(self.lateral_data),
                    feature_names=list(self.lateral_data.columns),
                    source_files=[lateral_path],
                    quality_score=self._assess_data_quality(self.lateral_data),
                    preprocessing_applied=[]
                )
                print(f"Загружено {len(self.lateral_data)} записей lateral данных")
            except Exception as e:
                print(f"Ошибка загрузки lateral данных: {e}")
        
        if mixed_path:
            try:
                mixed_data = pd.read_csv(mixed_path)
                # Разделение на supine и lateral
                if 'position' in mixed_data.columns:
                    self.supine_data = mixed_data[mixed_data['position'] == 'supine']
                    self.lateral_data = mixed_data[mixed_data['position'] == 'lateral']
                else:
                    # Попытка определить по именам колонок
                    supine_cols = [col for col in mixed_data.columns if 'supine' in col.lower()]
                    lateral_cols = [col for col in mixed_data.columns if 'lateral' in col.lower()]
                    
                    if supine_cols:
                        self.supine_data = mixed_data[supine_cols + [col for col in mixed_data.columns if col not in lateral_cols]]
                    if lateral_cols:
                        self.lateral_data = mixed_data[lateral_cols + [col for col in mixed_data.columns if col not in supine_cols]]
                
                data_info['mixed'] = UnpairedDataInfo(
                    data_type=UnpairedDataType.MIXED,
                    n_samples=len(mixed_data),
                    feature_names=list(mixed_data.columns),
                    source_files=[mixed_path],
                    quality_score=self._assess_data_quality(mixed_data),
                    preprocessing_applied=[]
                )
                print(f"Загружены смешанные данные: {len(self.supine_data) if self.supine_data is not None else 0} supine, {len(self.lateral_data) if self.lateral_data is not None else 0} lateral")
            except Exception as e:
                print(f"Ошибка загрузки смешанных данных: {e}")
        
        return data_info
    
    def _assess_data_quality(self, df: pd.DataFrame) -> float:
        """
        Оценить качество данных
        
        Args:
            df: DataFrame для оценки
            
        Returns:
            Оценка качества (0-1)
        """
        score = 1.0
        
        # Штраф за пропуски
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score -= missing_ratio * 0.3
        
        # Штраф за дубликаты
        duplicate_ratio = df.duplicated().sum() / len(df)
        score -= duplicate_ratio * 0.2
        
        # Штраф за выбросы (упрощенная проверка)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outlier_count = 0
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_count += outliers
            
            outlier_ratio = outlier_count / (len(df) * len(numeric_cols))
            score -= outlier_ratio * 0.2
        
        # Бонус за размер датасета
        if len(df) > 100:
            score += 0.1
        elif len(df) > 50:
            score += 0.05
        
        return np.clip(score, 0.0, 1.0)
    
    def preprocess_unpaired_data(self, 
                              feature_columns: Optional[List[str]] = None,
                              scale_features: bool = True) -> Dict[str, Any]:
        """
        Предобработать непарные данные
        
        Args:
            feature_columns: Список признаков для использования
            scale_features: Нормализовать признаки
            
        Returns:
            Результаты предобработки
        """
        results = {}
        
        # Определение общих признаков
        if feature_columns is None:
            if self.supine_data is not None and self.lateral_data is not None:
                supine_cols = set(self.supine_data.columns)
                lateral_cols = set(self.lateral_data.columns)
                common_cols = list(supine_cols.intersection(lateral_cols))
                feature_columns = common_cols
            elif self.supine_data is not None:
                feature_columns = list(self.supine_data.columns)
            elif self.lateral_data is not None:
                feature_columns = list(self.lateral_data.columns)
            else:
                raise ValueError("Нет данных для предобработки")
        
        # Предобработка supine данных
        if self.supine_data is not None:
            self.supine_features = self._preprocess_single_dataset(
                self.supine_data, feature_columns, scale_features, "supine"
            )
            results['supine'] = {
                'n_samples': len(self.supine_features),
                'n_features': len(feature_columns),
                'preprocessing_applied': ['missing_imputation', 'outlier_removal'] + (['scaling'] if scale_features else [])
            }
        
        # Предобработка lateral данных
        if self.lateral_data is not None:
            self.lateral_features = self._preprocess_single_dataset(
                self.lateral_data, feature_columns, scale_features, "lateral"
            )
            results['lateral'] = {
                'n_samples': len(self.lateral_features),
                'n_features': len(feature_columns),
                'preprocessing_applied': ['missing_imputation', 'outlier_removal'] + (['scaling'] if scale_features else [])
            }
        
        # Обучение распределений
        self._learn_feature_distributions(feature_columns)
        
        return results
    
    def _preprocess_single_dataset(self, 
                                 df: pd.DataFrame,
                                 feature_columns: List[str],
                                 scale_features: bool,
                                 data_type: str) -> np.ndarray:
        """
        Предобработать отдельный датасет
        
        Args:
            df: DataFrame
            feature_columns: Признаки
            scale_features: Нормализовать ли
            data_type: Тип данных
            
        Returns:
            Предобработанные признаки
        """
        # Выбор признаков
        available_cols = [col for col in feature_columns if col in df.columns]
        if not available_cols:
            raise ValueError(f"Нет доступных признаков из {feature_columns}")
        
        data = df[available_cols].copy()
        
        # Заполнение пропусков медианой
        for col in data.columns:
            if data[col].isnull().any():
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
        
        # Удаление выбросов (метод IQR)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Ограничение выбросов
            data[col] = np.clip(data[col], lower_bound, upper_bound)
        
        # Нормализация
        if scale_features:
            if self.scaler is None:
                self.scaler = StandardScaler()
                scaled_data = self.scaler.fit_transform(data.values)
            else:
                scaled_data = self.scaler.transform(data.values)
            return scaled_data
        else:
            return data.values
    
    def _learn_feature_distributions(self, feature_columns: List[str]):
        """Изучить распределения признаков"""
        if self.supine_features is not None:
            self.feature_distributions['supine'] = {}
            for i, col in enumerate(feature_columns):
                if i < self.supine_features.shape[1]:
                    feature_data = self.supine_features[:, i]
                    self.feature_distributions['supine'][col] = {
                        'mean': np.mean(feature_data),
                        'std': np.std(feature_data),
                        'min': np.min(feature_data),
                        'max': np.max(feature_data),
                        'median': np.median(feature_data),
                        'q25': np.percentile(feature_data, 25),
                        'q75': np.percentile(feature_data, 75)
                    }
        
        if self.lateral_features is not None:
            self.feature_distributions['lateral'] = {}
            for i, col in enumerate(feature_columns):
                if i < self.lateral_features.shape[1]:
                    feature_data = self.lateral_features[:, i]
                    self.feature_distributions['lateral'][col] = {
                        'mean': np.mean(feature_data),
                        'std': np.std(feature_data),
                        'min': np.min(feature_data),
                        'max': np.max(feature_data),
                        'median': np.median(feature_data),
                        'q25': np.percentile(feature_data, 25),
                        'q75': np.percentile(feature_data, 75)
                    }
    
    def generate_synthetic_paired_data(self, 
                                     n_samples: int = 100,
                                     method: str = "gaussian") -> Tuple[np.ndarray, np.ndarray]:
        """
        Сгенерировать синтетические парные данные из непарных
        
        Args:
            n_samples: Количество образцов для генерации
            method: Метод генерации ('gaussian', 'gmm', 'copula')
            
        Returns:
            (supine_features, lateral_features)
        """
        if self.supine_features is None or self.lateral_features is None:
            raise ValueError("Требуются и supine, и lateral данные")
        
        if method == "gaussian":
            return self._generate_gaussian_paired(n_samples)
        elif method == "gmm":
            return self._generate_gmm_paired(n_samples)
        elif method == "copula":
            return self._generate_copula_paired(n_samples)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    def _generate_gaussian_paired(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация парных данных с помощью гауссовских распределений"""
        n_features = min(self.supine_features.shape[1], self.lateral_features.shape[1])
        
        # Генерация из распределений
        synthetic_supine = np.zeros((n_samples, n_features))
        synthetic_lateral = np.zeros((n_samples, n_features))
        
        for i in range(n_features):
            # Supine распределение
            supine_mean = np.mean(self.supine_features[:, i])
            supine_std = np.std(self.supine_features[:, i])
            synthetic_supine[:, i] = np.random.normal(supine_mean, supine_std, n_samples)
            
            # Lateral распределение
            lateral_mean = np.mean(self.lateral_features[:, i])
            lateral_std = np.std(self.lateral_features[:, i])
            synthetic_lateral[:, i] = np.random.normal(lateral_mean, lateral_std, n_samples)
        
        return synthetic_supine, synthetic_lateral
    
    def _generate_gmm_paired(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация парных данных с помощью Gaussian Mixture Models"""
        n_features = min(self.supine_features.shape[1], self.lateral_features.shape[1])
        
        # Обучение GMM если еще не обучены
        if 'supine' not in self.gmm_models:
            self.gmm_models['supine'] = GaussianMixture(n_components=3, random_state=42)
            self.gmm_models['supine'].fit(self.supine_features[:, :n_features])
            
            self.gmm_models['lateral'] = GaussianMixture(n_components=3, random_state=42)
            self.gmm_models['lateral'].fit(self.lateral_features[:, :n_features])
        
        # Генерация из GMM
        synthetic_supine = self.gmm_models['supine'].sample(n_samples)[0]
        synthetic_lateral = self.gmm_models['lateral'].sample(n_samples)[0]
        
        return synthetic_supine, synthetic_lateral
    
    def _generate_copula_paired(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация парных данных с помощью копул"""
        n_features = min(self.supine_features.shape[1], self.lateral_features.shape[1])
        
        # Упрощенная копула - используем корреляцию
        combined_data = np.column_stack([
            self.supine_features[:, :n_features],
            self.lateral_features[:, :n_features]
        ])
        
        # Расчет корреляционной матрицы
        corr_matrix = np.corrcoef(combined_data.T)
        
        # Генерация из многомерного нормального распределения
        mean_vec = np.mean(combined_data, axis=0)
        synthetic_combined = np.random.multivariate_normal(mean_vec, corr_matrix, n_samples)
        
        # Разделение на supine и lateral
        synthetic_supine = synthetic_combined[:, :n_features]
        synthetic_lateral = synthetic_combined[:, n_features:n_features*2]
        
        return synthetic_supine, synthetic_lateral
    
    def calculate_regularization_loss(self, 
                                  paired_features: np.ndarray,
                                  paired_targets: np.ndarray,
                                  lambda_reg: float = 0.1) -> float:
        """
        Рассчитать регуляризационный loss на основе непарных данных
        
        Args:
            paired_features: Признаки парных данных
            paired_targets: Целевые переменные парных данных
            lambda_reg: Коэффициент регуляризации
            
        Returns:
            Регуляризационный loss
        """
        if self.supine_features is None or self.lateral_features is None:
            return 0.0
        
        # Расчет распределений парных данных
        paired_distributions = {}
        for i in range(min(paired_features.shape[1], 10)):  # Ограничиваем для скорости
            feature_data = paired_features[:, i]
            paired_distributions[f'feature_{i}'] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data)
            }
        
        # KL дивергенция между распределениями
        kl_loss = 0.0
        n_features_used = 0
        
        for feature_name, paired_dist in paired_distributions.items():
            if feature_name in self.feature_distributions.get('supine', {}):
                supine_dist = self.feature_distributions['supine'][feature_name]
                
                # KL дивергенция (упрощенная)
                paired_std = max(paired_dist['std'], 1e-6)
                supine_std = max(supine_dist['std'], 1e-6)
                
                kl = (np.log(supine_std / paired_std) + 
                       (paired_dist['std']**2 + (paired_dist['mean'] - supine_dist['mean'])**2) / (2 * supine_std**2) - 
                       0.5)
                
                kl_loss += abs(kl)
                n_features_used += 1
        
        if n_features_used > 0:
            kl_loss /= n_features_used
        
        return lambda_reg * kl_loss
    
    def augment_paired_training_data(self, 
                                  paired_features: np.ndarray,
                                  paired_targets: np.ndarray,
                                  augmentation_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Аугментировать парные данные с использованием непарных
        
        Args:
            paired_features: Признаки парных данных
            paired_targets: Целевые переменные
            augmentation_ratio: Доля аугментации (0-1)
            
        Returns:
            (аугментированные признаки, аугментированные цели)
        """
        if self.supine_features is None or self.lateral_features is None:
            return paired_features, paired_targets
        
        n_augment = int(len(paired_features) * augmentation_ratio)
        if n_augment == 0:
            return paired_features, paired_targets
        
        # Генерация синтетических данных
        synthetic_supine, synthetic_lateral = self.generate_synthetic_paired_data(n_augment)
        
        # Расчет синтетических целей (разница между lateral и supine)
        synthetic_targets = synthetic_lateral - synthetic_supine
        
        # Объединение с оригинальными данными
        augmented_features = np.vstack([paired_features, synthetic_supine])
        augmented_targets = np.vstack([paired_targets, synthetic_targets])
        
        print(f"Аугментировано {n_augment} образцов. Итого: {len(augmented_features)}")
        
        return augmented_features, augmented_targets
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Получить статистику по непарным данным
        
        Returns:
            Статистика данных
        """
        stats = {
            'supine_available': self.supine_features is not None,
            'lateral_available': self.lateral_features is not None,
            'n_supine_samples': len(self.supine_features) if self.supine_features is not None else 0,
            'n_lateral_samples': len(self.lateral_features) if self.lateral_features is not None else 0,
            'n_features': self.supine_features.shape[1] if self.supine_features is not None else 0,
            'feature_distributions_available': bool(self.feature_distributions),
            'scaling_applied': self.scaler is not None
        }
        
        # Статистика распределений
        if self.feature_distributions:
            stats['distribution_summary'] = {}
            for data_type, distributions in self.feature_distributions.items():
                stats['distribution_summary'][data_type] = {
                    'n_features': len(distributions),
                    'mean_std_ratio': np.mean([d['mean'] / max(d['std'], 1e-6) for d in distributions.values()]),
                    'avg_range': np.mean([d['max'] - d['min'] for d in distributions.values()])
                }
        
        return stats


def create_unpaired_manager() -> UnpairedDataManager:
    """
    Создать менеджер непарных данных
    
    Returns:
        UnpairedDataManager
    """
    return UnpairedDataManager()


if __name__ == "__main__":
    # Тестирование менеджера непарных данных
    print("=== Тестирование Unpaired Data Manager ===")
    
    # Создание тестовых данных
    np.random.seed(42)
    
    # Supine данные
    n_supine = 80
    supine_data = {
        'age': np.random.normal(50, 15, n_supine),
        'bmi': np.random.normal(25, 5, n_supine),
        'X_upper_supine': np.random.normal(-50, 30, n_supine),
        'Y_upper_supine': np.random.normal(-30, 20, n_supine),
        'Z_upper_supine': np.random.normal(200, 50, n_supine)
    }
    supine_df = pd.DataFrame(supine_data)
    supine_df.to_csv('test_supine.csv', index=False)
    
    # Lateral данные
    n_lateral = 40
    lateral_data = {
        'age': np.random.normal(55, 12, n_lateral),
        'bmi': np.random.normal(26, 4, n_lateral),
        'X_upper_lateral': np.random.normal(-45, 25, n_lateral),
        'Y_upper_lateral': np.random.normal(-25, 18, n_lateral),
        'Z_upper_lateral': np.random.normal(210, 45, n_lateral)
    }
    lateral_df = pd.DataFrame(lateral_data)
    lateral_df.to_csv('test_lateral.csv', index=False)
    
    # Загрузка и предобработка
    manager = create_unpaired_manager()
    
    print("Загрузка данных:")
    data_info = manager.load_unpaired_data(
        supine_path='test_supine.csv',
        lateral_path='test_lateral.csv'
    )
    
    for data_type, info in data_info.items():
        print(f"  {data_type}: {info.n_samples} образцов, качество={info.quality_score:.2f}")
    
    print("\nПредобработка:")
    preprocessing_results = manager.preprocess_unpaired_data(scale_features=True)
    for data_type, result in preprocessing_results.items():
        print(f"  {data_type}: {result}")
    
    # Генерация синтетических данных
    print("\nГенерация синтетических парных данных:")
    synthetic_supine, synthetic_lateral = manager.generate_synthetic_paired_data(
        n_samples=50, method="gmm"
    )
    print(f"Сгенерировано: {synthetic_supine.shape[0]} образцов")
    
    # Аугментация парных данных
    paired_features = np.random.randn(30, 5)
    paired_targets = np.random.randn(30, 5) * 10
    
    print("\nАугментация парных данных:")
    aug_features, aug_targets = manager.augment_paired_training_data(
        paired_features, paired_targets, augmentation_ratio=0.5
    )
    print(f"Размер после аугментации: {aug_features.shape}")
    
    # Статистика
    print("\nСтатистика данных:")
    stats = manager.get_data_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Очистка тестовых файлов
    import os
    if os.path.exists('test_supine.csv'):
        os.remove('test_supine.csv')
    if os.path.exists('test_lateral.csv'):
        os.remove('test_lateral.csv')
