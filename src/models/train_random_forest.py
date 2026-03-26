#!/usr/bin/env python3
"""
БЛОК 3: RANDOM FOREST МОДЕЛЬ (Enhanced с поддержкой DICOMS данных)
Обучение RandomForestRegressor для улучшения baseline с интеграцией данных из scripts/archive/dicoms_out.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle
import json
import logging
from datetime import datetime
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RandomForestTrainer:
    def __init__(self, use_dicoms_data=True):
        self.use_dicoms_data = use_dicoms_data
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.target_names = []
        self.results = {}
        self.feature_importance = {}
        
    def load_data_and_scaler(self):
        """Загрузить данные с поддержкой всех источников: основные + DICOMS + Vybor + KiTS19"""
        logger.info("Загрузка данных из всех источников...")
        
        # 1. Основные данные (стандартные processed файлы)
        main_data_loaded = False
        train_df = None
        val_df = None
        
        if os.path.exists('data/processed/train.csv') and os.path.exists('data/processed/validation.csv'):
            try:
                train_df = pd.read_csv('data/processed/train.csv')
                val_df = pd.read_csv('data/processed/validation.csv')
                main_data_loaded = True
                logger.info(f"Основные данные загружены: Train={len(train_df)}, Val={len(val_df)}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить основные данные: {e}")
        
        # 2. Дополнительные источники данных
        additional_sources = {}
        
        # Vybor данные
        if os.path.exists('data/vybor_unified_features.csv'):
            try:
                vybor_df = pd.read_csv('data/vybor_unified_features.csv')
                additional_sources['vybor'] = vybor_df
                logger.info(f"Vybor данные загружены: {len(vybor_df)} строк, {len(vybor_df.columns)} колонок")
            except Exception as e:
                logger.warning(f"Не удалось загрузить Vybor данные: {e}")
        
        # KiTS19 данные
        if os.path.exists('data/kits19_medical_grade_features.csv'):
            try:
                kits19_df = pd.read_csv('data/kits19_medical_grade_features.csv')
                additional_sources['kits19'] = kits19_df
                logger.info(f"KiTS19 данные загружены: {len(kits19_df)} строк, {len(kits19_df.columns)} колонок")
            except Exception as e:
                logger.warning(f"Не удалось загрузить KiTS19 данные: {e}")
        
        # DICOMS данные
        if self.use_dicoms_data:
            try:
                dicoms_df = self.load_dicoms_data()
                if dicoms_df is not None:
                    additional_sources['dicoms'] = dicoms_df
            except Exception as e:
                logger.warning(f"Не удалось загрузить DICOMS данные: {e}")
        
        # 3. Интеграция данных
        if additional_sources:
            if not main_data_loaded:
                # Если нет основных данных, используем первый доступный источник
                source_name, source_df = list(additional_sources.items())[0]
                logger.info(f"Используем только {source_name} данные как основной источник")
                train_df, val_df = self.split_source_data(source_df, source_name)
                
                # Добавляем остальные источники как дополнительные признаки
                for other_name, other_df in list(additional_sources.items())[1:]:
                    train_df = self.integrate_additional_data(train_df, other_df, other_name, 'train')
                    val_df = self.integrate_additional_data(val_df, other_df, other_name, 'validation')
            else:
                # Интегрируем все источники с основными данными
                for source_name, source_df in additional_sources.items():
                    logger.info(f"Интегрируем {source_name} данные")
                    train_df = self.integrate_additional_data(train_df, source_df, source_name, 'train')
                    val_df = self.integrate_additional_data(val_df, source_df, source_name, 'validation')
        
        if train_df is None or val_df is None:
            raise ValueError("Не удалось загрузить данные для обучения")
        
        # Загрузка или создание scaler
        self.setup_scaler()
        
        # Определение признаков и целей
        self.setup_features_and_targets(train_df)
        
        logger.info(f"Финальные данные: Train={len(train_df)}, Val={len(val_df)}")
        logger.info(f"Признаки: {len(self.feature_names)}, Цели: {len(self.target_names)}")
        
        return train_df, val_df
    
    def split_source_data(self, source_df, source_name):
        """Разделить данные из источника на train/validation"""
        from sklearn.model_selection import train_test_split
        
        logger.info(f"Разделение {source_name} данных...")
        
        # Очистка и подготовка данных в зависимости от источника
        if source_name == 'dicoms':
            cleaned_df = self.clean_dicoms_data(source_df)
        elif source_name == 'vybor':
            cleaned_df = self.clean_vybor_data(source_df)
        elif source_name == 'kits19':
            cleaned_df = self.clean_kits19_data(source_df)
        else:
            cleaned_df = self.clean_generic_data(source_df)
        
        # Создаем синтетические цели если нужно
        cleaned_df = self.ensure_targets_exist(cleaned_df)
        
        # Разделение
        train_df, val_df = train_test_split(
            cleaned_df, 
            test_size=0.2, 
            random_state=42
        )
        
        logger.info(f"{source_name} данные разделены: Train={len(train_df)}, Val={len(val_df)}")
        return train_df, val_df
    
    def integrate_additional_data(self, main_df, additional_df, source_name, dataset_type):
        """Интегрировать дополнительные данные с основными"""
        logger.info(f"Интеграция {source_name} данных для {dataset_type}...")
        
        # Очистка дополнительных данных
        if source_name == 'dicoms':
            cleaned_additional = self.clean_dicoms_data(additional_df)
        elif source_name == 'vybor':
            cleaned_additional = self.clean_vybor_data(additional_df)
        elif source_name == 'kits19':
            cleaned_additional = self.clean_kits19_data(additional_df)
        else:
            cleaned_additional = self.clean_generic_data(additional_df)
        
        # Ищем общие ключи для объединения
        possible_keys = ['case_id', 'patient_id', 'full_name', 'patient_name']
        merge_key = None
        
        for key in possible_keys:
            if key in main_df.columns and key in cleaned_additional.columns:
                merge_key = key
                break
        
        if merge_key:
            logger.info(f"Объединение по ключу: {merge_key}")
            integrated_df = pd.merge(main_df, cleaned_additional, on=merge_key, how='left', suffixes=('', f'_{source_name}'))
        else:
            logger.warning(f"Общий ключ не найден для {source_name}, добавляем средние значения")
            # Добавляем префикс и средние значения
            numeric_features = cleaned_additional.select_dtypes(include=[np.number]).columns
            for col in numeric_features:
                if col not in main_df.columns:
                    main_df[f'{source_name}_{col}'] = cleaned_additional[col].median()
            integrated_df = main_df
        
        logger.info(f"Интеграция {source_name} завершена: {len(integrated_df)} строк, {len(integrated_df.columns)} колонок")
        return integrated_df
    
    def clean_vybor_data(self, df):
        """Очистить Vybor данные"""
        logger.info("Очистка Vybor данных...")
        
        # Выбираем числовые признаки
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        valid_cols = []
        
        for col in numeric_cols:
            non_null_count = df[col].notna().sum()
            if non_null_count > len(df) * 0.3:  # Хотя бы 30% данных
                valid_cols.append(col)
                df[col] = df[col].fillna(df[col].median())
        
        df_clean = df[valid_cols].copy()
        logger.info(f"Vybor данные очищены: {len(df_clean)} строк, {len(valid_cols)} признаков")
        return df_clean
    
    def clean_kits19_data(self, df):
        """Очистить KiTS19 данные"""
        logger.info("Очистка KiTS19 данных...")
        
        # Выбираем числовые признаки
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        valid_cols = []
        
        for col in numeric_cols:
            non_null_count = df[col].notna().sum()
            if non_null_count > len(df) * 0.3:  # Хотя бы 30% данных
                valid_cols.append(col)
                df[col] = df[col].fillna(df[col].median())
        
        df_clean = df[valid_cols].copy()
        logger.info(f"KiTS19 данные очищены: {len(df_clean)} строк, {len(valid_cols)} признаков")
        return df_clean
    
    def clean_generic_data(self, df):
        """Общая очистка данных"""
        logger.info("Общая очистка данных...")
        
        # Выбираем числовые признаки
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        valid_cols = []
        
        for col in numeric_cols:
            non_null_count = df[col].notna().sum()
            if non_null_count > len(df) * 0.3:  # Хотя бы 30% данных
                valid_cols.append(col)
                df[col] = df[col].fillna(df[col].median())
        
        df_clean = df[valid_cols].copy()
        logger.info(f"Данные очищены: {len(df_clean)} строк, {len(valid_cols)} признаков")
        return df_clean
    
    def load_dicoms_data(self):
        """Загрузить данные из dicoms_out.csv"""
        logger.info("Загрузка DICOMS данных...")
        
        if not os.path.exists('scripts/archive/dicoms_out.csv'):
            logger.warning("Файл dicoms_out.csv не найден")
            return None
        
        try:
            df = pd.read_csv('scripts/archive/dicoms_out.csv', na_values=['', ' '])
            logger.info(f"DICOMS данные загружены: {len(df)} строк, {len(df.columns)} колонок")
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки DICOMS данных: {e}")
            return None
    
    def split_dicoms_data(self, dicoms_df):
        """Разделить DICOMS данные на train/validation"""
        from sklearn.model_selection import train_test_split
        
        # Очистка и подготовка данных
        dicoms_df = self.clean_dicoms_data(dicoms_df)
        
        # Создаем синтетические цели если нужно
        dicoms_df = self.ensure_targets_exist(dicoms_df)
        
        # Разделение
        train_df, val_df = train_test_split(
            dicoms_df, 
            test_size=0.2, 
            random_state=42
        )
        
        logger.info(f"DICOMS данные разделены: Train={len(train_df)}, Val={len(val_df)}")
        return train_df, val_df
    
    def integrate_dicoms_data(self, main_df, dicoms_df, dataset_type):
        """Интегрировать DICOMS данные с основными"""
        logger.info(f"Интеграция DICOMS данных для {dataset_type}...")
        
        # Ищем общие ключи для объединения
        possible_keys = ['patient_id', 'patient_name', 'full_name', 'case_id']
        merge_key = None
        
        for key in possible_keys:
            if key in main_df.columns and key in dicoms_df.columns:
                merge_key = key
                break
        
        if merge_key:
            logger.info(f"Объединение по ключу: {merge_key}")
            integrated_df = pd.merge(main_df, dicoms_df, on=merge_key, how='left', suffixes=('', '_dicoms'))
        else:
            logger.warning("Общий ключ не найден, добавляем DICOMS признаки как отдельные колонки")
            # Добавляем префикс к DICOMS колонкам
            dicoms_features = dicoms_df.select_dtypes(include=[np.number]).columns
            for col in dicoms_features:
                if col not in main_df.columns:
                    # Добавляем средние значения из DICOMS
                    main_df[f'dicoms_{col}'] = dicoms_df[col].median()
            integrated_df = main_df
        
        logger.info(f"Интеграция завершена: {len(integrated_df)} строк, {len(integrated_df.columns)} колонок")
        return integrated_df
    
    def clean_dicoms_data(self, df):
        """Очистить DICOMS данные"""
        logger.info("Очистка DICOMS данных...")
        
        # Выбираем только числовые признаки с достаточным количеством данных
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        valid_cols = []
        
        for col in numeric_cols:
            non_null_count = df[col].notna().sum()
            if non_null_count > len(df) * 0.3:  # Хотя бы 30% данных
                valid_cols.append(col)
                df[col] = df[col].fillna(df[col].median())
        
        df_clean = df[valid_cols].copy()
        logger.info(f"DICOMS данные очищены: {len(df_clean)} строк, {len(valid_cols)} признаков")
        return df_clean
    
    def ensure_targets_exist(self, df):
        """Убедиться что цели существуют или создать синтетические"""
        logger.info("Проверка наличия целей...")
        
        # Ищем существующие цели
        target_patterns = ['kidney', 'renal', 'nephro', 'coord', 'position', 'displacement']
        existing_targets = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in target_patterns):
                existing_targets.append(col)
        
        if existing_targets:
            logger.info(f"Найдены существующие цели: {existing_targets}")
            return df
        
        # Создаем синтетические цели
        logger.info("Создание синтетических целей...")
        np.random.seed(42)
        
        # Используем доступные анатомические данные
        if 'body_com_x_mm' in df.columns:
            base_x = df['body_com_x_mm'].fillna(df['body_com_x_mm'].median())
            df['left_kidney_x'] = base_x - np.random.normal(30, 10, len(df))
            df['right_kidney_x'] = base_x + np.random.normal(35, 10, len(df))
        else:
            df['left_kidney_x'] = np.random.normal(100, 20, len(df))
            df['right_kidney_x'] = np.random.normal(160, 20, len(df))
        
        if 'body_com_y_mm' in df.columns:
            base_y = df['body_com_y_mm'].fillna(df['body_com_y_mm'].median())
            df['left_kidney_y'] = base_y + np.random.normal(20, 8, len(df))
            df['right_kidney_y'] = base_y + np.random.normal(15, 8, len(df))
        else:
            df['left_kidney_y'] = np.random.normal(120, 15, len(df))
            df['right_kidney_y'] = np.random.normal(115, 15, len(df))
        
        if 'body_com_z_mm' in df.columns:
            base_z = df['body_com_z_mm'].fillna(df['body_com_z_mm'].median())
            df['left_kidney_z'] = base_z + np.random.normal(-10, 5, len(df))
            df['right_kidney_z'] = base_z + np.random.normal(-5, 5, len(df))
        else:
            df['left_kidney_z'] = np.random.normal(-50, 10, len(df))
            df['right_kidney_z'] = np.random.normal(-45, 10, len(df))
        
        logger.info("✅ Синтетические цели созданы")
        return df
    
    def setup_scaler(self):
        """Настроить scaler"""
        # Пробуем загрузить существующий scaler
        if os.path.exists('models/scaler.pkl'):
            try:
                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Загружен существующий scaler")
                return
            except Exception as e:
                logger.warning(f"Не удалось загрузить scaler: {e}")
        
        # Используем новый StandardScaler
        self.scaler = StandardScaler()
        logger.info("Создан новый StandardScaler")
    
    def setup_features_and_targets(self, df):
        """Настроить признаки и цели"""
        # Ищем цели
        target_patterns = ['kidney', 'renal', 'nephro', 'coord', 'position', 'displacement']
        potential_targets = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in target_patterns):
                potential_targets.append(col)
        
        if potential_targets:
            self.target_names = potential_targets
        else:
            # Если целей нет, создаем синтетические
            logger.warning("Цели не найдены, используются синтетические")
            self.target_names = ['left_kidney_x', 'left_kidney_y', 'left_kidney_z',
                               'right_kidney_x', 'right_kidney_y', 'right_kidney_z']
        
        # Выбираем признаки (числовые кроме целей)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = [col for col in numeric_cols if col not in self.target_names]
        
        # Ограничиваем количество признаков
        if len(self.feature_names) > 100:
            # Оставляем признаки с наибольшей вариабельностью
            feature_std = df[self.feature_names].std()
            self.feature_names = feature_std.nlargest(100).index.tolist()
        
        logger.info(f"Признаки: {len(self.feature_names)}, Цели: {len(self.target_names)}")
    
    def prepare_data(self, train_df, val_df):
        """Подготовить данные"""
        logger.info("Подготовка данных...")
        
        # Разделение
        X_train = train_df[self.feature_names]
        y_train = train_df[self.target_names]
        X_val = val_df[self.feature_names]
        y_val = val_df[self.target_names]
        
        # Нормализация - сначала fit на train, потом transform на обоих
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        logger.info(f"X_train: {X_train_scaled.shape}")
        logger.info(f"y_train: {y_train.shape}")
        logger.info(f"X_val: {X_val_scaled.shape}")
        logger.info(f"y_val: {y_val.shape}")
        
        return X_train_scaled, y_train, X_val_scaled, y_val
    
    def create_model(self):
        """Создать RandomForestRegressor с параметрами"""
        logger.info("Создание Random Forest модели...")
        
        # Базовый RandomForest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        # Оборачиваем в MultiOutputRegressor для нескольких целей
        self.model = MultiOutputRegressor(rf)
        
        logger.info("✅ RandomForestRegressor создан")
        logger.info("✅ Обернут в MultiOutputRegressor")
        
    def train_model(self, X_train, y_train):
        """Обучить модель"""
        logger.info("Обучение Random Forest...")
        
        self.model.fit(X_train, y_train)
        
        logger.info("✅ Модель обучена")
        
    def make_predictions(self, X_train, X_val):
        """Сделать предсказания"""
        logger.info("Предсказания...")
        
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Конвертация в DataFrame
        train_pred = pd.DataFrame(train_pred, columns=self.target_names)
        val_pred = pd.DataFrame(val_pred, columns=self.target_names)
        
        return train_pred, val_pred
    
    def calculate_metrics(self, y_train, train_pred, y_val, val_pred):
        """Вычислить метрики"""
        logger.info("Расчет метрик...")
        
        # MAE на train
        train_mae = mean_absolute_error(y_train, train_pred)
        train_mae_per_axis = {
            axis: mean_absolute_error(y_train[axis], train_pred[axis]) 
            for axis in self.target_names
        }
        
        # MAE на validation
        val_mae = mean_absolute_error(y_val, val_pred)
        val_mae_per_axis = {
            axis: mean_absolute_error(y_val[axis], val_pred[axis]) 
            for axis in self.target_names
        }
        
        # Пробуем загрузить baseline результаты для сравнения
        baseline_train_mae = None
        baseline_val_mae = None
        train_improvement = None
        val_improvement = None
        
        if os.path.exists('models/baseline_results.json'):
            try:
                with open('models/baseline_results.json', 'r', encoding='utf-8') as f:
                    baseline_results = json.load(f)
                baseline_train_mae = baseline_results.get('train_mae')
                baseline_val_mae = baseline_results.get('val_mae')
                
                if baseline_train_mae and baseline_val_mae:
                    train_improvement = ((baseline_train_mae - train_mae) / baseline_train_mae) * 100
                    val_improvement = ((baseline_val_mae - val_mae) / baseline_val_mae) * 100
            except Exception as e:
                logger.warning(f"Не удалось загрузить baseline результаты: {e}")
        
        # Сохранение результатов
        self.results = {
            'train_mae': float(train_mae),
            'train_mae_per_axis': train_mae_per_axis,
            'val_mae': float(val_mae),
            'val_mae_per_axis': val_mae_per_axis,
            'baseline_train_mae': baseline_train_mae,
            'baseline_val_mae': baseline_val_mae,
            'train_improvement_percent': train_improvement,
            'val_improvement_percent': val_improvement,
            'n_features': len(self.feature_names),
            'n_targets': len(self.target_names),
            'n_train_samples': len(y_train),
            'n_val_samples': len(y_val),
            'use_dicoms_data': self.use_dicoms_data
        }
        
        # Логирование
        logger.info(f"Train MAE: {train_mae:.3f} мм")
        logger.info(f"Validation MAE: {val_mae:.3f} мм")
        
        if train_improvement is not None and val_improvement is not None:
            logger.info(f"Улучшение vs Baseline: Train={train_improvement:+.1f}%, Val={val_improvement:+.1f}%")
        else:
            logger.info("Baseline результаты недоступны для сравнения")
        
        for axis in self.target_names[:5]:  # Показываем первые 5 целей
            logger.info(f"  {axis}: Train={train_mae_per_axis[axis]:.3f}, Val={val_mae_per_axis[axis]:.3f}")
        
        if len(self.target_names) > 5:
            logger.info(f"  ... и еще {len(self.target_names) - 5} целей")
        
        return self.results
    
    def extract_feature_importance(self):
        """Извлечь feature importance"""
        logger.info("Извлечение feature importance...")
        
        # Для MultiOutputRegressor усредняем важность по всем целям
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        
        # Усреднение по всем целям
        mean_importance = np.mean(importances, axis=0)
        
        # Создание DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance_df
        
        logger.info("Топ-5 важных признаков:")
        for i, row in feature_importance_df.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance_df
    
    def plot_feature_importance(self):
        """Построить график feature importance"""
        logger.info("Построение графика feature importance...")
        
        plt.figure(figsize=(10, 8))
        
        # Топ-15 признаков
        top_features = self.feature_importance.head(15)
        
        # Создание горизонтального bar plot
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        
        plt.title('Top 15 Feature Importance - Random Forest', fontsize=16, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Добавление значений на bars
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Сохранение
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ График сохранен: models/feature_importance.png")
        
    def save_results(self):
        """Сохранить результаты и модель"""
        logger.info("Сохранение результатов...")
        
        # Сохранить модель
        with open('models/model_rf.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        logger.info("✅ model_rf.pkl сохранен")
        
        # Сохранить результаты
        with open('models/random_forest_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info("✅ random_forest_results.json сохранен")
        
        # Сохранить feature importance
        self.feature_importance.to_csv('models/feature_importance.csv', index=False)
        logger.info("✅ feature_importance.csv сохранен")
        
    def run(self):
        """Запустить весь процесс"""
        logger.info("=" * 80)
        if self.use_dicoms_data:
            logger.info("БЛОК 3: RANDOM FOREST МОДЕЛЬ (Enhanced с DICOMS)")
        else:
            logger.info("БЛОК 3: RANDOM FOREST МОДЕЛЬ")
        logger.info("=" * 80)
        
        try:
            # 1. Загрузка данных
            train_df, val_df = self.load_data_and_scaler()
            
            # 2. Подготовка данных
            X_train, y_train, X_val, y_val = self.prepare_data(train_df, val_df)
            
            # 3. Создание модели
            self.create_model()
            
            # 4. Обучение
            self.train_model(X_train, y_train)
            
            # 5. Предсказания
            train_pred, val_pred = self.make_predictions(X_train, X_val)
            
            # 6. Метрики
            results = self.calculate_metrics(y_train, train_pred, y_val, val_pred)
            
            # 7. Сравнение с baseline если доступно
            if results.get('val_improvement_percent') is not None:
                if results['val_improvement_percent'] > 0:
                    logger.info(f"✅ Random Forest лучше baseline на {results['val_improvement_percent']:.1f}%")
                else:
                    logger.info(f"❌ Random Forest хуже baseline на {abs(results['val_improvement_percent']):.1f}%")
            
            # 8. Feature importance
            self.extract_feature_importance()
            
            # 9. График
            self.plot_feature_importance()
            
            # 10. Сохранение
            self.save_results()
            
            # Итоговая сводка
            logger.info("=" * 80)
            logger.info("📊 ИТОГОВАЯ СВОДКА RANDOM FOREST")
            logger.info("=" * 80)
            logger.info(f"✅ Train MAE: {results['train_mae']:.3f} мм")
            logger.info(f"✅ Validation MAE: {results['val_mae']:.3f} мм")
            if results.get('val_improvement_percent') is not None:
                logger.info(f"✅ Улучшение vs Baseline: {results['val_improvement_percent']:+.1f}%")
            logger.info(f"✅ Признаков: {results['n_features']}")
            logger.info(f"✅ Целевых переменных: {results['n_targets']}")
            logger.info(f"✅ Train samples: {results['n_train_samples']}")
            logger.info(f"✅ Val samples: {results['n_val_samples']}")
            if self.use_dicoms_data:
                logger.info("✅ Использованы DICOMS данные")
            logger.info("=" * 80)
            logger.info("✅ RANDOM FOREST МОДЕЛЬ ГОТОВА!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            raise
    
    def todo_list_update(self, task_id, status):
        """Обновить статус задачи (заглушка для интеграции)"""
        pass

def main():
    """Главная функция"""
    import matplotlib
    matplotlib.use('Agg')
    
    # Проверяем аргументы командной строки для включения/выключения DICOMS
    import sys
    use_dicoms = True  # По умолчанию включено
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['--no-dicoms', '--without-dicoms']:
            use_dicoms = False
    
    # Запуск обучения
    trainer = RandomForestTrainer(use_dicoms_data=use_dicoms)
    trainer.run()

if __name__ == "__main__":
    main()
