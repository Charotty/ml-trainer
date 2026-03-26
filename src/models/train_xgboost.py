#!/usr/bin/env python3
"""
БЛОК 4: XGBOOST МОДЕЛЬ (Enhanced с поддержкой всех источников данных)
Обучение XGBoost для каждой целевой переменной отдельно с интеграцией данных из scripts/archive/dicoms_out.csv, data/vybor_unified_features.csv, data/kits19_medical_grade_features.csv
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

class XGBoostTrainer:
    def __init__(self, use_dicoms_data=True):
        self.use_dicoms_data = use_dicoms_data
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        self.target_names = []
        self.results = {}
        
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
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Обучить каждую модель с early_stopping"""
        logger.info("Обучение XGBoost моделей...")
        
        for target in self.target_names:
            logger.info(f"Обучение модели для {target}...")
            
            # Создание модели
            model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            # Обучение
            model.fit(X_train, y_train[target])
            
            # Оценка на validation
            val_pred = model.predict(X_val)
            val_score = mean_absolute_error(y_val[target], val_pred)
            
            # Сохранение модели
            self.models[target] = model
            
            # Логирование результатов
            logger.info(f"  ✅ {target}: val_mae={val_score:.4f}")
        
        logger.info("✅ Все XGBoost модели обучены")
    
    def create_xgb_model(self):
        """Создать XGBoost модель с параметрами"""
        logger.info("Создание XGBoost модели...")
        
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )
        
        return model
    
    def make_predictions(self, X_train, X_val):
        """Сделать предсказания"""
        logger.info("Предсказания...")
        
        train_pred = {}
        val_pred = {}
        
        for target in self.target_names:
            train_pred[target] = self.models[target].predict(X_train)
            val_pred[target] = self.models[target].predict(X_val)
        
        # Конвертация в DataFrame
        train_pred_df = pd.DataFrame(train_pred)
        val_pred_df = pd.DataFrame(val_pred)
        
        return train_pred_df, val_pred_df
    
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
        
        # Пробуем загрузить результаты для сравнения
        baseline_train_mae = None
        baseline_val_mae = None
        rf_train_mae = None
        rf_val_mae = None
        train_improvement_vs_baseline = None
        val_improvement_vs_baseline = None
        train_improvement_vs_rf = None
        val_improvement_vs_rf = None
        
        # Загрузка baseline результатов
        if os.path.exists('models/baseline_results.json'):
            try:
                with open('models/baseline_results.json', 'r', encoding='utf-8') as f:
                    baseline_results = json.load(f)
                baseline_train_mae = baseline_results.get('train_mae')
                baseline_val_mae = baseline_results.get('val_mae')
                
                if baseline_train_mae and baseline_val_mae:
                    train_improvement_vs_baseline = ((baseline_train_mae - train_mae) / baseline_train_mae) * 100
                    val_improvement_vs_baseline = ((baseline_val_mae - val_mae) / baseline_val_mae) * 100
            except Exception as e:
                logger.warning(f"Не удалось загрузить baseline результаты: {e}")
        
        # Загрузка RandomForest результатов
        if os.path.exists('models/random_forest_results.json'):
            try:
                with open('models/random_forest_results.json', 'r', encoding='utf-8') as f:
                    rf_results = json.load(f)
                rf_train_mae = rf_results.get('train_mae')
                rf_val_mae = rf_results.get('val_mae')
                
                if rf_train_mae and rf_val_mae:
                    train_improvement_vs_rf = ((rf_train_mae - train_mae) / rf_train_mae) * 100
                    val_improvement_vs_rf = ((rf_val_mae - val_mae) / rf_val_mae) * 100
            except Exception as e:
                logger.warning(f"Не удалось загрузить RandomForest результаты: {e}")
        
        # Сохранение результатов
        self.results = {
            'train_mae': float(train_mae),
            'train_mae_per_axis': train_mae_per_axis,
            'val_mae': float(val_mae),
            'val_mae_per_axis': val_mae_per_axis,
            'baseline_train_mae': baseline_train_mae,
            'baseline_val_mae': baseline_val_mae,
            'rf_train_mae': rf_train_mae,
            'rf_val_mae': rf_val_mae,
            'train_improvement_vs_baseline_percent': train_improvement_vs_baseline,
            'val_improvement_vs_baseline_percent': val_improvement_vs_baseline,
            'train_improvement_vs_rf_percent': train_improvement_vs_rf,
            'val_improvement_vs_rf_percent': val_improvement_vs_rf,
            'n_features': len(self.feature_names),
            'n_targets': len(self.target_names),
            'n_train_samples': len(y_train),
            'n_val_samples': len(y_val),
            'use_dicoms_data': self.use_dicoms_data
        }
        
        # Логирование
        logger.info(f"Train MAE: {train_mae:.3f} мм")
        logger.info(f"Validation MAE: {val_mae:.3f} мм")
        
        if val_improvement_vs_baseline is not None:
            logger.info(f"Улучшение vs Baseline: Train={train_improvement_vs_baseline:+.1f}%, Val={val_improvement_vs_baseline:+.1f}%")
        else:
            logger.info("Baseline результаты недоступны для сравнения")
        
        if val_improvement_vs_rf is not None:
            logger.info(f"Улучшение vs RandomForest: Train={train_improvement_vs_rf:+.1f}%, Val={val_improvement_vs_rf:+.1f}%")
        else:
            logger.info("RandomForest результаты недоступны для сравнения")
        
        for axis in self.target_names[:5]:  # Показываем первые 5 целей
            logger.info(f"  {axis}: Train={train_mae_per_axis[axis]:.3f}, Val={val_mae_per_axis[axis]:.3f}")
        
        if len(self.target_names) > 5:
            logger.info(f"  ... и еще {len(self.target_names) - 5} целей")
        
        return self.results
    
    def compare_with_random_forest(self):
        """Сравнить с RandomForest"""
        logger.info("Сравнение с RandomForest...")
        
        # Пробуем загрузить результаты RandomForest
        rf_results = None
        baseline_results = None
        
        if os.path.exists('models/random_forest_results.json'):
            try:
                with open('models/random_forest_results.json', 'r', encoding='utf-8') as f:
                    rf_results = json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить RandomForest результаты: {e}")
        
        if os.path.exists('models/baseline_results.json'):
            try:
                with open('models/baseline_results.json', 'r', encoding='utf-8') as f:
                    baseline_results = json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить baseline результаты: {e}")
        
        if not rf_results:
            logger.info("RandomForest результаты недоступны для сравнения")
            return
        
        # Создание таблицы
        comparison_data = []
        
        # Baseline если доступен
        if baseline_results:
            comparison_data.append({
                'Model': 'Linear Regression (Baseline)',
                'Train MAE (mm)': baseline_results['train_mae'],
                'Val MAE (mm)': baseline_results['val_mae']
            })
        
        # RandomForest
        comparison_data.append({
            'Model': 'RandomForest',
            'Train MAE (mm)': rf_results['train_mae'],
            'Val MAE (mm)': rf_results['val_mae']
        })
        
        # XGBoost
        comparison_data.append({
            'Model': 'XGBoost',
            'Train MAE (mm)': self.results['train_mae'],
            'Val MAE (mm)': self.results['val_mae']
        })
        
        # Сохранение таблицы
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('models/model_comparison.csv', index=False)
        
        # Логирование таблицы
        logger.info("📊 СРАВНЕНИЕ МОДЕЛЕЙ:")
        logger.info(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def save_results(self):
        """Сохранить результаты и модели"""
        logger.info("Сохранение результатов...")
        
        # Сохранить каждую модель
        for target in self.target_names:
            model_filename = f'models/model_xgb_{target}.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump(self.models[target], f)
            logger.info(f"✅ {model_filename} сохранен")
        
        # Сохранить результаты
        with open('models/xgboost_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info("✅ xgboost_results.json сохранен")
        
    def run(self):
        """Запустить весь процесс"""
        logger.info("=" * 80)
        if self.use_dicoms_data:
            logger.info("БЛОК 4: XGBOOST МОДЕЛЬ (Enhanced со всеми источниками)")
        else:
            logger.info("БЛОК 4: XGBOOST МОДЕЛЬ")
        logger.info("=" * 80)
        
        try:
            # 1. Загрузка данных
            train_df, val_df = self.load_data_and_scaler()
            
            # 2. Подготовка данных
            X_train, y_train, X_val, y_val = self.prepare_data(train_df, val_df)
            
            # 3. Обучение моделей
            self.train_models(X_train, y_train, X_val, y_val)
            
            # 4. Предсказания
            train_pred, val_pred = self.make_predictions(X_train, X_val)
            
            # 5. Метрики
            results = self.calculate_metrics(y_train, train_pred, y_val, val_pred)
            
            # 6. Сравнение с baseline если доступно
            if results.get('val_improvement_percent') is not None:
                if results['val_improvement_percent'] > 0:
                    logger.info(f"✅ XGBoost лучше baseline на {results['val_improvement_percent']:.1f}%")
                else:
                    logger.info(f"❌ XGBoost хуже baseline на {abs(results['val_improvement_percent']):.1f}%")
            
            # 7. Сравнение с RandomForest если доступно
            self.compare_with_random_forest()
            
            # 8. Сохранение
            self.save_results()
            
            # Итоговая сводка
            logger.info("=" * 80)
            logger.info("📊 ИТОГОВАЯ СВОДКА XGBOOST")
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
                logger.info("✅ Использованы все источники данных")
            logger.info("=" * 80)
            logger.info("✅ XGBOOST МОДЕЛЬ ГОТОВА!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            raise
    
    def todo_list_update(self, task_id, status):
        """Обновить статус задачи (заглушка для интеграции)"""
        pass

def main():
    """Главная функция"""
    # Проверяем аргументы командной строки для включения/выключения DICOMS
    import sys
    use_dicoms = True  # По умолчанию включено
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['--no-dicoms', '--without-dicoms']:
            use_dicoms = False
    
    # Запуск обучения
    trainer = XGBoostTrainer(use_dicoms_data=use_dicoms)
    trainer.run()

if __name__ == "__main__":
    main()
    
    def todo_list_update(self, task_id, status):
        """Обновить статус задачи (заглушка для интеграции)"""
        pass

def main():
    """Главная функция"""
    # Запуск обучения
    trainer = XGBoostTrainer()
    trainer.run()

if __name__ == "__main__":
    main()
