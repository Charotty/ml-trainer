#!/usr/bin/env python3
"""
ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ ДАННЫХ
Полная очистка и подготовка данных для обучения моделей
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def final_data_fix():
    """Финальная очистка данных"""
    logger.info("Загрузка данных...")
    
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/validation.csv')
    
    logger.info(f"Train до очистки: {train_df.shape}")
    logger.info(f"Val до очистки: {val_df.shape}")
    
    def clean_dataframe(df, name):
        logger.info(f"Финальная очистка {name}...")
        
        # 1. Удаляем явно строковые колонки (идентификаторы)
        string_cols_to_remove = [
            'case_id', 'full_name', 'study_date', 'scan_position', 
            'contrast_phase', 'source_name', 'universal_id', 'source'
        ]
        
        cols_to_remove = [col for col in string_cols_to_remove if col in df.columns]
        if cols_to_remove:
            logger.info(f"Удалены строковые колонки: {cols_to_remove}")
            df = df.drop(columns=cols_to_remove)
        
        # 2. Конвертируем оставшиеся объектные колонки
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.info(f"Обработка объектной колонки: {col}")
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 10:  # Категориальная
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    logger.info(f"  Закодирована категориальная колонка {col}")
                else:
                    # Слишком много уникальных значений - удаляем
                    logger.warning(f"Удалена колонка {col} (слишком много уникальных: {len(unique_vals)})")
                    df = df.drop(columns=[col])
        
        # 3. Убедимся что все числовые данные действительно числовые
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Конвертируем в numeric, ошибки заменяем на NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 4. Удаляем колонки с слишком многими NaN
        nan_threshold = len(df) * 0.5  # 50%
        cols_to_keep = []
        
        for col in df.columns:
            nan_count = df[col].isnull().sum()
            if nan_count <= nan_threshold:
                cols_to_keep.append(col)
            else:
                logger.warning(f"Удалена колонка {col}: {nan_count/len(df)*100:.1f}% пропусков")
        
        df = df[cols_to_keep]
        
        # 5. Заполняем оставшиеся NaN
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Заполнен NaN в {col}: медиана = {median_val}")
                else:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                    df[col] = df[col].fillna(mode_val)
        
        # 6. Финальная проверка
        remaining_nan = df.isnull().sum().sum()
        if remaining_nan > 0:
            logger.warning(f"Удаление строк с оставшимися NaN: {remaining_nan}")
            df = df.dropna()
        
        logger.info(f"{name} после очистки: {df.shape}")
        logger.info(f"  Типы колонок: {df.dtypes.value_counts().to_dict()}")
        
        return df
    
    # Очищаем данные
    train_clean = clean_dataframe(train_df, "train")
    val_clean = clean_dataframe(val_df, "validation")
    
    # Сохраняем очищенные данные
    train_clean.to_csv('data/processed/train.csv', index=False)
    val_clean.to_csv('data/processed/validation.csv', index=False)
    
    # Создаем backup
    train_clean.to_csv('data/processed/train_backup.csv', index=False)
    val_clean.to_csv('data/processed/validation_backup.csv', index=False)
    
    logger.info("✅ Финальная очистка завершена")
    
    return train_clean, val_clean

if __name__ == "__main__":
    train_clean, val_clean = final_data_fix()
    
    print(f"\n📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Train: {len(train_clean)} строк, {len(train_clean.columns)} колонок")
    print(f"   Validation: {len(val_clean)} строк, {len(val_clean.columns)} колонок")
    print(f"   NaN в train: {train_clean.isnull().sum().sum()}")
    print(f"   NaN в validation: {val_clean.isnull().sum().sum()}")
    print(f"   Типы данных: {train_clean.dtypes.value_counts().to_dict()}")
