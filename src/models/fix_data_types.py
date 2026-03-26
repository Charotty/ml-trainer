#!/usr/bin/env python3
"""
ИСПРАВЛЕНИЕ ТИПОВ ДАННЫХ
Удаление строковых значений из числовых колонок
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_data_types():
    """Исправить типы данных в train/validation"""
    logger.info("Загрузка данных...")
    
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/validation.csv')
    
    logger.info(f"Train до исправления: {train_df.shape}")
    logger.info(f"Val до исправления: {val_df.shape}")
    
    def fix_dataframe_types(df, name):
        logger.info(f"Исправление типов в {name}...")
        
        # Находим числовые колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Найдено числовых колонок: {len(numeric_cols)}")
        
        # Проверяем на строковые значения в числовых колонках
        string_issues = {}
        for col in numeric_cols:
            non_numeric = df[col].apply(lambda x: isinstance(x, str) and not x.replace('.', '').replace('-', '').isdigit())
            if non_numeric.any():
                string_issues[col] = df[col][non_numeric].unique().tolist()
                logger.warning(f"Колонка {col} содержит строковые значения: {string_issues[col][:5]}")
        
        # Конвертируем problematic колонки
        for col in numeric_cols:
            if col in string_issues:
                # Заменяем строковые значения на NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"Конвертирована колонка {col}")
        
        # Удаляем строки с NaN в ключевых колонках
        key_cols = ['age', 'bmi', 'sex']  # Ключевые демографические данные
        
        for col in key_cols:
            if col in df.columns:
                before_count = len(df)
                df = df.dropna(subset=[col])
                after_count = len(df)
                if before_count != after_count:
                    logger.info(f"Удалено строк с NaN в {col}: {before_count - after_count}")
        
        # Конвертируем sex в числовой формат
        if 'sex' in df.columns:
            df['sex'] = df['sex'].map({1.0: 1, 2.0: 2, 1: 1, 2: 2, 'male': 1, 'female': 2, 'm': 1, 'f': 2})
            df['sex'] = pd.to_numeric(df['sex'], errors='coerce')
            df['sex'] = df['sex'].fillna(df['sex'].median())
            logger.info(f"Конвертирована колонка sex: уникальные значения {df['sex'].unique()}")
        
        # Финальная проверка на NaN
        total_nan = df.isnull().sum().sum()
        if total_nan > 0:
            logger.warning(f"Осталось NaN: {total_nan}")
            # Заполняем оставшиеся NaN
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        
        logger.info(f"{name} после исправления: {df.shape}")
        return df
    
    # Исправляем данные
    train_fixed = fix_dataframe_types(train_df, "train")
    val_fixed = fix_dataframe_types(val_df, "validation")
    
    # Сохраняем исправленные данные
    train_fixed.to_csv('data/processed/train.csv', index=False)
    val_fixed.to_csv('data/processed/validation.csv', index=False)
    
    logger.info("✅ Типы данных исправлены")
    
    return train_fixed, val_fixed

if __name__ == "__main__":
    train_fixed, val_fixed = fix_data_types()
    
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Train: {len(train_fixed)} строк, {len(train_fixed.columns)} колонок")
    print(f"   Validation: {len(val_fixed)} строк, {len(val_fixed.columns)} колонок")
    print(f"   Типы данных в train: {train_fixed.dtypes.value_counts().to_dict()}")
