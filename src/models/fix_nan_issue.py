#!/usr/bin/env python3
"""
ИСПРАВЛЕНИЕ ПРОБЛЕМЫ С NaN В ДАННЫХ
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_nan_in_data():
    """Исправить NaN в train/validation данных"""
    logger.info("Загрузка данных...")
    
    # Загрузка текущих данных
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/validation.csv')
    
    logger.info(f"Train до очистки: {train_df.shape}")
    logger.info(f"Val до очистки: {val_df.shape}")
    
    # Анализ NaN
    train_nan = train_df.isnull().sum()
    val_nan = val_df.isnull().sum()
    
    logger.info(f"Колонки с NaN в train: {(train_nan > 0).sum()}")
    logger.info(f"Колонки с NaN в val: {(val_nan > 0).sum()}")
    
    # Очистка данных
    def clean_dataframe(df, name):
        logger.info(f"Очистка {name}...")
        
        # Удаляем колонки с >50% пропусков
        threshold = len(df) * 0.5
        cols_to_keep = []
        
        for col in df.columns:
            nan_count = df[col].isnull().sum()
            if nan_count <= threshold:
                cols_to_keep.append(col)
            else:
                logger.warning(f"Удалена колонка {col}: {nan_count/len(df)*100:.1f}% пропусков")
        
        df_clean = df[cols_to_keep].copy()
        
        # Заполняем оставшиеся NaN
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                logger.info(f"Заполнено NaN в {col}: медиана = {median_val}")
        
        # Проверяем что не осталось NaN
        remaining_nan = df_clean.isnull().sum().sum()
        if remaining_nan > 0:
            logger.error(f"Осталось NaN: {remaining_nan}")
            # Удаляем строки с NaN
            df_clean = df_clean.dropna()
            logger.info(f"Удалено строк с NaN: {len(df) - len(df_clean)}")
        
        logger.info(f"{name} после очистки: {df_clean.shape}")
        return df_clean
    
    # Очищаем данные
    train_clean = clean_dataframe(train_df, "train")
    val_clean = clean_dataframe(val_df, "validation")
    
    # Сохраняем очищенные данные
    train_clean.to_csv('data/processed/train_clean.csv', index=False)
    val_clean.to_csv('data/processed/validation_clean.csv', index=False)
    
    # Заменяем оригинальные файлы
    train_clean.to_csv('data/processed/train.csv', index=False)
    val_clean.to_csv('data/processed/validation.csv', index=False)
    
    logger.info("✅ Данные очищены и сохранены")
    
    return train_clean, val_clean

if __name__ == "__main__":
    train_clean, val_clean = fix_nan_in_data()
    
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Train: {len(train_clean)} строк, {len(train_clean.columns)} колонок")
    print(f"   Validation: {len(val_clean)} строк, {len(val_clean.columns)} колонок")
    print(f"   NaN в train: {train_clean.isnull().sum().sum()}")
    print(f"   NaN в validation: {val_clean.isnull().sum().sum()}")
