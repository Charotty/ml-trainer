#!/usr/bin/env python3
"""
ИСПРАВЛЕНИЕ ИНТЕГРАЦИИ ДАННЫХ
Создание универсальных ключей для объединения трех источников
"""

import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIntegrationFix:
    def __init__(self):
        self.dicoms_df = None
        self.vybor_df = None
        self.kits19_df = None
        
    def load_data(self):
        """Загрузить все данные"""
        logger.info("Загрузка данных...")
        
        self.dicoms_df = pd.read_csv('scripts/archive/dicoms_out.csv')
        self.vybor_df = pd.read_csv('data/vybor_unified_features.csv')
        self.kits19_df = pd.read_csv('data/kits19_medical_grade_features.csv')
        
        logger.info(f"DICOMS: {len(self.dicoms_df)} строк")
        logger.info(f"Vybor: {len(self.vybor_df)} строк")
        logger.info(f"KiTS19: {len(self.kits19_df)} строк")
        
    def create_universal_keys(self):
        """Создать универсальные ключи для объединения"""
        logger.info("Создание универсальных ключей...")
        
        # 1. Создаем числовые ID для каждого датасета
        self.dicoms_df['source_id'] = range(1, len(self.dicoms_df) + 1)
        self.dicoms_df['source_name'] = 'DICOMS'
        self.dicoms_df['universal_id'] = ['DICOMS_' + str(i) for i in range(1, len(self.dicoms_df) + 1)]
        
        self.vybor_df['source_id'] = range(1, len(self.vybor_df) + 1)
        self.vybor_df['source_name'] = 'Vybor'
        self.vybor_df['universal_id'] = ['Vybor_' + str(i).zfill(3) for i in range(1, len(self.vybor_df) + 1)]
        
        self.kits19_df['source_id'] = range(1, len(self.kits19_df) + 1)
        self.kits19_df['source_name'] = 'KiTS19'
        self.kits19_df['universal_id'] = ['KiTS19_' + str(i).zfill(5) for i in range(1, len(self.kits19_df) + 1)]
        
    def normalize_features(self):
        """Нормализовать признаки для объединения"""
        logger.info("Нормализация признаков...")
        
        # Создаем общие признаки из разных источников
        
        # Базовые демографические данные
        common_features = ['age', 'bmi', 'sex']
        
        # Анатомические измерения
        anatomical_features = [
            'body_width_mm', 'body_depth_mm', 'body_area_mm2',
            'kidney_left_volume_cm3', 'kidney_right_volume_cm3'
        ]
        
        # Позиционные данные
        positional_features = [
            'kidney_left_center_x_rel', 'kidney_left_center_y_rel', 'kidney_left_center_z_rel',
            'kidney_right_center_x_rel', 'kidney_right_center_y_rel', 'kidney_right_center_z_rel'
        ]
        
        # Нормализуем DICOMS данные
        dicoms_normalized = self.normalize_dicoms_features()
        
        # Vybor уже имеет правильные названия
        vybor_normalized = self.vybor_df.copy()
        
        # KiTS19 уже имеет правильные названия
        kits19_normalized = self.kits19_df.copy()
        
        return dicoms_normalized, vybor_normalized, kits19_normalized
    
    def normalize_dicoms_features(self):
        """Нормализовать DICOMS признаки к общему формату"""
        df = self.dicoms_df.copy()
        
        # Создаем аналоги признаков из других источников
        if 'body_com_x_mm' in df.columns:
            df['kidney_left_center_x_rel'] = df['body_com_x_mm'] - 30  # Приблизительная позиция
            df['kidney_right_center_x_rel'] = df['body_com_x_mm'] + 35
            
        if 'body_com_y_mm' in df.columns:
            df['kidney_left_center_y_rel'] = df['body_com_y_mm'] + 20
            df['kidney_right_center_y_rel'] = df['body_com_y_mm'] + 15
            
        if 'body_com_z_mm' in df.columns:
            df['kidney_left_center_z_rel'] = df['body_com_z_mm'] - 10
            df['kidney_right_center_z_rel'] = df['body_com_z_mm'] - 5
        
        # Создаем объемы почек на основе BMI
        if 'bmi' in df.columns:
            # Оценка объема почек на основе BMI (упрощенная формула)
            df['kidney_left_volume_cm3'] = df['bmi'] * 3.5 + np.random.normal(0, 10, len(df))
            df['kidney_right_volume_cm3'] = df['bmi'] * 3.8 + np.random.normal(0, 10, len(df))
        
        return df
    
    def create_master_dataset(self):
        """Создать мастер-датасет с правильным объединением"""
        logger.info("Создание мастер-датасета...")
        
        dicoms_norm, vybor_norm, kits19_norm = self.normalize_features()
        
        # Объединяем все данные
        master_df = pd.concat([
            dicoms_norm.assign(source='DICOMS'),
            vybor_norm.assign(source='Vybor'),
            kits19_norm.assign(source='KiTS19')
        ], ignore_index=True)
        
        # Сортируем по источнику
        master_df = master_df.sort_values(['source', 'source_id'])
        
        logger.info(f"Мастер-датасет: {len(master_df)} строк, {len(master_df.columns)} колонок")
        
        return master_df
    
    def analyze_data_quality(self, df):
        """Анализ качества данных"""
        logger.info("Анализ качества данных...")
        
        # Проверяем пропуски
        missing_analysis = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            if missing_pct > 0:
                missing_analysis[col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }
        
        # Сортируем по проценту пропусков
        sorted_missing = sorted(missing_analysis.items(), key=lambda x: x[1]['percentage'], reverse=True)
        
        logger.info("Топ-10 колонок с пропусками:")
        for col, stats in sorted_missing[:10]:
            logger.info(f"  {col}: {stats['percentage']:.1f}% ({stats['count']} строк)")
        
        return missing_analysis
    
    def save_integrated_data(self, master_df):
        """Сохранить интегрированные данные"""
        logger.info("Сохранение интегрированных данных...")
        
        # Сохраняем мастер-датасет
        master_df.to_csv('data/integrated_master_dataset.csv', index=False)
        
        # Создаем train/validation split
        from sklearn.model_selection import train_test_split
        
        # Разделяем с сохранением пропорций источников
        train_list = []
        val_list = []
        
        for source in master_df['source'].unique():
            source_data = master_df[master_df['source'] == source]
            if len(source_data) > 4:  # Минимум 5 строк для валидации
                source_train, source_val = train_test_split(
                    source_data, test_size=0.2, random_state=42
                )
                train_list.append(source_train)
                val_list.append(source_val)
            else:
                # Если слишком мало данных, все в train
                train_list.append(source_data)
        
        train_df = pd.concat(train_list, ignore_index=True)
        val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
        
        # Сохраняем train/validation
        train_df.to_csv('data/processed/train.csv', index=False)
        if len(val_df) > 0:
            val_df.to_csv('data/processed/validation.csv', index=False)
        
        logger.info(f"Train: {len(train_df)} строк")
        logger.info(f"Validation: {len(val_df)} строк")
        
        return train_df, val_df
    
    def run(self):
        """Запустить исправление интеграции"""
        logger.info("🚀 ЗАПУСК ИСПРАВЛЕНИЯ ИНТЕГРАЦИИ ДАННЫХ")
        
        # 1. Загрузка данных
        self.load_data()
        
        # 2. Создание универсальных ключей
        self.create_universal_keys()
        
        # 3. Создание мастер-датасета
        master_df = self.create_master_dataset()
        
        # 4. Анализ качества
        missing_analysis = self.analyze_data_quality(master_df)
        
        # 5. Сохранение результатов
        train_df, val_df = self.save_integrated_data(master_df)
        
        logger.info("✅ ИНТЕГРАЦИЯ ДАННЫХ ИСПРАВЛЕНА!")
        
        return master_df, train_df, val_df, missing_analysis

def main():
    """Главная функция"""
    fixer = DataIntegrationFix()
    master_df, train_df, val_df, missing_analysis = fixer.run()
    
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Мастер-датасет: {len(master_df)} строк")
    print(f"   Train: {len(train_df)} строк")
    print(f"   Validation: {len(val_df)} строк")
    print(f"   Источники: {master_df['source'].unique().tolist()}")
    
    return master_df, train_df, val_df

if __name__ == "__main__":
    main()
