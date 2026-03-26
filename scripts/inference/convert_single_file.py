#!/usr/bin/env python3
"""
КОНВЕРТАЦИЯ ОДНОГО ФАЙЛА В CSV
Правильная обработка файла Выборка - 50.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SingleFileConverter:
    def __init__(self, excel_file_path, output_dir):
        self.excel_file_path = Path(excel_file_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_clean_data(self):
        """Загрузить и очистить данные"""
        logger.info(f"Загрузка файла: {self.excel_file_path.name}")
        
        # Читаем с правильным заголовком (строка 4)
        df = pd.read_excel(self.excel_file_path, header=4)
        
        # Удаляем пустые строки
        df = df.dropna(how='all')
        
        logger.info(f"Загружено строк: {len(df)}")
        logger.info(f"Колонок: {len(df.columns)}")
        
        # Показываем основные колонки
        key_columns = ['ФИО', 'Пол', 'Возраст', 'ИМТ']
        for col in key_columns:
            if col in df.columns:
                logger.info(f"  {col}: {df[col].count()} значений")
        
        return df
    
    def extract_coordinates(self, df):
        """Извлечь координаты почек"""
        logger.info("Извлечение координат...")
        
        # Ищем все колонки с координатами
        coord_columns = []
        for col in df.columns:
            col_str = str(col)
            if 'Ось' in col_str and ('Y' in col_str or 'Z' in col_str):
                coord_columns.append(col)
        
        logger.info(f"Найдено колонок с координатами: {len(coord_columns)}")
        logger.info(f"Координатные колонки: {coord_columns[:10]}...")
        
        # Определяем позиции по номерам
        # .1 - первая позиция (supine), .2 - вторая позиция (lateral)
        positions = {
            'supine': {},
            'lateral': {}
        }
        
        for col in coord_columns:
            # Определяем позицию по номеру
            if '.1' in col or col.endswith(' (мм)') or col == 'Ось Y' or col == 'Ось Z':
                position = 'supine'
            elif '.2' in col:
                position = 'lateral'
            else:
                continue  # Пропускаем остальные
            
            # Определяем ось
            if 'Y' in col:
                axis = 'Y'
            elif 'Z' in col:
                axis = 'Z'
            else:
                continue
            
            positions[position][axis] = col
        
        logger.info(f"Найденные позиции: {list(positions.keys())}")
        for pos, axes in positions.items():
            logger.info(f"  {pos}: {list(axes.keys())}")
        
        # Создаем новые колонки с координатами
        for position, axes in positions.items():
            for axis, col in axes.items():
                new_col_name = f"{axis}_{position}"
                df[new_col_name] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"  {new_col_name}: {df[new_col_name].count()} значений")
        
        return df, positions
    
    def extract_demographics(self, df):
        """Извлечь демографические данные"""
        logger.info("Извлечение демографических данных...")
        
        # Пол
        if 'Пол' in df.columns:
            df['sex'] = df['Пол'].map({'м': 1, 'ж': 0})
            logger.info(f"Пол: {df['sex'].count()} значений")
        
        # Возраст
        if 'Возраст' in df.columns:
            df['age'] = pd.to_numeric(df['Возраст'], errors='coerce')
            logger.info(f"Возраст: {df['age'].count()} значений")
        
        # ИМТ
        if 'ИМТ' in df.columns:
            df['bmi'] = pd.to_numeric(df['ИМТ'], errors='coerce')
            logger.info(f"ИМТ: {df['bmi'].count()} значений")
        
        # Телосложение
        if 'Телосложение' in df.columns:
            body_type_map = {
                'норма': 0,
                'нормостеническое': 0,
                'астеническое': 1,
                'астеническое ': 1,
                'гиперстеническое': 2,
                'гипер': 2
            }
            df['body_type'] = df['Телосложение'].map(body_type_map)
            logger.info(f"Телосложение: {df['body_type'].count()} значений")
        
        return df
    
    def create_target_variables(self, df):
        """Создать целевые переменные (смещения)"""
        logger.info("Создание целевых переменных...")
        
        # Целевые переменные - координаты в lateral позиции
        target_cols = []
        for col in df.columns:
            if col.endswith('_lateral') and ('Y_' in col or 'Z_' in col):
                target_cols.append(col)
        
        logger.info(f"Целевые колонки: {target_cols}")
        
        # Переименовываем для удобства
        rename_map = {}
        for col in target_cols:
            if 'Y_lateral' in col:
                rename_map[col] = 'Y_upper_lateral'
            elif 'Z_lateral' in col:
                rename_map[col] = 'Z_upper_lateral'
        
        df = df.rename(columns=rename_map)
        
        return df
    
    def create_features(self, df):
        """Создать признаки для модели"""
        logger.info("Создание признаков...")
        
        # Признаки - демографические данные и координаты в supine
        feature_cols = []
        
        # Демографические признаки
        demo_features = ['sex', 'age', 'bmi', 'body_type']
        for col in demo_features:
            if col in df.columns:
                feature_cols.append(col)
        
        # Координаты в supine позиции
        for col in df.columns:
            if col.endswith('_supine') and ('Y_' in col or 'Z_' in col):
                feature_cols.append(col)
        
        logger.info(f"Признаки: {feature_cols}")
        
        return df, feature_cols
    
    def clean_final_data(self, df, feature_cols, target_cols):
        """Финальная очистка данных"""
        logger.info("Финальная очистка данных...")
        
        # Собираем все нужные колонки
        all_cols = feature_cols + target_cols + ['ФИО']
        df_clean = df[all_cols].copy()
        
        # Удаляем строки где нет целевых переменных
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=target_cols, how='any')
        final_count = len(df_clean)
        
        logger.info(f"Удалено строк без целевых переменных: {initial_count - final_count}")
        
        # Заполняем пропуски в признаках медианой
        for col in feature_cols:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                logger.info(f"  Заполнено пропусков в {col}: {df_clean[col].isna().sum()}")
        
        return df_clean
    
    def split_data(self, df):
        """Разделить данные на train/validation/test"""
        logger.info("Разделение данных...")
        
        # Перемешиваем данные
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Разделение 70%/15%/15%
        train_df, temp_df = train_test_split(df_shuffled, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        logger.info(f"Train: {len(train_df)} строк")
        logger.info(f"Validation: {len(val_df)} строк")
        logger.info(f"Test: {len(test_df)} строк")
        
        return train_df, val_df, test_df
    
    def save_data(self, train_df, val_df, test_df, feature_cols, target_cols):
        """Сохранить данные и метаданные"""
        logger.info("Сохранение данных...")
        
        # Сохраняем CSV файлы
        train_df.to_csv(self.output_dir / 'train.csv', index=False)
        val_df.to_csv(self.output_dir / 'validation.csv', index=False)
        test_df.to_csv(self.output_dir / 'test.csv', index=False)
        
        # Сохраняем метаданные
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'source_file': self.excel_file_path.name,
            'total_patients': len(train_df) + len(val_df) + len(test_df),
            'train_patients': len(train_df),
            'validation_patients': len(val_df),
            'test_patients': len(test_df),
            'features': feature_cols,
            'targets': target_cols,
            'feature_count': len(feature_cols),
            'target_count': len(target_cols)
        }
        
        with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Сохраняем названия признаков и целей
        with open(self.output_dir / 'feature_names.json', 'w', encoding='utf-8') as f:
            json.dump(feature_cols, f, indent=2)
        
        with open(self.output_dir / 'target_names.json', 'w', encoding='utf-8') as f:
            json.dump(target_cols, f, indent=2)
        
        logger.info("✅ Данные сохранены")
        
        return metadata
    
    def convert(self):
        """Основной метод конвертации"""
        logger.info("НАЧАЛО КОНВЕРТАЦИИ ДАННЫХ")
        logger.info("="*50)
        
        try:
            # 1. Загрузка данных
            df = self.load_and_clean_data()
            
            # 2. Извлечение координат
            df, positions = self.extract_coordinates(df)
            
            # 3. Извлечение демографии
            df = self.extract_demographics(df)
            
            # 4. Создание целевых переменных
            df = self.create_target_variables(df)
            
            # 5. Создание признаков
            df, feature_cols = self.create_features(df)
            
            # 6. Определение целевых переменных
            target_cols = ['Y_upper_lateral', 'Z_upper_lateral']
            target_cols = [col for col in target_cols if col in df.columns]
            
            # 7. Финальная очистка
            df_clean = self.clean_final_data(df, feature_cols, target_cols)
            
            # 8. Разделение данных
            train_df, val_df, test_df = self.split_data(df_clean)
            
            # 9. Сохранение
            metadata = self.save_data(train_df, val_df, test_df, feature_cols, target_cols)
            
            # 10. Вывод статистики
            logger.info("="*50)
            logger.info("СТАТИСТИКА КОНВЕРТАЦИИ")
            logger.info("="*50)
            logger.info(f"Исходный файл: {self.excel_file_path.name}")
            logger.info(f"Всего пациентов: {metadata['total_patients']}")
            logger.info(f"Признаков: {metadata['feature_count']}")
            logger.info(f"Целевых переменных: {metadata['target_count']}")
            logger.info(f"Train: {metadata['train_patients']}")
            logger.info(f"Validation: {metadata['validation_patients']}")
            logger.info(f"Test: {metadata['test_patients']}")
            
            logger.info("\n✅ КОНВЕРТАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            raise

def main():
    """Главная функция"""
    converter = SingleFileConverter(
        excel_file_path="Выборка - 50.xlsx",
        output_dir="data/processed"
    )
    
    converter.convert()

if __name__ == "__main__":
    main()
