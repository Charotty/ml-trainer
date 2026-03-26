#!/usr/bin/env python3
"""
Скрипт подготовки данных для ML модели предсказания смещения почек.

Выполняет:
1. Чтение всех файлов с данными
2. Объединение в единый датасет
3. Очистку данных (пропуски, дубликаты, выбросы)
4. Feature engineering
5. Разделение на train/val/test
6. Сохранение обработанных данных

Автор: AR Kidney ML Project
Версия: 1.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KidneyDatasetPreparator:
    """Класс для подготовки датасета с данными о смещении почек."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Инициализация.
        
        Args:
            input_dir: Директория с исходными файлами
            output_dir: Директория для сохранения результатов
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'files_processed': [],
            'total_rows_before': 0,
            'total_rows_after': 0,
            'duplicates_removed': 0,
            'rows_with_missing_coords': 0,
            'outliers_removed': 0
        }
    
    def _normalize_name(self, value: object) -> str:
        if value is None:
            return ''
        s = str(value).strip().lower()
        s = s.replace('ё', 'е')
        s = re.sub(r'\s+', '', s)
        s = re.sub(r'[^a-zа-я0-9]+', '', s)
        return s

    def merge_dicom_features(self, df: pd.DataFrame, dicom_features_path: Path) -> pd.DataFrame:
        if not dicom_features_path.exists():
            logger.info(f"DICOM признаки не найдены: {dicom_features_path.name} (пропускаем)")
            return df

        if 'ФИО' not in df.columns:
            logger.info("Колонка ФИО не найдена, невозможно объединить DICOM признаки")
            return df

        try:
            dicom_df = pd.read_csv(dicom_features_path)
        except Exception as e:
            logger.info(f"Не удалось прочитать DICOM признаки ({dicom_features_path.name}): {e}")
            return df

        if 'full_name_key' not in dicom_df.columns:
            if 'full_name' in dicom_df.columns:
                dicom_df['full_name_key'] = dicom_df['full_name'].apply(self._normalize_name)
            else:
                logger.info("В DICOM признаках нет full_name/full_name_key, пропускаем объединение")
                return df

        result = df.copy()
        result['fio_key'] = result['ФИО'].apply(self._normalize_name)

        dicom_feature_cols = [
            c for c in dicom_df.columns
            if c not in {'dicom_folder', 'full_name', 'full_name_key', 'dicom_folder_key'}
        ]

        dicom_df = dicom_df[['full_name_key'] + dicom_feature_cols].drop_duplicates(subset=['full_name_key'])

        before_cols = set(result.columns)
        result = result.merge(
            dicom_df,
            how='left',
            left_on='fio_key',
            right_on='full_name_key'
        )

        result = result.drop(columns=[c for c in ['fio_key', 'full_name_key'] if c in result.columns])

        added_cols = [c for c in result.columns if c not in before_cols]
        if added_cols:
            logger.info(f"✅ Добавлены DICOM признаки: {len(added_cols)}")

        return result
    
    def read_file_smart(self, file_path: Path) -> pd.DataFrame:
        """
        Умное чтение файла с автоопределением заголовка.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            DataFrame с данными
        """
        logger.info(f"Чтение файла: {file_path.name}")
        
        suffix = file_path.suffix.lower()
        if suffix in {'.csv', '.txt'}:
            # CSV может иметь много строк "шапки" и ; разделитель
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                preview_lines = [next(f, '') for _ in range(20)]

            header_row = 0
            best_score = -1
            for i, line in enumerate(preview_lines):
                s = line.strip().lower()
                if not s:
                    continue
                score = 0
                if 'фио' in s:
                    score += 2
                if 'пол' in s:
                    score += 1
                if 'возраст' in s:
                    score += 1
                if ';' in s:
                    score += 1
                if score > best_score:
                    best_score = score
                    header_row = i

            if best_score >= 2:
                logger.info(f"  Заголовок найден в строке {header_row}")
            else:
                logger.info("  Заголовок не распознан, используем строку 0")

            df = pd.read_csv(
                file_path,
                sep=';',
                header=header_row,
                decimal=',',
                engine='python'
            )
        else:
            # Читаем первые 10 строк для поиска заголовка
            df_preview = pd.read_excel(file_path, header=None, nrows=10)

            # Ищем строку с заголовками
            header_row = 0
            for i in range(7):
                row_str = ' '.join([str(v).lower() for v in df_preview.iloc[i] if pd.notna(v)])
                if 'фио' in row_str or ('пол' in row_str and 'возраст' in row_str):
                    header_row = i
                    logger.info(f"  Заголовок найден в строке {i}")
                    break

            # Читаем файл с найденным заголовком
            df = pd.read_excel(file_path, header=header_row)
        
        # Удаляем полностью пустые строки
        df = df.dropna(how='all')
        
        # Удаляем строки где почти всё пусто (< 3 непустых значений)
        df = df[df.notna().sum(axis=1) > 3]
        
        logger.info(f"  Загружено строк: {len(df)}")
        logger.info(f"  Колонок: {len(df.columns)}")

        # Нормализуем "пустые" колонки после многострочной шапки
        unnamed_cols = [c for c in df.columns if str(c).lower().startswith('unnamed')]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        return df
    
    def combine_files(self, file_paths: List[Path]) -> pd.DataFrame:
        """
        Объединяет несколько файлов в один DataFrame.
        
        Args:
            file_paths: Список путей к файлам
            
        Returns:
            Объединённый DataFrame
        """
        logger.info(f"\n{'='*80}")
        logger.info("ШАГ 1: ОБЪЕДИНЕНИЕ ФАЙЛОВ")
        logger.info(f"{'='*80}\n")
        
        dfs = []
        
        for file_path in file_paths:
            try:
                df = self.read_file_smart(file_path)
                
                # Добавляем информацию об источнике
                df['source_file'] = file_path.name
                
                dfs.append(df)
                
                self.stats['files_processed'].append({
                    'filename': file_path.name,
                    'rows': len(df),
                    'columns': len(df.columns)
                })
                
            except Exception as e:
                logger.error(f"Ошибка при чтении {file_path.name}: {e}")
                continue
        
        if not dfs:
            raise ValueError("Не удалось прочитать ни одного файла!")
        
        # Объединяем все DataFrame
        combined = pd.concat(dfs, ignore_index=True, sort=False)
        
        self.stats['total_rows_before'] = len(combined)
        logger.info(f"\n✅ Объединено {len(dfs)} файлов")
        logger.info(f"   Всего строк: {len(combined)}")
        logger.info(f"   Всего колонок: {len(combined.columns)}")
        
        return combined
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка данных: дубликаты, пропуски, выбросы.
        
        Args:
            df: Исходный DataFrame
            
        Returns:
            Очищенный DataFrame
        """
        logger.info(f"\n{'='*80}")
        logger.info("ШАГ 2: ОЧИСТКА ДАННЫХ")
        logger.info(f"{'='*80}\n")
        
        initial_rows = len(df)
        
        # 1. Удаление дубликатов по ФИО
        if 'ФИО' in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=['ФИО'], keep='first')
            duplicates = before - len(df)
            self.stats['duplicates_removed'] = duplicates
            if duplicates > 0:
                logger.info(f"✅ Удалено дубликатов по ФИО: {duplicates}")
        
        # 2. Находим колонки с координатами
        coord_cols = [c for c in df.columns 
                     if any(k in str(c).lower() for k in ['ось x', 'ось y', 'ось z', 'x (мм)', 'y (мм)', 'z (мм)'])]
        
        logger.info(f"   Найдено колонок с координатами: {len(coord_cols)}")
        
        # 3. Удаляем строки где > 50% координат пропущены
        if coord_cols:
            before = len(df)
            missing_threshold = len(coord_cols) * 0.5
            
            df['missing_coords_count'] = df[coord_cols].isna().sum(axis=1)
            df = df[df['missing_coords_count'] <= missing_threshold]
            df = df.drop('missing_coords_count', axis=1)
            
            removed = before - len(df)
            self.stats['rows_with_missing_coords'] = removed
            if removed > 0:
                logger.info(f"✅ Удалено строк с >50% пропусков в координатах: {removed}")
        
        # 4. Удаление выбросов в числовых колонках
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numeric_cols:
            if col in coord_cols or 'δ' in str(col) or 'delta' in str(col).lower():
                before = len(df)
                mean = df[col].mean()
                std = df[col].std()
                
                if pd.notna(std) and std > 0:
                    # Удаляем значения > 3 стандартных отклонений
                    df = df[np.abs(df[col] - mean) <= 3 * std]
                    outliers_removed += (before - len(df))
        
        if outliers_removed > 0:
            self.stats['outliers_removed'] = outliers_removed
            logger.info(f"✅ Удалено выбросов (>3σ): {outliers_removed}")
        
        logger.info(f"\n📊 Результат очистки:")
        logger.info(f"   Было строк: {initial_rows}")
        logger.info(f"   Стало строк: {len(df)}")
        logger.info(f"   Удалено: {initial_rows - len(df)} ({(initial_rows - len(df))/initial_rows*100:.1f}%)")
        
        return df
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Стандартизация названий колонок и структуры данных.
        
        Args:
            df: DataFrame с исходными колонками
            
        Returns:
            DataFrame со стандартизированными колонками
        """
        logger.info(f"\n{'='*80}")
        logger.info("ШАГ 3: СТАНДАРТИЗАЦИЯ КОЛОНОК")
        logger.info(f"{'='*80}\n")
        
        # Создаем новый DataFrame для стандартизированных данных
        result = pd.DataFrame()
        
        # 1. Демографические данные
        logger.info("Обработка демографии...")
        
        # Пол
        if 'Пол' in df.columns:
            result['sex'] = df['Пол'].map({'м': 1, 'М': 1, 'ж': 0, 'Ж': 0})
            logger.info("  ✅ Пол → sex (М=1, Ж=0)")
        
        # Возраст
        if 'Возраст' in df.columns:
            result['age'] = pd.to_numeric(df['Возраст'], errors='coerce')
            logger.info("  ✅ Возраст → age")
        
        # ИМТ
        bmi_col = None
        for col in df.columns:
            if 'имт' in str(col).lower() or 'бми' in str(col).lower():
                bmi_col = col
                break
        
        if bmi_col:
            result['bmi'] = pd.to_numeric(df[bmi_col], errors='coerce')
            logger.info(f"  ✅ {bmi_col} → bmi")
        
        # Телосложение
        if 'Телосложение' in df.columns:
            body_type_map = {
                'норма': 0, 'Норма': 0, 'нормостеническое': 0,
                'астеническое': 1, 'астеник': 1, 'Астеническое': 1,
                'гиперстеническое': 2, 'гипер': 2, 'Гиперстеническое': 2
            }
            result['body_type'] = df['Телосложение'].map(body_type_map)
            logger.info("  ✅ Телосложение → body_type (норма=0, астеник=1, гипер=2)")
        
        # 2. Координаты почек
        logger.info("\nОбработка координат...")
        
        # Определяем какая почка (правая/левая)
        # Для простоты берём правую почку, если есть обе
        
        coord_mapping = self._create_coordinate_mapping(df)
        
        for target_col, source_cols in coord_mapping.items():
            if source_cols:
                # Берём первую найденную колонку
                result[target_col] = pd.to_numeric(df[source_cols[0]], errors='coerce')
        
        logger.info(f"  ✅ Создано {len(coord_mapping)} колонок с координатами")
        
        # 3. Вычисляем смещения (deltas) если их нет
        logger.info("\nВычисление смещений (deltas)...")
        
        delta_cols_created = 0
        for point in ['upper', 'middle', 'lower']:
            for axis in ['X', 'Y', 'Z']:
                delta_col = f'delta_{axis}_{point}'
                supine_col = f'{axis}_{point}_supine'
                lateral_col = f'{axis}_{point}_lateral'
                
                if supine_col in result.columns and lateral_col in result.columns:
                    result[delta_col] = result[lateral_col] - result[supine_col]
                    delta_cols_created += 1
        
        logger.info(f"  ✅ Вычислено {delta_cols_created} колонок со смещениями")
        
        # Сохраняем идентификатор пациента
        if 'ФИО' in df.columns:
            result['ФИО'] = df['ФИО'].astype(str)
        
        # Сохраняем источник данных
        result['source_file'] = df['source_file']
        
        logger.info(f"\n📊 Итого колонок в стандартизированном датасете: {len(result.columns)}")
        
        return result
    
    def _create_coordinate_mapping(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Создаёт маппинг колонок с координатами.
        
        Returns:
            Словарь {стандартное_название: [список_возможных_источников]}
        """
        mapping = {
            # Верхняя треть - на спине
            'X_upper_supine': [],
            'Y_upper_supine': [],
            'Z_upper_supine': [],
            # Средняя треть - на спине
            'X_middle_supine': [],
            'Y_middle_supine': [],
            'Z_middle_supine': [],
            # Нижняя треть - на спине
            'X_lower_supine': [],
            'Y_lower_supine': [],
            'Z_lower_supine': [],
            # Верхняя треть - на боку
            'X_upper_lateral': [],
            'Y_upper_lateral': [],
            'Z_upper_lateral': [],
            # Средняя треть - на боку
            'X_middle_lateral': [],
            'Y_middle_lateral': [],
            'Z_middle_lateral': [],
            # Нижняя треть - на боку
            'X_lower_lateral': [],
            'Y_lower_lateral': [],
            'Z_lower_lateral': [],
        }
        
        # Находим колонки с координатами
        coord_cols = [c for c in df.columns 
                     if any(k in str(c).lower() for k in ['ось x', 'ось y', 'ось z'])]

        # Специальный устойчивый разбор для файлов типа "Выборка - 50.csv":
        # там координаты идут плотными блоками, а названия осей повторяются.
        # В таких файлах pandas обычно делает имена: "Ось Х (мм)", "Ось Х (мм).1", ...
        # Берём первые 6 колонок каждой оси в порядке появления:
        #   0..2 -> supine (upper/middle/lower)
        #   3..5 -> lateral (upper/middle/lower)
        try:
            def _axis_cols(axis_letters: Tuple[str, ...]) -> List[str]:
                cols = []
                for c in df.columns:
                    s = str(c).lower()
                    if 'ось' not in s:
                        continue
                    if any(a in s for a in axis_letters):
                        cols.append(c)
                return cols

            x_cols = _axis_cols(('x', 'х'))
            y_cols = _axis_cols(('y', 'у'))
            z_cols = _axis_cols(('z', 'з'))

            if len(x_cols) >= 6 and len(y_cols) >= 6 and len(z_cols) >= 6:
                x6, y6, z6 = x_cols[:6], y_cols[:6], z_cols[:6]

                mapping['X_upper_supine'] = [x6[0]]
                mapping['Y_upper_supine'] = [y6[0]]
                mapping['Z_upper_supine'] = [z6[0]]
                mapping['X_middle_supine'] = [x6[1]]
                mapping['Y_middle_supine'] = [y6[1]]
                mapping['Z_middle_supine'] = [z6[1]]
                mapping['X_lower_supine'] = [x6[2]]
                mapping['Y_lower_supine'] = [y6[2]]
                mapping['Z_lower_supine'] = [z6[2]]

                mapping['X_upper_lateral'] = [x6[3]]
                mapping['Y_upper_lateral'] = [y6[3]]
                mapping['Z_upper_lateral'] = [z6[3]]
                mapping['X_middle_lateral'] = [x6[4]]
                mapping['Y_middle_lateral'] = [y6[4]]
                mapping['Z_middle_lateral'] = [z6[4]]
                mapping['X_lower_lateral'] = [x6[5]]
                mapping['Y_lower_lateral'] = [y6[5]]
                mapping['Z_lower_lateral'] = [z6[5]]

                return mapping
        except Exception:
            pass
        
        # Группируем по осям и позициям
        # Паттерны: "Ось X (мм)", "Ось X (мм).1", "Ось X (мм).2" и т.д.
        # .0 или без суффикса = на спине, .1+ = на боку
        
        for col in coord_cols:
            col_str = str(col).lower()
            
            # Определяем ось
            if 'x' in col_str or 'х' in col_str:
                axis = 'X'
            elif 'y' in col_str or 'у' in col_str:
                axis = 'Y'
            elif 'z' in col_str or 'з' in col_str:
                axis = 'Z'
            else:
                continue
            
            # Определяем позицию (на спине / на боку)
            # Если есть .1, .2, .3 после названия - это на боку
            is_lateral = '.1' in str(col) or '.2' in str(col) or '.3' in str(col) or \
                        '.4' in str(col) or '.5' in str(col)
            
            position = 'lateral' if is_lateral else 'supine'
            
            # Определяем часть почки (верх/середина/низ)
            # По порядку появления в файле: первая = верх, вторая = середина, третья = низ
            # Используем суффикс .0, .1, .2 для разных точек
            
            if '.0' in str(col) or (not any(s in str(col) for s in ['.1', '.2', '.3', '.4', '.5'])):
                # Первая встреча этой оси в этой позиции
                point = 'upper'
            elif '.1' in str(col) and not is_lateral:
                point = 'middle'
            elif '.2' in str(col) and not is_lateral:
                point = 'lower'
            elif is_lateral:
                # Для lateral считаем .1, .2, .3
                if str(col).count('.') == 1:
                    point = 'upper'
                elif str(col).count('.') == 2:
                    point = 'middle'
                else:
                    point = 'lower'
            else:
                point = 'upper'  # по умолчанию
            
            # Формируем ключ
            key = f'{axis}_{point}_{position}'
            
            if key in mapping:
                mapping[key].append(col)
        
        return mapping
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering: создание новых признаков.
        
        Args:
            df: DataFrame с базовыми признаками
            
        Returns:
            DataFrame с дополнительными признаками
        """
        logger.info(f"\n{'='*80}")
        logger.info("ШАГ 4: FEATURE ENGINEERING")
        logger.info(f"{'='*80}\n")
        
        result = df.copy()
        features_created = 0
        
        # 1. Длина почки (на спине)
        try:
            if all(col in result.columns for col in ['Z_upper_supine', 'Z_lower_supine']):
                result['kidney_length_supine'] = np.abs(
                    result['Z_upper_supine'] - result['Z_lower_supine']
                )
                features_created += 1
                logger.info("  ✅ kidney_length_supine")
        except:
            pass
        
        # 2. Центроид почки (на спине)
        try:
            if all(col in result.columns for col in ['X_upper_supine', 'X_middle_supine', 'X_lower_supine']):
                result['X_centroid_supine'] = (
                    result['X_upper_supine'] + 
                    result['X_middle_supine'] + 
                    result['X_lower_supine']
                ) / 3
                features_created += 1
                logger.info("  ✅ X_centroid_supine")
            
            if all(col in result.columns for col in ['Y_upper_supine', 'Y_middle_supine', 'Y_lower_supine']):
                result['Y_centroid_supine'] = (
                    result['Y_upper_supine'] + 
                    result['Y_middle_supine'] + 
                    result['Y_lower_supine']
                ) / 3
                features_created += 1
                logger.info("  ✅ Y_centroid_supine")
            
            if all(col in result.columns for col in ['Z_upper_supine', 'Z_middle_supine', 'Z_lower_supine']):
                result['Z_centroid_supine'] = (
                    result['Z_upper_supine'] + 
                    result['Z_middle_supine'] + 
                    result['Z_lower_supine']
                ) / 3
                features_created += 1
                logger.info("  ✅ Z_centroid_supine")
        except:
            pass
        
        # 3. Вектор ориентации почки (угол наклона)
        try:
            if all(col in result.columns for col in ['X_upper_supine', 'X_lower_supine', 
                                                      'Y_upper_supine', 'Y_lower_supine']):
                dx = result['X_upper_supine'] - result['X_lower_supine']
                dy = result['Y_upper_supine'] - result['Y_lower_supine']
                result['kidney_angle_xy'] = np.arctan2(dy, dx) * 180 / np.pi
                features_created += 1
                logger.info("  ✅ kidney_angle_xy")
        except:
            pass
        
        logger.info(f"\n📊 Создано новых признаков: {features_created}")
        
        return result
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Заполнение пропущенных значений.
        
        Args:
            df: DataFrame с пропусками
            
        Returns:
            DataFrame с заполненными пропусками
        """
        logger.info(f"\n{'='*80}")
        logger.info("ШАГ 5: ЗАПОЛНЕНИЕ ПРОПУСКОВ")
        logger.info(f"{'='*80}\n")
        
        result = df.copy()
        
        # Демография - заполняем медианой
        demographic_cols = ['age', 'bmi', 'body_type']
        
        for col in demographic_cols:
            if col in result.columns:
                missing_before = result[col].isna().sum()
                if missing_before > 0:
                    median_val = result[col].median()
                    if pd.notna(median_val):
                        result[col] = result[col].fillna(median_val)
                        logger.info(f"  ✅ {col}: заполнено {missing_before} пропусков (медиана={median_val:.1f})")
                    else:
                        logger.info(f"  ⚠️  {col}: медиана NaN, пропуски не заполнены")
        
        # Координаты - НЕ заполняем (лучше удалить строку)
        # Но показываем статистику
        coord_cols = [c for c in result.columns if any(k in c for k in ['_supine', '_lateral', 'delta_'])]
        
        total_missing_coords = sum(result[col].isna().sum() for col in coord_cols if col in result.columns)
        
        if total_missing_coords > 0:
            logger.info(f"\n  ⚠️  Осталось пропусков в координатах: {total_missing_coords}")
            logger.info(f"     (Эти строки будут удалены перед обучением)")
        
        return result
    
    def split_dataset(self, df: pd.DataFrame, 
                     train_ratio: float = 0.64,
                     val_ratio: float = 0.16,
                     test_ratio: float = 0.20,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделение датасета на train/validation/test.
        
        Args:
            df: Полный датасет
            train_ratio: Доля train
            val_ratio: Доля validation
            test_ratio: Доля test
            random_state: Random seed
            
        Returns:
            (train_df, val_df, test_df)
        """
        logger.info(f"\n{'='*80}")
        logger.info("ШАГ 6: РАЗДЕЛЕНИЕ НА TRAIN/VAL/TEST")
        logger.info(f"{'='*80}\n")
        
        # Перемешиваем датасет
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        n = len(df_shuffled)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df_shuffled[:train_end]
        val_df = df_shuffled[train_end:val_end]
        test_df = df_shuffled[val_end:]
        
        logger.info(f"  Всего строк: {n}")
        logger.info(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
        logger.info(f"  Validation: {len(val_df)} ({len(val_df)/n*100:.1f}%)")
        logger.info(f"  Test: {len(test_df)} ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_datasets(self, train_df: pd.DataFrame, 
                     val_df: pd.DataFrame, 
                     test_df: pd.DataFrame):
        """
        Сохранение датасетов в файлы.
        
        Args:
            train_df: Train датасет
            val_df: Validation датасет
            test_df: Test датасет
        """
        logger.info(f"\n{'='*80}")
        logger.info("ШАГ 7: СОХРАНЕНИЕ ДАННЫХ")
        logger.info(f"{'='*80}\n")
        
        # Сохраняем CSV
        train_path = self.output_dir / 'train.csv'
        val_path = self.output_dir / 'validation.csv'
        test_path = self.output_dir / 'test.csv'
        
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        val_df.to_csv(val_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        logger.info(f"  ✅ {train_path}")
        logger.info(f"  ✅ {val_path}")
        logger.info(f"  ✅ {test_path}")

        # Автогенерация списков признаков/таргетов для обучения
        delta_targets = [c for c in train_df.columns if c.startswith('delta_')]
        if delta_targets:
            target_names = sorted(delta_targets)
        else:
            target_candidates = [c for c in train_df.columns if c in ['Y_upper_lateral', 'Z_upper_lateral']]
            target_names = target_candidates
        
        service_cols = {'source_file', 'ФИО'}
        feature_names = [c for c in train_df.columns if c not in set(target_names) and c not in service_cols]

        feature_names_path = self.output_dir / 'feature_names.json'
        target_names_path = self.output_dir / 'target_names.json'

        with open(feature_names_path, 'w', encoding='utf-8') as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)
        with open(target_names_path, 'w', encoding='utf-8') as f:
            json.dump(target_names, f, ensure_ascii=False, indent=2)

        logger.info(f"  ✅ {feature_names_path}")
        logger.info(f"  ✅ {target_names_path}")
        
        # Сохраняем метаданные
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'dataset_stats': {
                'total_patients': len(train_df) + len(val_df) + len(test_df),
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'num_features': len(train_df.columns),
                'feature_names': train_df.columns.tolist()
            },
            'processing_stats': self.stats,
            'column_info': {
                'demographics': [c for c in train_df.columns if c in ['sex', 'age', 'bmi', 'body_type']],
                'coordinates_supine': [c for c in train_df.columns if '_supine' in c],
                'coordinates_lateral': [c for c in train_df.columns if '_lateral' in c],
                'deltas': [c for c in train_df.columns if 'delta_' in c],
                'engineered': [c for c in train_df.columns if any(k in c for k in ['centroid', 'length', 'angle'])]
            }
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  ✅ {metadata_path}")
        
        # Сохраняем словарь данных
        data_dict = self._create_data_dictionary(train_df)
        dict_path = self.output_dir / 'DATA_DICTIONARY.md'
        
        with open(dict_path, 'w', encoding='utf-8') as f:
            f.write(data_dict)
        
        logger.info(f"  ✅ {dict_path}")
        
        self.stats['total_rows_after'] = len(train_df) + len(val_df) + len(test_df)
    
    def _create_data_dictionary(self, df: pd.DataFrame) -> str:
        """Создаёт описание датасета в формате Markdown."""
        
        lines = [
            "# DATA DICTIONARY",
            "",
            f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Описание датасета",
            "",
            f"- **Всего колонок**: {len(df.columns)}",
            f"- **Всего строк**: {len(df)}",
            "",
            "## Колонки",
            ""
        ]
        
        # Группируем колонки по типам
        col_groups = {
            'Демография': [c for c in df.columns if c in ['sex', 'age', 'bmi', 'body_type']],
            'Координаты (на спине)': [c for c in df.columns if '_supine' in c and 'delta' not in c],
            'Координаты (на боку)': [c for c in df.columns if '_lateral' in c and 'delta' not in c],
            'Смещения (delta)': [c for c in df.columns if 'delta_' in c],
            'Признаки (engineered)': [c for c in df.columns if any(k in c for k in ['centroid', 'length', 'angle'])],
            'Служебные': [c for c in df.columns if c in ['source_file', 'ФИО']]
        }
        
        for group_name, cols in col_groups.items():
            if cols:
                lines.append(f"### {group_name}")
                lines.append("")
                for col in cols:
                    dtype = str(df[col].dtype)
                    non_null = df[col].notna().sum()
                    lines.append(f"- **{col}** ({dtype}): {non_null}/{len(df)} непустых значений")
                lines.append("")
        
        # Описания колонок
        lines.extend([
            "## Значения категориальных переменных",
            "",
            "### sex (пол)",
            "- 0 = женский",
            "- 1 = мужской",
            "",
            "### body_type (телосложение)",
            "- 0 = нормостеническое",
            "- 1 = астеническое",
            "- 2 = гиперстеническое",
            "",
            "## Единицы измерения",
            "",
            "- Координаты (X, Y, Z): **миллиметры (мм)**",
            "- Возраст: **годы**",
            "- ИМТ (BMI): **кг/м²**",
            "- Углы: **градусы**",
            ""
        ])
        
        return '\n'.join(lines)
    
    def print_summary(self):
        """Выводит итоговую сводку."""
        logger.info(f"\n{'='*80}")
        logger.info("📊 ИТОГОВАЯ СВОДКА")
        logger.info(f"{'='*80}\n")
        
        logger.info(f"Обработано файлов: {len(self.stats['files_processed'])}")
        for file_info in self.stats['files_processed']:
            logger.info(f"  - {file_info['filename']}: {file_info['rows']} строк")
        
        logger.info(f"\nОчистка данных:")
        logger.info(f"  Исходных строк: {self.stats['total_rows_before']}")
        logger.info(f"  Удалено дубликатов: {self.stats['duplicates_removed']}")
        logger.info(f"  Удалено с пропусками: {self.stats['rows_with_missing_coords']}")
        logger.info(f"  Удалено выбросов: {self.stats['outliers_removed']}")
        logger.info(f"  Итоговых строк: {self.stats['total_rows_after']}")
        
        retention_rate = self.stats['total_rows_after'] / self.stats['total_rows_before'] * 100
        logger.info(f"\n  Сохранено данных: {retention_rate:.1f}%")
        
        logger.info(f"\n ГОТОВО! Датасет подготовлен для ML.")


def main():
    """Главная функция."""
    
    # Настройки
    INPUT_DIR = "."  # Текущая директория, где лежат Excel файлы
    OUTPUT_DIR = "data/processed"

    DICOM_FEATURES_FILE = "results.csv"
    
    # Список файлов для обработки
    files_to_process = [
        "Выборка - 50.csv",
        "Смещение доп - 5.xlsx", 
        "Смещение почек - 2.xlsx",
        "Смещение почек 15.01.xlsx"
        # "Смещение почек.xlsx" - пропускаем, т.к. пустой
    ]
    
    print("\n" + "="*80)
    print(" ПОДГОТОВКА ДАТАСЕТА ДЛЯ ML")
    print("="*80 + "\n")
    
    try:
        # Создаём препаратор
        preparator = KidneyDatasetPreparator(INPUT_DIR, OUTPUT_DIR)
        
        # Формируем полные пути к файлам
        file_paths = [Path(INPUT_DIR) / filename for filename in files_to_process]
        
        # 1. Объединяем файлы
        combined_df = preparator.combine_files(file_paths)
        
        # 2. Очищаем данные
        cleaned_df = preparator.clean_data(combined_df)
        
        # 3. Стандартизируем колонки
        standardized_df = preparator.standardize_columns(cleaned_df)

        # 3.1 Подмешиваем DICOM признаки (если есть)
        standardized_df = preparator.merge_dicom_features(
            standardized_df,
            Path(INPUT_DIR) / DICOM_FEATURES_FILE
        )
        
        # 4. Создаём новые признаки
        featured_df = preparator.create_features(standardized_df)
        
        # 5. Заполняем пропуски
        filled_df = preparator.fill_missing_values(featured_df)
        
        # 6. Разделяем на train/val/test
        train_df, val_df, test_df = preparator.split_dataset(filled_df)
        
        # 7. Сохраняем
        preparator.save_datasets(train_df, val_df, test_df)
        
        # 8. Выводим сводку
        preparator.print_summary()
        
        print("\n" + "="*80)
        print("✅ УСПЕШНО! Данные готовы для обучения ML-модели.")
        print("="*80)
        print(f"\n📁 Файлы сохранены в: {OUTPUT_DIR}/")
        print("   - train.csv")
        print("   - validation.csv")
        print("   - test.csv")
        print("   - metadata.json")
        print("   - DATA_DICTIONARY.md")
        print("\n🎯 Следующий шаг: Блок 2 - Обучение модели\n")
        
    except Exception as e:
        logger.error(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())