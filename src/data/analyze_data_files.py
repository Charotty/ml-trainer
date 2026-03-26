#!/usr/bin/env python3
"""
Анализатор медицинских данных для ML проекта AR Kidney Displacement.

Сканирует директорию с данными и создаёт подробный отчёт о пригодности
файлов для обучения ML-модели. Поддерживает как табличные данные (CSV/Excel),
так и DICOM изображения.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any
import warnings
import os
import re
warnings.filterwarnings('ignore')

# Попытка импорта pyarrow для pandas
try:
    import pyarrow
except ImportError:
    pyarrow = None
    print("Предупреждение: pyarrow не установлен, некоторые функции pandas могут работать медленнее")

class DataAnalyzer:
    """Анализатор данных для ML проекта."""
    
    def __init__(self, data_dir: str):
        """
        Инициализация анализатора.
        
        Args:
            data_dir: Путь к директории с данными
        """
        self.data_dir = Path(data_dir)
        self.report = {
            'scan_date': datetime.now().isoformat(),
            'data_directory': str(self.data_dir),
            'files_analyzed': [],
            'summary': {},
            'recommendations': []
        }
        
        # Ожидаемые колонки (русские названия)
        self.expected_columns_ru = {
            'demographics': ['пол', 'возраст', 'имт', 'телосложение'],
            'coordinates_supine': ['ось x', 'ось y', 'ось z', 'на спине'],
            'coordinates_lateral': ['на боку'],
            'displacement': ['δ', 'delta', 'смещение']
        }
        
    def scan_directory(self) -> Dict[str, List[Path]]:
        """
        Сканирует директорию и находит все файлы данных.
        
        Returns:
            Словарь с путями к найденным файлам по типам
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {self.data_dir}")
        
        result = {
            'tabular_files': [],
            'dicom_files': [],
            'presentation_files': [],
            'other_files': []
        }
        
        # Табличные данные
        tabular_extensions = ['*.csv', '*.xlsx', '*.xls']
        for ext in tabular_extensions:
            result['tabular_files'].extend(self.data_dir.glob(ext))
            result['tabular_files'].extend(self.data_dir.glob(f'**/{ext}'))
        
        # DICOM файлы (без расширения или с .dcm)
        for file_path in self.data_dir.glob('**/*'):
            if file_path.is_file():
                # DICOM файлы обычно не имеют расширения или .dcm
                if file_path.suffix.lower() in ['', '.dcm']:
                    # Проверяем, является ли файл DICOM по заголовку
                    if self._is_dicom_file(file_path):
                        result['dicom_files'].append(file_path)
                elif file_path.suffix.lower() in ['.pptx', '.pdf']:
                    result['presentation_files'].append(file_path)
                elif file_path.suffix.lower() not in ['.exe', '.inf', '.cds']:
                    result['other_files'].append(file_path)
        
        # Сортируем все списки
        for key in result:
            result[key] = sorted(result[key])
        
        return result
    
    def _is_dicom_file(self, file_path: Path) -> bool:
        """Проверяет, является ли файл DICOM по заголовку."""
        try:
            with open(file_path, 'rb') as f:
                # DICOM файлы начинаются с 128 байт пропуска, затем "DICM"
                f.seek(128)
                header = f.read(4)
                return header == b'DICM'
        except:
            return False
    
    def analyze_dicom_directory(self, directory_path: Path) -> Dict[str, Any]:
        """Анализирует директорию с DICOM файлами."""
        analysis = {
            'directory_name': directory_path.name,
            'directory_path': str(directory_path),
            'type': 'dicom_study',
            'status': 'unknown',
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Находим все DICOM файлы в директории
            dicom_files = []
            for file_path in directory_path.glob('*'):
                if file_path.is_file() and self._is_dicom_file(file_path):
                    dicom_files.append(file_path)
            
            if not dicom_files:
                analysis['status'] = 'no_dicom_files'
                analysis['errors'].append("DICOM файлы не найдены")
                return analysis
            
            analysis['details']['dicom_file_count'] = len(dicom_files)
            analysis['details']['total_size_mb'] = round(
                sum(f.stat().st_size for f in dicom_files) / (1024 * 1024), 2
            )
            analysis['details']['dicom_files'] = [f.name for f in dicom_files]
            
            # Анализ структуры директории
            subdirs = [d for d in directory_path.iterdir() if d.is_dir()]
            analysis['details']['subdirectories'] = [d.name for d in subdirs]
            analysis['details']['has_subdirs'] = len(subdirs) > 0
            
            # Проверяем наличие служебных файлов
            dicomdir = directory_path / 'DICOMDIR'
            analysis['details']['has_dicomdir'] = dicomdir.exists()
            
            # Извлекаем информацию из имени директории
            patient_info = self._extract_patient_info_from_name(directory_path.name)
            analysis['details']['extracted_patient_info'] = patient_info
            
            # Оцениваем пригодность для ML
            if len(dicom_files) >= 10:
                analysis['status'] = 'ready_for_processing'
                analysis['details']['ml_readiness'] = {
                    'is_suitable': True,
                    'quality_score': 0.8,
                    'estimated_slices': len(dicom_files),
                    'notes': 'Достаточное количество срезов для анализа'
                }
            else:
                analysis['status'] = 'insufficient_data'
                analysis['details']['ml_readiness'] = {
                    'is_suitable': False,
                    'quality_score': 0.3,
                    'estimated_slices': len(dicom_files),
                    'notes': 'Мало срезов для надежного анализа'
                }
            
        except Exception as e:
            analysis['status'] = 'error'
            analysis['errors'].append(f"Ошибка при анализе: {str(e)}")
        
        return analysis
    
    def _extract_patient_info_from_name(self, dir_name: str) -> Dict[str, Any]:
        """Извлекает информацию о пациенте из имени директории."""
        info = {
            'full_name': None,
            'date': None,
            'age': None,
            'notes': []
        }
        
        # Ищем ФИО (слова с большой буквы)
        words = re.findall(r'[А-Я][а-я]+', dir_name)
        if len(words) >= 2:
            info['full_name'] = ' '.join(words[:3])  # Фамилия Имя Отчество
        
        # Ищем дату
        date_patterns = [
            r'(\d{2}\.\d{2}\.\d{4})',  # DD.MM.YYYY
            r'(\d{1,2}\s+[а-я]+\s+\d{4})',  # 25 октября 2023
            r'(\d{6,8})'  # DDMMYY или DDMMYYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, dir_name)
            if match:
                info['date'] = match.group(1)
                break
        
        # Ищем возраст
        age_match = re.search(r'(\d+)\s*(?:лет|год|года)', dir_name, re.IGNORECASE)
        if age_match:
            info['age'] = int(age_match.group(1))
        
        return info
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Анализирует один файл.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Словарь с результатами анализа
        """
        analysis = {
            'filename': file_path.name,
            'filepath': str(file_path),
            'extension': file_path.suffix,
            'file_size_kb': round(file_path.stat().st_size / 1024, 2),
            'status': 'unknown',
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Загрузка данных
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                analysis['status'] = 'unsupported'
                analysis['errors'].append(f"Неподдерживаемый формат: {file_path.suffix}")
                return analysis
            
            # Базовая информация
            analysis['details']['shape'] = df.shape
            analysis['details']['num_rows'] = len(df)
            analysis['details']['num_columns'] = len(df.columns)
            analysis['details']['columns'] = df.columns.tolist()
            
            # Анализ колонок
            analysis['details']['column_types'] = df.dtypes.astype(str).to_dict()
            
            # Проверка на пустоту
            if df.empty:
                analysis['status'] = 'empty'
                analysis['errors'].append("Файл пустой (0 строк)")
                return analysis
            
            # Анализ пропусков
            missing_info = self._analyze_missing_values(df)
            analysis['details']['missing_values'] = missing_info
            
            # Проверка наличия критичных колонок
            column_check = self._check_required_columns(df)
            analysis['details']['column_check'] = column_check
            
            # Проверка типов данных
            dtype_check = self._check_data_types(df)
            analysis['details']['data_type_check'] = dtype_check
            
            # Статистика по числовым колонкам
            numeric_stats = self._get_numeric_statistics(df)
            analysis['details']['numeric_statistics'] = numeric_stats
            
            # Проверка на дубликаты
            duplicates = self._check_duplicates(df)
            analysis['details']['duplicates'] = duplicates
            
            # Определение пригодности для ML
            ml_readiness = self._assess_ml_readiness(df, analysis)
            analysis['details']['ml_readiness'] = ml_readiness
            
            # Финальная оценка
            if ml_readiness['is_suitable']:
                analysis['status'] = 'ready' if ml_readiness['quality_score'] >= 0.8 else 'needs_cleaning'
            else:
                analysis['status'] = 'not_suitable'
            
            # Формирование рекомендаций
            recommendations = self._generate_recommendations(analysis)
            analysis['recommendations'] = recommendations
            
        except Exception as e:
            analysis['status'] = 'error'
            analysis['errors'].append(f"Ошибка при обработке: {str(e)}")
        
        return analysis
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализирует пропущенные значения."""
        missing = df.isnull().sum()
        missing_percent = (missing / len(df) * 100).round(2)
        
        return {
            'total_missing': int(missing.sum()),
            'missing_by_column': {
                col: {
                    'count': int(missing[col]),
                    'percent': float(missing_percent[col])
                }
                for col in df.columns if missing[col] > 0
            },
            'columns_with_missing': missing[missing > 0].index.tolist()
        }
    
    def _check_required_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Проверяет наличие необходимых колонок."""
        columns_lower = [col.lower() for col in df.columns]
        
        found_categories = {
            'demographics': [],
            'coordinates_supine': [],
            'coordinates_lateral': [],
            'displacement': []
        }
        
        # Ищем колонки по категориям
        for category, keywords in self.expected_columns_ru.items():
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in keywords):
                    found_categories[category].append(col)
        
        # Специальная проверка координат X, Y, Z
        coord_axes = {'x': [], 'y': [], 'z': []}
        for col in df.columns:
            col_lower = col.lower()
            if 'ось x' in col_lower or col_lower.strip() == 'x':
                coord_axes['x'].append(col)
            elif 'ось y' in col_lower or col_lower.strip() == 'y':
                coord_axes['y'].append(col)
            elif 'ось z' in col_lower or col_lower.strip() == 'z':
                coord_axes['z'].append(col)
        
        return {
            'found_categories': found_categories,
            'coordinate_axes': coord_axes,
            'has_demographics': len(found_categories['demographics']) > 0,
            'has_coordinates': any(len(v) > 0 for v in coord_axes.values()),
            'has_displacement': len(found_categories['displacement']) > 0
        }
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Проверяет типы данных в колонках."""
        type_info = {
            'numeric_columns': [],
            'text_columns': [],
            'datetime_columns': [],
            'mixed_columns': []
        }
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                type_info['numeric_columns'].append(col)
            elif pd.api.types.is_string_dtype(dtype) or dtype == 'object':
                # Проверяем, может ли быть числовой
                try:
                    pd.to_numeric(df[col].dropna(), errors='raise')
                    type_info['mixed_columns'].append(col)
                except:
                    type_info['text_columns'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_info['datetime_columns'].append(col)
        
        return type_info
    
    def _get_numeric_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Получает статистику по числовым колонкам."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'message': 'Нет числовых колонок'}
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'has_outliers': self._detect_outliers(df[col])
            }
        
        return stats
    
    def _detect_outliers(self, series: pd.Series) -> bool:
        """Определяет наличие выбросов (>3 std)."""
        if series.dtype not in [np.float64, np.int64]:
            return False
        
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return False
        
        mean = clean_series.mean()
        std = clean_series.std()
        
        if pd.isna(std) or std == 0:
            return False
        
        outliers = np.abs(clean_series - mean) > 3 * std
        return outliers.sum() > 0
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Проверяет дубликаты."""
        duplicate_rows = df.duplicated().sum()
        
        # Проверка дубликатов по потенциальному ID (ФИО, номер пациента и т.д.)
        potential_id_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['фио', 'имя', 'пациент', 'id', 'номер']):
                potential_id_cols.append(col)
        
        duplicate_ids = {}
        for col in potential_id_cols:
            dup_count = df[col].duplicated().sum()
            if dup_count > 0:
                duplicate_ids[col] = int(dup_count)
        
        return {
            'duplicate_rows': int(duplicate_rows),
            'duplicate_rows_percent': round(duplicate_rows / len(df) * 100, 2),
            'potential_id_columns': potential_id_cols,
            'duplicate_ids': duplicate_ids
        }
    
    def _assess_ml_readiness(self, df: pd.DataFrame, analysis: Dict) -> Dict[str, Any]:
        """Оценивает готовность данных для ML."""
        score = 1.0
        issues = []
        
        # Проверка размера датасета
        if len(df) < 50:
            score -= 0.3
            issues.append(f"Мало данных: {len(df)} строк (нужно минимум 100)")
        elif len(df) < 100:
            score -= 0.1
            issues.append(f"Желательно больше данных: {len(df)} строк")
        
        # Проверка наличия координат
        column_check = analysis['details']['column_check']
        if not column_check['has_coordinates']:
            score -= 0.5
            issues.append("Не найдены координаты почки (X, Y, Z)")
        
        # Проверка пропусков
        missing_info = analysis['details']['missing_values']
        if missing_info['total_missing'] > 0:
            missing_percent = missing_info['total_missing'] / (len(df) * len(df.columns)) * 100
            if missing_percent > 20:
                score -= 0.3
                issues.append(f"Много пропусков: {missing_percent:.1f}%")
            elif missing_percent > 5:
                score -= 0.1
                issues.append(f"Есть пропуски: {missing_percent:.1f}%")
        
        # Проверка числовых колонок
        dtype_check = analysis['details']['data_type_check']
        if len(dtype_check['numeric_columns']) < 5:
            score -= 0.2
            issues.append(f"Мало числовых признаков: {len(dtype_check['numeric_columns'])}")
        
        # Проверка дубликатов
        duplicates = analysis['details']['duplicates']
        if duplicates['duplicate_rows'] > 0:
            dup_percent = duplicates['duplicate_rows_percent']
            if dup_percent > 10:
                score -= 0.2
                issues.append(f"Много дубликатов: {dup_percent:.1f}%")
        
        score = max(0, score)
        
        return {
            'is_suitable': score >= 0.5,
            'quality_score': round(score, 2),
            'issues': issues,
            'estimated_patients': len(df)
        }
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Генерирует рекомендации по улучшению данных."""
        recommendations = []
        
        ml_readiness = analysis['details']['ml_readiness']
        
        # Рекомендации по размеру
        if ml_readiness['estimated_patients'] < 100:
            recommendations.append(
                f"⚠️ КРИТИЧНО: Нужно больше данных. Текущее количество: {ml_readiness['estimated_patients']}, "
                "рекомендуется минимум 120 пациентов"
            )
        
        # Рекомендации по пропускам
        missing_info = analysis['details']['missing_values']
        if missing_info['total_missing'] > 0:
            critical_missing = []
            for col, info in missing_info['missing_by_column'].items():
                if info['percent'] > 20:
                    critical_missing.append(f"{col} ({info['percent']:.1f}%)")
            
            if critical_missing:
                recommendations.append(
                    f"🔧 Критичные пропуски в колонках: {', '.join(critical_missing)}. "
                    "Рекомендуется удалить эти строки или заполнить значениями"
                )
            else:
                recommendations.append(
                    "🔧 Есть пропуски, но не критичные. Можно заполнить медианой/средним"
                )
        
        # Рекомендации по колонкам
        column_check = analysis['details']['column_check']
        if not column_check['has_coordinates']:
            recommendations.append(
                "❌ КРИТИЧНО: Не найдены координаты почки (X, Y, Z). "
                "Проверьте названия колонок"
            )
        
        if not column_check['has_displacement']:
            recommendations.append(
                "⚠️ Не найдены колонки со смещением (delta). "
                "Возможно, нужно вычислить их как разницу между позициями"
            )
        
        # Рекомендации по дубликатам
        duplicates = analysis['details']['duplicates']
        if duplicates['duplicate_rows'] > 0:
            recommendations.append(
                f"🔧 Найдено {duplicates['duplicate_rows']} дубликатов строк. "
                "Рекомендуется удалить"
            )
        
        # Рекомендации по выбросам
        numeric_stats = analysis['details'].get('numeric_statistics', {})
        outlier_cols = [col for col, stats in numeric_stats.items() 
                       if isinstance(stats, dict) and stats.get('has_outliers')]
        
        if outlier_cols:
            recommendations.append(
                f"🔧 Обнаружены выбросы в колонках: {', '.join(outlier_cols[:5])}. "
                "Рекомендуется проверить и удалить значения > 3 стандартных отклонений"
            )
        
        # Итоговая рекомендация
        if ml_readiness['is_suitable']:
            if ml_readiness['quality_score'] >= 0.8:
                recommendations.insert(0, "✅ Данные готовы для ML с минимальной предобработкой")
            else:
                recommendations.insert(0, "⚠️ Данные можно использовать, но требуется очистка")
        else:
            recommendations.insert(0, "❌ Данные НЕ готовы для ML. Требуется серьёзная доработка")
        
        return recommendations
    
    def analyze_all_files(self) -> Dict[str, Any]:
        """Анализирует все файлы в директории."""
        print(f"🔍 Сканирование директории: {self.data_dir}")
        
        files_by_type = self.scan_directory()
        
        total_files = sum(len(files) for files in files_by_type.values())
        
        if total_files == 0:
            print("❌ Файлы не найдены!")
            self.report['summary']['status'] = 'no_files'
            return self.report
        
        print(f"📁 Найдено файлов: {total_files}")
        print(f"   📊 Табличные данные: {len(files_by_type['tabular_files'])}")
        print(f"   🏥 DICOM файлы: {len(files_by_type['dicom_files'])}")
        print(f"   📄 Презентации: {len(files_by_type['presentation_files'])}")
        print(f"   📁 Другие файлы: {len(files_by_type['other_files'])}")
        print()
        
        # Анализируем табличные файлы
        if files_by_type['tabular_files']:
            print("📊 Анализ табличных файлов:")
            for i, file_path in enumerate(files_by_type['tabular_files'], 1):
                print(f"[{i}/{len(files_by_type['tabular_files'])}] Анализ: {file_path.name}")
                analysis = self.analyze_file(file_path)
                self.report['files_analyzed'].append(analysis)
                
                status = analysis['status']
                if status == 'ready':
                    print(f"  ✅ Готов для ML")
                elif status == 'needs_cleaning':
                    print(f"  ⚠️ Требуется очистка")
                elif status == 'not_suitable':
                    print(f"  ❌ Не подходит для ML")
                else:
                    print(f"  ⚠️ Ошибка при обработке")
                
                if 'details' in analysis and 'shape' in analysis['details']:
                    shape = analysis['details']['shape']
                    print(f"     Размер: {shape[0]} строк × {shape[1]} колонок")
                
                print()
        
        # Анализируем DICOM директории
        dicom_directories = self._find_dicom_directories()
        if dicom_directories:
            print("🏥 Анализ DICOM исследований:")
            for i, dir_path in enumerate(dicom_directories, 1):
                print(f"[{i}/{len(dicom_directories)}] Анализ исследования: {dir_path.name}")
                analysis = self.analyze_dicom_directory(dir_path)
                self.report['files_analyzed'].append(analysis)
                
                status = analysis['status']
                if status == 'ready_for_processing':
                    print(f"  ✅ Готово для обработки")
                elif status == 'insufficient_data':
                    print(f"  ⚠️ Мало данных")
                else:
                    print(f"  ❌ Проблемы с данными")
                
                if 'details' in analysis and 'dicom_file_count' in analysis['details']:
                    count = analysis['details']['dicom_file_count']
                    size = analysis['details'].get('total_size_mb', 0)
                    print(f"     Срезы: {count}, Размер: {size} МБ")
                
                print()
        
        # Анализируем все остальные файлы и директории
        print("📁 Анализ всех файлов и директорий:")
        all_items_analysis = self._analyze_all_directory_contents()
        self.report['files_analyzed'].extend(all_items_analysis)
        
        # Формируем сводку
        self._generate_summary()
        
        # Общие рекомендации
        self._generate_overall_recommendations()
        
        return self.report
    
    def _find_dicom_directories(self) -> List[Path]:
        """Находит директории, содержащие DICOM файлы."""
        dicom_dirs = []
        
        for item in self.data_dir.glob('*'):
            if item.is_dir():
                # Проверяем, есть ли в директории DICOM файлы
                has_dicom = False
                for file_path in item.glob('**/*'):
                    if file_path.is_file() and self._is_dicom_file(file_path):
                        has_dicom = True
                        break
                
                if has_dicom:
                    dicom_dirs.append(item)
        
        return sorted(dicom_dirs)
    
    def _analyze_all_directory_contents(self) -> List[Dict[str, Any]]:
        """Анализирует все файлы и директории в корневой папке."""
        analyses = []
        
        # Получаем все элементы в корневой директории
        all_items = []
        try:
            all_items = list(self.data_dir.glob('*'))
        except Exception as e:
            print(f"⚠️ Ошибка при чтении директории: {e}")
            return analyses
        
        for item in all_items:
            if item.is_dir():
                # Анализ директории
                analysis = self._analyze_directory(item)
                analyses.append(analysis)
            elif item.is_file():
                # Анализ файла (кроме тех, что уже проанализированы)
                if not self._is_already_analyzed(item):
                    analysis = self._analyze_generic_file(item)
                    analyses.append(analysis)
        
        return analyses
    
    def _is_already_analyzed(self, file_path: Path) -> bool:
        """Проверяет, был ли файл уже проанализирован."""
        # Проверяем по расширению, был ли файл уже включен в другие анализы
        ext = file_path.suffix.lower()
        if ext in ['.csv', '.xlsx', '.xls']:
            return True  # Уже проанализирован как табличный файл
        if ext in ['.pptx', '.pdf']:
            return True  # Уже проанализирован как презентация
        if ext in ['.exe', '.inf', '.cds']:
            return True  # Уже исключен из анализа
        if ext == '' or ext == '.dcm':
            return True  # Уже проанализирован как DICOM
        
        return False
    
    def _analyze_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Анализирует директорию."""
        analysis = {
            'name': dir_path.name,
            'path': str(dir_path),
            'type': 'directory',
            'status': 'analyzed',
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            # Базовая информация
            analysis['details']['item_count'] = len(list(dir_path.glob('*')))
            
            # Анализ содержимого
            files = []
            subdirs = []
            
            for item in dir_path.glob('*'):
                if item.is_file():
                    file_info = self._get_file_info(item)
                    files.append(file_info)
                elif item.is_dir():
                    subdir_info = {
                        'name': item.name,
                        'path': str(item),
                        'type': 'subdirectory'
                    }
                    subdirs.append(subdir_info)
            
            analysis['details']['files'] = files
            analysis['details']['subdirectories'] = subdirs
            analysis['details']['file_count'] = len(files)
            analysis['details']['subdir_count'] = len(subdirs)
            
            # Классификация файлов по типам
            file_types = {}
            for file_info in files:
                ext = file_info['extension']
                file_types[ext] = file_types.get(ext, 0) + 1
            
            analysis['details']['file_types'] = file_types
            
            # Общий размер
            total_size = sum(f['size_bytes'] for f in files)
            analysis['details']['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            # Извлечение информации из имени директории
            patient_info = self._extract_patient_info_from_name(dir_path.name)
            if any(patient_info.values()):
                analysis['details']['extracted_info'] = patient_info
            
        except Exception as e:
            analysis['status'] = 'error'
            analysis['errors'].append(f"Ошибка при анализе директории: {str(e)}")
        
        return analysis
    
    def _analyze_generic_file(self, file_path: Path) -> Dict[str, Any]:
        """Анализирует обычный файл."""
        analysis = {
            'name': file_path.name,
            'path': str(file_path),
            'type': 'file',
            'status': 'analyzed',
            'errors': [],
            'warnings': [],
            'details': {}
        }
        
        try:
            file_info = self._get_file_info(file_path)
            analysis['details'].update(file_info)
            
            # Дополнительная информация в зависимости от типа файла
            ext = file_path.suffix.lower()
            
            if ext == '.json':
                analysis['details']['file_category'] = 'json_data'
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(1000)  # Читаем первые 1000 символов
                        analysis['details']['content_preview'] = content[:200] + "..." if len(content) > 200 else content
                except:
                    analysis['warnings'].append("Не удалось прочитать JSON файл")
            
            elif ext == '.txt':
                analysis['details']['file_category'] = 'text_file'
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(500)
                        analysis['details']['content_preview'] = content[:200] + "..." if len(content) > 200 else content
                except:
                    analysis['warnings'].append("Не удалось прочитать текстовый файл")
            
            elif ext in ['.exe', '.msi']:
                analysis['details']['file_category'] = 'executable'
                analysis['warnings'].append("Исполняемый файл - требует осторожности")
            
            elif ext in ['.cds', '.img']:
                analysis['details']['file_category'] = 'dicom_related'
                analysis['details']['description'] = 'Служебный файл DICOM'
            
            elif ext in ['.inf']:
                analysis['details']['file_category'] = 'system_file'
                analysis['details']['description'] = 'Системный файл конфигурации'
            
            else:
                analysis['details']['file_category'] = 'other'
                analysis['details']['description'] = f'Файл типа {ext}'
            
        except Exception as e:
            analysis['status'] = 'error'
            analysis['errors'].append(f"Ошибка при анализе файла: {str(e)}")
        
        return analysis
    
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Получает базовую информацию о файле."""
        try:
            stat = file_path.stat()
            return {
                'extension': file_path.suffix.lower(),
                'size_bytes': stat.st_size,
                'size_kb': round(stat.st_size / 1024, 2),
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat()
            }
        except Exception as e:
            return {
                'extension': file_path.suffix.lower(),
                'size_bytes': 0,
                'size_kb': 0,
                'size_mb': 0,
                'error': f"Не удалось получить информацию о файле: {str(e)}"
            }
    
    def _generate_summary(self):
        """Генерирует сводку по всем проанализированным файлам."""
        tabular_ready = 0
        tabular_needs_cleaning = 0
        tabular_not_suitable = 0
        tabular_errors = 0
        dicom_ready = 0
        dicom_insufficient = 0
        dicom_errors = 0
        directories_analyzed = 0
        generic_files_analyzed = 0
        
        for analysis in self.report['files_analyzed']:
            if analysis.get('type') == 'dicom_study':
                if analysis['status'] == 'ready_for_processing':
                    dicom_ready += 1
                elif analysis['status'] == 'insufficient_data':
                    dicom_insufficient += 1
                else:
                    dicom_errors += 1
            elif analysis.get('type') == 'directory':
                directories_analyzed += 1
            elif analysis.get('type') == 'file':
                generic_files_analyzed += 1
            else:
                # Табличные файлы
                if analysis['status'] == 'ready':
                    tabular_ready += 1
                elif analysis['status'] == 'needs_cleaning':
                    tabular_needs_cleaning += 1
                elif analysis['status'] == 'not_suitable':
                    tabular_not_suitable += 1
                else:
                    tabular_errors += 1
        
        # Подсчет общего количества файлов по типам
        total_files_by_type = {}
        total_size_mb = 0
        
        for analysis in self.report['files_analyzed']:
            if analysis.get('type') == 'directory' and 'details' in analysis:
                file_types = analysis['details'].get('file_types', {})
                for ext, count in file_types.items():
                    total_files_by_type[ext] = total_files_by_type.get(ext, 0) + count
                
                total_size_mb += analysis['details'].get('total_size_mb', 0)
        
        self.report['summary'] = {
            'total_analyzed': len(self.report['files_analyzed']),
            'tabular_files': {
                'ready': tabular_ready,
                'needs_cleaning': tabular_needs_cleaning,
                'not_suitable': tabular_not_suitable,
                'errors': tabular_errors
            },
            'dicom_studies': {
                'ready': dicom_ready,
                'insufficient': dicom_insufficient,
                'errors': dicom_errors
            },
            'directories': {
                'analyzed': directories_analyzed
            },
            'generic_files': {
                'analyzed': generic_files_analyzed
            },
            'file_types_summary': total_files_by_type,
            'total_size_mb': round(total_size_mb, 2),
            'overall_status': 'mixed' if (tabular_ready + dicom_ready) > 0 else 'poor'
        }
    
    def _generate_overall_recommendations(self):
        """Генерирует общие рекомендации по всем файлам."""
        summary = self.report['summary']
        recommendations = []
        
        # Подсчёт DICOM исследований
        total_dicom_studies = summary['dicom_studies']['ready'] + summary['dicom_studies']['insufficient']
        total_dicom_files = sum(
            len(analysis['details'].get('dicom_files', []))
            for analysis in self.report['files_analyzed']
            if analysis.get('type') == 'dicom_study' and 'details' in analysis
        )
        
        if total_dicom_studies > 0:
            recommendations.append(
                f"🏥 Найдено DICOM исследований: {total_dicom_studies} "
                f"(всего срезов: {total_dicom_files})"
            )
            
            if summary['dicom_studies']['ready'] > 0:
                recommendations.append(
                    f"✅ Готовых к обработке: {summary['dicom_studies']['ready']} исследований"
                )
            
            if summary['dicom_studies']['insufficient'] > 0:
                recommendations.append(
                    f"⚠️ Мало данных: {summary['dicom_studies']['insufficient']} исследований"
                )
        
        # Подсчёт табличных данных
        total_tabular = sum(summary['tabular_files'].values())
        if total_tabular > 0:
            total_rows = sum(
                analysis['details'].get('num_rows', 0)
                for analysis in self.report['files_analyzed']
                if analysis.get('type') != 'dicom_study' and 'details' in analysis
            )
            
            recommendations.append(
                f"📊 Табличные данные: {total_tabular} файлов ({total_rows} строк)"
            )
        
        # Общая оценка
        if summary['overall_status'] == 'mixed':
            recommendations.append(
                "📋 ПЛАН ДЕЙСТВИЙ:\n"
                "1. Для DICOM: использовать pydicom для извлечения метаданных\n"
                "2. Создать табличный датасет с координатами почек\n"
                "3. Извлечь признаки из изображений (текстура, форма)\n"
                "4. Объединить с клиническими данными\n"
                "5. Разделить на train/val/test (64%/16%/20%)"
            )
        else:
            recommendations.append(
                "❌ Недостаточно данных для ML. Требуется больше исследований"
            )
        
        self.report['recommendations'] = recommendations
    
    def save_report(self, output_path: str = 'data_analysis_report.json'):
        """
        Сохраняет отчёт в JSON файл.
        
        Args:
            output_path: Путь для сохранения отчёта
        """
        output_file = Path(output_path)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Отчёт сохранён: {output_file.absolute()}")
    
    def print_summary(self):
        """Выводит краткую сводку в консоль."""
        print("\n" + "="*80)
        print("📊 ИТОГОВАЯ СВОДКА")
        print("="*80)
        
        summary = self.report['summary']
        
        print(f"\n📁 Всего проанализировано: {summary['total_analyzed']}")
        
        if summary['tabular_files']:
            tab = summary['tabular_files']
            print(f"\n📊 Табличные файлы:")
            print(f"   ✅ Готовы: {tab['ready']}")
            print(f"   ⚠️ Требуют очистки: {tab['needs_cleaning']}")
            print(f"   ❌ Не подходят: {tab['not_suitable']}")
            print(f"   ⚠️ Ошибки: {tab['errors']}")
        
        if summary['dicom_studies']:
            dic = summary['dicom_studies']
            print(f"\n🏥 DICOM исследования:")
            print(f"   ✅ Готовы к обработке: {dic['ready']}")
            print(f"   ⚠️ Мало данных: {dic['insufficient']}")
            print(f"   ❌ Ошибки: {dic['errors']}")
        
        if summary['directories']['analyzed'] > 0:
            print(f"\n📁 Директории проанализировано: {summary['directories']['analyzed']}")
        
        if summary['generic_files']['analyzed'] > 0:
            print(f"\n📄 Другие файлы: {summary['generic_files']['analyzed']}")
        
        # Показываем статистику по типам файлов
        if summary['file_types_summary']:
            print(f"\n📈 Статистика по типам файлов:")
            for ext, count in sorted(summary['file_types_summary'].items()):
                ext_name = ext if ext else '(без расширения)'
                if ext == '':
                    ext_name = 'DICOM'
                print(f"   {ext_name}: {count}")
        
        if summary['total_size_mb'] > 0:
            print(f"\n💾 Общий размер данных: {summary['total_size_mb']} МБ")
        
        print("\n" + "="*80)
        print("💡 РЕКОМЕНДАЦИИ")
        print("="*80)
        
        for rec in self.report['recommendations']:
            print(f"\n{rec}")
        
        print("\n" + "="*80)


def main():
    """Основная функция."""
    import sys
    
    # Путь к директории с данными
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
    else:
        # По умолчанию ищем в /mnt/user-data/uploads
        data_directory = "/mnt/user-data/uploads"
    
    print("="*80)
    print("🔬 АНАЛИЗАТОР МЕДИЦИНСКИХ ДАННЫХ ДЛЯ ML")
    print("="*80)
    print(f"\n📂 Директория для анализа: {data_directory}\n")
    
    try:
        # Создаём анализатор
        analyzer = DataAnalyzer(data_directory)
        
        # Анализируем все файлы
        report = analyzer.analyze_all_files()
        
        # Выводим сводку
        analyzer.print_summary()
        
        # Сохраняем отчёт
        analyzer.save_report('data_analysis_report.json')
        
        print("\n✅ Анализ завершён успешно!")
        print(f"📄 Подробный отчёт: data_analysis_report.json")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()