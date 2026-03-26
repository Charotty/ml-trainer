#!/usr/bin/env python3
"""
Унифицированный загрузчик данных для проекта AR Kidney ML

Поддерживает:
- KiTS19 (NIfTI изображения + сегментация)
- Табличные данные (CSV/Excel с координатами смещения)
- Непарные КТ сканы
- Смешанные датасеты

Автор: AR Kidney ML Project
Версия: 2.0 (Унифицированная)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import nibabel as nib
from enum import Enum

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Типы источников данных."""
    KITS19 = "kits19"
    TABULAR = "tabular"
    UNPAIRED_CT = "unpaired_ct"
    MIXED = "mixed"


class BaseDataLoader(ABC):
    """Абстрактный базовый класс для загрузчиков данных."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metadata = {}
    
    @abstractmethod
    def load_case(self, case_id: Union[str, int]) -> Dict[str, Any]:
        """Загрузить один случай."""
        pass
    
    @abstractmethod
    def get_case_list(self) -> List[str]:
        """Получить список всех случаев."""
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Получить информацию о датасете."""
        pass


class KiTS19Loader(BaseDataLoader):
    """Загрузчик для KiTS19 датасета."""
    
    def __init__(self, data_dir: str, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.data_dir = Path(data_dir)
        self.cases = []
        self.kits_metadata = {}
        self._discover_cases()
    
    def _discover_cases(self):
        """Поиск случаев KiTS19."""
        case_dirs = [d for d in self.data_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('case_')]
        self.cases = sorted(case_dirs, key=lambda x: int(x.name.split('_')[1]))
        
        # Загрузка метаданных
        kits_json = self.data_dir / 'kits.json'
        if kits_json.exists():
            with open(kits_json, 'r') as f:
                self.kits_metadata = json.load(f)
    
    def load_case(self, case_id: Union[str, int]) -> Dict[str, Any]:
        """Загрузить случай KiTS19."""
        if isinstance(case_id, int):
            case_dir = self.cases[case_id]
        else:
            case_dir = self.data_dir / case_id
        
        # Загрузка изображений
        img_path = case_dir / 'imaging.nii.gz'
        seg_path = case_dir / 'segmentation.nii.gz'
        
        img_nii = nib.load(str(img_path))
        seg_nii = nib.load(str(seg_path))
        
        image = img_nii.get_fdata().astype(np.float32)
        segmentation = seg_nii.get_fdata().astype(np.uint8)
        
        # Извлечение признаков из сегментации
        features = self._extract_features_from_segmentation(segmentation, img_nii.affine)
        
        return {
            'case_id': case_dir.name,
            'source_type': DataSourceType.KITS19,
            'image': image,
            'segmentation': segmentation,
            'affine': img_nii.affine.tolist(),
            'voxel_spacing': img_nii.header.get_zooms(),
            'extracted_features': features,
            'coordinates': self._segmentation_to_coordinates(segmentation, img_nii.affine)
        }
    
    def _extract_features_from_segmentation(self, segmentation: np.ndarray, affine: np.ndarray) -> Dict[str, Any]:
        """Извлечение признаков из сегментации."""
        features = {
            'organ_volume': np.sum(segmentation == 1),
            'tumor_volume': np.sum(segmentation == 2),
            'organ_center': self._find_center_of_mass(segmentation == 1),
            'tumor_center': self._find_center_of_mass(segmentation == 2) if np.any(segmentation == 2) else None,
            'organ_bounds': self._find_bounds(segmentation == 1),
            'tumor_bounds': self._find_bounds(segmentation == 2) if np.any(segmentation == 2) else None
        }
        return features
    
    def _find_center_of_mass(self, mask: np.ndarray) -> Tuple[float, float, float]:
        """Находит центр масс маски."""
        indices = np.where(mask)
        if len(indices[0]) == 0:
            return (0, 0, 0)
        center = np.mean(indices, axis=1)
        return tuple(center)
    
    def _find_bounds(self, mask: np.ndarray) -> Dict[str, Tuple[int, int, int]]:
        """Находит границы маски."""
        indices = np.where(mask)
        if len(indices[0]) == 0:
            return {'min': (0, 0, 0), 'max': (0, 0, 0)}
        
        bounds = {
            'min': tuple(np.min(indices, axis=1)),
            'max': tuple(np.max(indices, axis=1))
        }
        return bounds
    
    def _segmentation_to_coordinates(self, segmentation: np.ndarray, affine: np.ndarray) -> Dict[str, Any]:
        """Конвертация сегментации в координаты смещения."""
        # Находим ключевые точки почки
        organ_mask = segmentation == 1
        
        if not np.any(organ_mask):
            return {}
        
        # Разделяем почку на верх/середину/низ по Z координате
        organ_indices = np.where(organ_mask)
        z_coords = organ_indices[0]
        
        if len(z_coords) == 0:
            return {}
        
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_third = (z_max - z_min) // 3
        
        # Верхняя треть
        upper_mask = np.zeros_like(organ_mask)
        upper_mask[organ_indices[0][z_coords <= z_min + z_third],
                   organ_indices[1][z_coords <= z_min + z_third],
                   organ_indices[2][z_coords <= z_min + z_third]] = 1
        
        # Средняя треть
        middle_mask = np.zeros_like(organ_mask)
        middle_z_mask = (z_coords > z_min + z_third) & (z_coords <= z_min + 2*z_third)
        middle_mask[organ_indices[0][middle_z_mask],
                   organ_indices[1][middle_z_mask],
                   organ_indices[2][middle_z_mask]] = 1
        
        # Нижняя треть
        lower_mask = np.zeros_like(organ_mask)
        lower_z_mask = z_coords > z_min + 2*z_third
        lower_mask[organ_indices[0][lower_z_mask],
                   organ_indices[1][lower_z_mask],
                   organ_indices[2][lower_z_mask]] = 1
        
        # Находим центры масс для каждой трети
        centers = {}
        for name, mask in [('upper', upper_mask), ('middle', middle_mask), ('lower', lower_mask)]:
            if np.any(mask):
                center = self._find_center_of_mass(mask)
                # Конвертируем в физические координаты
                center_phys = nib.affines.apply_affine(affine, center)
                centers[f'{name}_supine'] = center_phys.tolist()
        
        return coordinates
    
    def get_case_list(self) -> List[str]:
        return [case.name for case in self.cases]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            'source_type': DataSourceType.KITS19,
            'total_cases': len(self.cases),
            'data_structure': 'NIfTI images + segmentation masks',
            'coordinates_available': True,
            'features_extracted': True
        }


class TabularDataLoader(BaseDataLoader):
    """Загрузчик для табличных данных с координатами."""
    
    def __init__(self, data_dir: str, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.data_dir = Path(data_dir)
        self.data_files = []
        self.combined_data = None
        self._discover_data_files()
        self._load_data()
    
    def _discover_data_files(self):
        """Поиск файлов с данными."""
        for ext in ['*.csv', '*.xlsx', '*.xls']:
            self.data_files.extend(self.data_dir.glob(ext))
    
    def _load_data(self):
        """Загрузка и объединение данных."""
        if not self.data_files:
            logger.warning("Файлы с данными не найдены")
            return
        
        dfs = []
        for file_path in self.data_files:
            try:
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path, sep=';', decimal=',')
                else:
                    df = pd.read_excel(file_path)
                
                df['source_file'] = file_path.name
                dfs.append(df)
                logger.info(f"Загружен файл: {file_path.name} ({len(df)} строк)")
            except Exception as e:
                logger.error(f"Ошибка загрузки {file_path.name}: {e}")
        
        if dfs:
            self.combined_data = pd.concat(dfs, ignore_index=True, sort=False)
            logger.info(f"Объединено {len(dfs)} файлов, всего {len(self.combined_data)} строк")
    
    def load_case(self, case_id: Union[str, int]) -> Dict[str, Any]:
        """Загрузить случай из табличных данных."""
        if self.combined_data is None:
            raise ValueError("Данные не загружены")
        
        if isinstance(case_id, int):
            if case_id >= len(self.combined_data):
                raise IndexError(f"Индекс {case_id} выходит за пределы")
            row = self.combined_data.iloc[case_id]
        else:
            # Поиск по ФИО или другому идентификатору
            mask = self.combined_data['ФИО'].astype(str).str.contains(str(case_id), na=False)
            if not mask.any():
                raise ValueError(f"Случай {case_id} не найден")
            row = self.combined_data[mask].iloc[0]
        
        # Извлечение координат
        coordinates = self._extract_coordinates_from_row(row)
        
        # Извлечение демографических данных
        demographics = self._extract_demographics_from_row(row)
        
        return {
            'case_id': str(row.get('ФИО', f'case_{case_id}')),
            'source_type': DataSourceType.TABULAR,
            'raw_data': row.to_dict(),
            'coordinates': coordinates,
            'demographics': demographics,
            'image': None,  # Нет изображения
            'segmentation': None
        }
    
    def _extract_coordinates_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """Извлечение координат из строки данных."""
        coordinates = {}
        
        # Ищем колонки с координатами
        coord_cols = [col for col in row.index 
                     if any(k in str(col).lower() for k in ['ось x', 'ось y', 'ось z', 'x (мм)', 'y (мм)', 'z (мм)'])]
        
        for col in coord_cols:
            if pd.notna(row[col]):
                coordinates[col] = float(row[col])
        
        return coordinates
    
    def _extract_demographics_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """Извлечение демографических данных."""
        demographics = {}
        
        # Пол
        if 'Пол' in row and pd.notna(row['Пол']):
            demographics['sex'] = 1 if str(row['Пол']).lower() in ['м', 'male'] else 0
        
        # Возраст
        if 'Возраст' in row and pd.notna(row['Возраст']):
            demographics['age'] = float(row['Возраст'])
        
        # ИМТ
        bmi_cols = [col for col in row.index if 'имт' in str(col).lower()]
        if bmi_cols and pd.notna(row[bmi_cols[0]]):
            demographics['bmi'] = float(row[bmi_cols[0]])
        
        return demographics
    
    def get_case_list(self) -> List[str]:
        if self.combined_data is None:
            return []
        return [f"case_{i}" for i in range(len(self.combined_data))]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            'source_type': DataSourceType.TABULAR,
            'total_cases': len(self.combined_data) if self.combined_data is not None else 0,
            'data_structure': 'Tabular CSV/Excel files',
            'coordinates_available': True,
            'features_extracted': False
        }


class UnpairedCTLoader(BaseDataLoader):
    """Загрузчик для непарных КТ сканов."""
    
    def __init__(self, data_dir: str, config: Optional[Dict] = None):
        super().__init__(config or {})
        self.data_dir = Path(data_dir)
        self.ct_files = []
        self._discover_ct_files()
    
    def _discover_ct_files(self):
        """Поиск КТ файлов."""
        for ext in ['*.nii.gz', '*.nii', '*.dcm']:
            self.ct_files.extend(self.data_dir.rglob(ext))
    
    def load_case(self, case_id: Union[str, int]) -> Dict[str, Any]:
        """Загрузить КТ скан."""
        if isinstance(case_id, int):
            if case_id >= len(self.ct_files):
                raise IndexError(f"Индекс {case_id} выходит за пределы")
            ct_path = self.ct_files[case_id]
        else:
            ct_path = self.data_dir / case_id
            if not ct_path.exists():
                raise FileNotFoundError(f"Файл не найден: {case_id}")
        
        # Загрузка КТ
        if ct_path.suffix in ['.nii.gz', '.nii']:
            img_nii = nib.load(str(ct_path))
            image = img_nii.get_fdata().astype(np.float32)
            affine = img_nii.affine
        else:
            # DICOM обработка (упрощенная)
            raise NotImplementedError("DICOM поддержка в разработке")
        
        return {
            'case_id': ct_path.stem,
            'source_type': DataSourceType.UNPAIRED_CT,
            'image': image,
            'segmentation': None,  # Нет сегментации
            'affine': affine.tolist(),
            'voxel_spacing': img_nii.header.get_zooms(),
            'coordinates': {}  # Нет координат смещения
        }
    
    def get_case_list(self) -> List[str]:
        return [f.name for f in self.ct_files]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            'source_type': DataSourceType.UNPAIRED_CT,
            'total_cases': len(self.ct_files),
            'data_structure': 'Unpaired CT scans (NIfTI/DICOM)',
            'coordinates_available': False,
            'features_extracted': False
        }


class UnifiedDataLoader:
    """
    Унифицированный загрузчик данных для всех типов источников.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация унифицированного загрузчика.
        
        Args:
            config: Конфигурация с путями к данным
        """
        self.config = config
        self.loaders = {}
        self.case_mapping = {}
        
        # Инициализация загрузчиков
        self._initialize_loaders()
        
        logger.info(f"Унифицированный загрузчик инициализирован с {len(self.loaders)} источниками")
    
    def _initialize_loaders(self):
        """Инициализация загрузчиков для каждого источника данных."""
        
        # KiTS19
        if 'kits19' in self.config and self.config['kits19'].get('enabled', False):
            kits19_config = self.config['kits19']
            self.loaders['kits19'] = KiTS19Loader(kits19_config['data_dir'], kits19_config)
        
        # Табличные данные
        if 'tabular' in self.config and self.config['tabular'].get('enabled', False):
            tabular_config = self.config['tabular']
            self.loaders['tabular'] = TabularDataLoader(tabular_config['data_dir'], tabular_config)
        
        # Непарные КТ
        if 'unpaired_ct' in self.config and self.config['unpaired_ct'].get('enabled', False):
            ct_config = self.config['unpaired_ct']
            self.loaders['unpaired_ct'] = UnpairedCTLoader(ct_config['data_dir'], ct_config)
        
        # Создание unified маппинга случаев
        self._create_case_mapping()
    
    def _create_case_mapping(self):
        """Создание маппинга всех случаев."""
        case_id = 0
        for source_name, loader in self.loaders.items():
            for case_name in loader.get_case_list():
                self.case_mapping[f"case_{case_id}"] = {
                    'source': source_name,
                    'original_id': case_name
                }
                case_id += 1
    
    def load_case(self, case_id: str) -> Dict[str, Any]:
        """
        Загрузить случай из любого источника.
        
        Args:
            case_id: Unified ID случая (например "case_0")
            
        Returns:
            Данные случая с унифицированной структурой
        """
        if case_id not in self.case_mapping:
            raise ValueError(f"Случай {case_id} не найден")
        
        mapping = self.case_mapping[case_id]
        source = mapping['source']
        original_id = mapping['original_id']
        
        # Загрузка из соответствующего источника
        loader = self.loaders[source]
        case_data = loader.load_case(original_id)
        
        # Добавление unified информации
        case_data['unified_id'] = case_id
        case_data['source_name'] = source
        
        return case_data
    
    def get_all_cases(self) -> List[Dict[str, Any]]:
        """Получить информацию обо всех случаях."""
        cases_info = []
        
        for unified_id, mapping in self.case_mapping.items():
            source = mapping['source']
            original_id = mapping['original_id']
            
            cases_info.append({
                'unified_id': unified_id,
                'source': source,
                'original_id': original_id,
                'source_type': self.loaders[source].get_dataset_info()['source_type']
            })
        
        return cases_info
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Получить статистику по всему датасету."""
        stats = {
            'total_cases': len(self.case_mapping),
            'sources': {}
        }
        
        for source_name, loader in self.loaders.items():
            source_info = loader.get_dataset_info()
            stats['sources'][source_name] = source_info
        
        return stats
    
    def create_train_val_test_split(self, 
                                  train_ratio: float = 0.7,
                                  val_ratio: float = 0.15,
                                  test_ratio: float = 0.15,
                                  random_seed: int = 42,
                                  stratify_by_source: bool = True) -> Tuple[List[str], List[str], List[str]]:
        """
        Создание разделения на train/val/test.
        
        Args:
            train_ratio: Доля обучающих данных
            val_ratio: Доля валидационных данных
            test_ratio: Доля тестовых данных
            random_seed: Случайное зерно
            stratify_by_source: Стратифицировать по источникам
            
        Returns:
            (train_cases, val_cases, test_cases)
        """
        all_cases = list(self.case_mapping.keys())
        np.random.seed(random_seed)
        np.random.shuffle(all_cases)
        
        n = len(all_cases)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_cases = all_cases[:train_end]
        val_cases = all_cases[train_end:val_end]
        test_cases = all_cases[val_end:]
        
        logger.info(f"Разделение: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)}")
        
        return train_cases, val_cases, test_cases


def create_unified_config() -> Dict[str, Any]:
    """
    Создание конфигурации для унифицированного загрузчика.
    
    Returns:
        Конфигурация для всех источников данных
    """
    config = {
        'kits19': {
            'enabled': True,
            'data_dir': 'data/kits19',
            'description': 'KiTS19 dataset with CT scans and segmentations'
        },
        'tabular': {
            'enabled': True,
            'data_dir': 'data/tabular',
            'description': 'Tabular data with kidney displacement coordinates'
        },
        'unpaired_ct': {
            'enabled': False,
            'data_dir': 'data/unpaired_ct',
            'description': 'Unpaired CT scans without segmentations'
        }
    }
    
    return config


if __name__ == "__main__":
    # Пример использования
    config = create_unified_config()
    
    try:
        # Создаем унифицированный загрузчик
        unified_loader = UnifiedDataLoader(config)
        
        # Получаем статистику
        stats = unified_loader.get_dataset_statistics()
        print("Статистика датасета:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        # Загружаем первый случай для теста
        all_cases = unified_loader.get_all_cases()
        if all_cases:
            first_case_id = all_cases[0]['unified_id']
            case_data = unified_loader.load_case(first_case_id)
            
            print(f"\nПервый случай ({first_case_id}):")
            print(f"  Источник: {case_data['source_name']}")
            print(f"  Тип: {case_data['source_type']}")
            print(f"  Доступны координаты: {bool(case_data.get('coordinates'))}")
            print(f"  Есть изображение: {case_data.get('image') is not None}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        print("Проверьте конфигурацию и наличие данных")
