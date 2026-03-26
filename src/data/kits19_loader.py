#!/usr/bin/env python3
"""
KiTS19 Dataset Loader

Загрузчик и обработчик датасета KiTS19 для 3D сегментации почек и опухолей.

Структура KiTS19:
data/
├── case_00000/
│   ├── imaging.nii.gz      # CT скан
│   └── segmentation.nii.gz # Сегментация (0=фон, 1=почка, 2=опухоль)
├── case_00001/
└── kits.json              # Метаданные

Автор: AR Kidney ML Project
Версия: 1.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KiTS19Dataset:
    """
    Класс для работы с датасетом KiTS19.
    
    Предоставляет функции для загрузки, предобработки и анализа данных.
    """
    
    def __init__(self, data_dir: str):
        """
        Инициализация датасета.
        
        Args:
            data_dir: Путь к директории data/ KiTS19
        """
        self.data_dir = Path(data_dir)
        self.cases = []
        self.metadata = {}
        
        # Проверяем структуру директории
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {data_dir}")
        
        # Загружаем метаданные
        self._load_metadata()
        
        # Находим все случаи
        self._discover_cases()
        
        logger.info(f"Загружено {len(self.cases)} случаев из KiTS19")
    
    def _load_metadata(self):
        """Загрузка метаданных из kits.json"""
        kits_json_path = self.data_dir / 'kits.json'
        if kits_json_path.exists():
            with open(kits_json_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info("Метаданные kits.json загружены")
        else:
            logger.warning("kits.json не найден")
    
    def _discover_cases(self):
        """Поиск всех случаев (case_XXXXX) в директории data/"""
        case_dirs = [d for d in self.data_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('case_')]
        
        self.cases = sorted(case_dirs, key=lambda x: int(x.name.split('_')[1]))
        
        # Валидация каждого случая
        valid_cases = []
        for case_dir in self.cases:
            imaging_path = case_dir / 'imaging.nii.gz'
            segmentation_path = case_dir / 'segmentation.nii.gz'
            
            if imaging_path.exists() and segmentation_path.exists():
                valid_cases.append(case_dir)
            else:
                logger.warning(f"Неполный случай: {case_dir.name}")
        
        self.cases = valid_cases
    
    def load_case(self, case_identifier: Union[str, int]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Загрузка одного случая.
        
        Args:
            case_identifier: ID случая (например 'case_00000' или 0)
            
        Returns:
            tuple: (image, segmentation, metadata)
                - image: 3D numpy array CT скана
                - segmentation: 3D numpy array с метками (0=фон, 1=почка, 2=опухоль)
                - metadata: словарь с метаданными случая
        """
        # Определяем путь к случаю
        if isinstance(case_identifier, int):
            if case_identifier >= len(self.cases):
                raise IndexError(f"Индекс {case_identifier} выходит за пределы {len(self.cases)}")
            case_dir = self.cases[case_identifier]
        else:
            case_dir = self.data_dir / case_identifier
            if not case_dir.exists():
                raise FileNotFoundError(f"Случай не найден: {case_identifier}")
        
        # Пути к файлам
        imaging_path = case_dir / 'imaging.nii.gz'
        segmentation_path = case_dir / 'segmentation.nii.gz'
        
        # Загрузка NIfTI файлов
        try:
            # Загрузка изображения
            img_nii = nib.load(str(imaging_path))
            image = img_nii.get_fdata().astype(np.float32)
            
            # Загрузка сегментации
            seg_nii = nib.load(str(segmentation_path))
            segmentation = seg_nii.get_fdata().astype(np.uint8)
            
            # Метаданные
            metadata = {
                'case_id': case_dir.name,
                'shape': image.shape,
                'affine': img_nii.affine.tolist(),
                'voxel_spacing': img_nii.header.get_zooms(),
                'data_type': str(image.dtype),
                'unique_labels': np.unique(segmentation).tolist(),
                'organ_volume': np.sum(segmentation == 1),
                'tumor_volume': np.sum(segmentation == 2),
                'total_volume': np.sum(segmentation > 0)
            }
            
            # Добавляем метаданные из kits.json если есть
            case_id_num = case_dir.name.split('_')[1]
            if case_id_num in self.metadata:
                metadata.update(self.metadata[case_id_num])
            
            return image, segmentation, metadata
            
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки {case_dir.name}: {e}")
    
    def get_dataset_statistics(self) -> Dict:
        """
        Получение статистики по всему датасету.
        
        Returns:
            Словарь со статистикой
        """
        logger.info("Вычисление статистики датасета...")
        
        stats = {
            'total_cases': len(self.cases),
            'shapes': [],
            'voxel_spacings': [],
            'organ_volumes': [],
            'tumor_volumes': [],
            'cases_with_tumor': 0,
            'cases_without_tumor': 0
        }
        
        for i in tqdm(range(len(self.cases)), desc="Анализ случаев"):
            try:
                _, seg, metadata = self.load_case(i)
                
                stats['shapes'].append(metadata['shape'])
                stats['voxel_spacings'].append(metadata['voxel_spacing'])
                stats['organ_volumes'].append(metadata['organ_volume'])
                stats['tumor_volumes'].append(metadata['tumor_volume'])
                
                if metadata['tumor_volume'] > 0:
                    stats['cases_with_tumor'] += 1
                else:
                    stats['cases_without_tumor'] += 1
                    
            except Exception as e:
                logger.warning(f"Ошибка в случае {i}: {e}")
                continue
        
        # Вычисляем агрегированную статистику
        shapes = np.array(stats['shapes'])
        voxel_spacings = np.array(stats['voxel_spacings'])
        
        stats.update({
            'shape_stats': {
                'min': shapes.min(axis=0).tolist(),
                'max': shapes.max(axis=0).tolist(),
                'mean': shapes.mean(axis=0).tolist(),
                'median': np.median(shapes, axis=0).tolist()
            },
            'voxel_spacing_stats': {
                'min': voxel_spacings.min(axis=0).tolist(),
                'max': voxel_spacings.max(axis=0).tolist(),
                'mean': voxel_spacings.mean(axis=0).tolist(),
                'median': np.median(voxel_spacings, axis=0).tolist()
            },
            'organ_volume_stats': {
                'min': min(stats['organ_volumes']),
                'max': max(stats['organ_volumes']),
                'mean': np.mean(stats['organ_volumes']),
                'median': np.median(stats['organ_volumes'])
            },
            'tumor_volume_stats': {
                'min': min(stats['tumor_volumes']),
                'max': max(stats['tumor_volumes']),
                'mean': np.mean(stats['tumor_volumes']),
                'median': np.median(stats['tumor_volumes'])
            },
            'tumor_prevalence': stats['cases_with_tumor'] / stats['total_cases']
        })
        
        return stats
    
    def create_train_val_test_split(self, 
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15,
                                   random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
        """
        Создание разделения на train/validation/test.
        
        Args:
            train_ratio: Доля обучающих данных
            val_ratio: Доля валидационных данных  
            test_ratio: Доля тестовых данных
            random_seed: Случайное зерно
            
        Returns:
            Tuple[List[str], List[str], List[str]]: (train_cases, val_cases, test_cases)
        """
        np.random.seed(random_seed)
        
        case_names = [c.name for c in self.cases]
        np.random.shuffle(case_names)
        
        n = len(case_names)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_cases = case_names[:train_end]
        val_cases = case_names[train_end:val_end]
        test_cases = case_names[val_end:]
        
        logger.info(f"Разделение: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)}")
        
        return train_cases, val_cases, test_cases
    
    def save_dataset_info(self, output_path: str):
        """
        Сохранение информации о датасете в JSON файл.
        
        Args:
            output_path: Путь для сохранения
        """
        dataset_info = {
            'dataset_path': str(self.data_dir),
            'total_cases': len(self.cases),
            'case_names': [c.name for c in self.cases],
            'metadata': self.metadata,
            'statistics': self.get_dataset_statistics()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Информация о датасете сохранена: {output_path}")


def load_case_kits19(data_dir: str, case_identifier: Union[str, int]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Упрощенная функция для загрузки одного случая.
    
    Args:
        data_dir: Путь к директории data/ KiTS19
        case_identifier: ID случая
        
    Returns:
        (image, segmentation, metadata)
    """
    dataset = KiTS19Dataset(data_dir)
    return dataset.load_case(case_identifier)


if __name__ == "__main__":
    # Пример использования
    data_dir = "data"  # Путь к директории data/ KiTS19
    
    try:
        # Создаем датасет
        dataset = KiTS19Dataset(data_dir)
        
        # Загружаем первый случай для теста
        image, segmentation, metadata = dataset.load_case(0)
        print(f"Размер изображения: {image.shape}")
        print(f"Размер сегментации: {segmentation.shape}")
        print(f"Уникальные метки: {np.unique(segmentation)}")
        print(f"Метаданные: {metadata}")
        
        # Получаем статистику
        stats = dataset.get_dataset_statistics()
        print(f"\nСтатистика датасета:")
        print(f"Всего случаев: {stats['total_cases']}")
        print(f"Случаев с опухолью: {stats['cases_with_tumor']}")
        print(f"Случаев без опухоли: {stats['cases_without_tumor']}")
        print(f"Распространенность опухоли: {stats['tumor_prevalence']:.2%}")
        
        # Сохраняем информацию
        dataset.save_dataset_info("kits19_dataset_info.json")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        print("Убедитесь что датасет KiTS19 загружен и структура папок правильная")
