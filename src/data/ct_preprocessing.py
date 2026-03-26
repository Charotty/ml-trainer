#!/usr/bin/env python3
"""
CT Data Preprocessing for KiTS19

Модуль для предобработки КТ данных датасета KiTS19.
Включает нормализацию Hounsfield units, ресемплинг, аугментации.

Автор: AR Kidney ML Project
Версия: 1.0
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, List, Union
from scipy import ndimage
from skimage.transform import resize
import SimpleITK as sitk
from skimage.exposure import equalize_hist
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


class CTPreprocessor:
    """
    Класс для предобработки КТ данных.
    """
    
    def __init__(self, 
                 clip_range: Tuple[float, float] = (-1000, 1000),
                 target_spacing: Optional[Tuple[float, float, float]] = None,
                 target_size: Optional[Tuple[int, int, int]] = None,
                 normalize: bool = True):
        """
        Инициализация предобработчика.
        
        Args:
            clip_range: Диапазон отсечения Hounsfield units
            target_spacing: Целевой воксельный размер (mm)
            target_size: Целевой размер изображения (voxels)
            normalize: Применять нормализацию
        """
        self.clip_range = clip_range
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.normalize = normalize
        
        logger.info(f"CT Preprocessor инициализирован:")
        logger.info(f"  Clip range: {clip_range}")
        logger.info(f"  Target spacing: {target_spacing}")
        logger.info(f"  Target size: {target_size}")
        logger.info(f"  Normalize: {normalize}")
    
    def clip_hounsfield_units(self, image: np.ndarray) -> np.ndarray:
        """
        Отсечение значений Hounsfield Units.
        
        Args:
            image: Входное изображение
            
        Returns:
            Изображение с отсеченными значениями
        """
        return np.clip(image, self.clip_range[0], self.clip_range[1])
    
    def normalize_hounsfield_units(self, image: np.ndarray) -> np.ndarray:
        """
        Нормализация Hounsfield Units в [0, 1].
        
        Args:
            image: Входное изображение
            
        Returns:
            Нормализованное изображение
        """
        min_val, max_val = self.clip_range
        normalized = (image - min_val) / (max_val - min_val)
        return normalized.astype(np.float32)
    
    def resample_image(self, 
                      image: np.ndarray, 
                      original_spacing: Tuple[float, float, float],
                      target_spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Ресемплинг изображения с изменением воксельного размера.
        
        Args:
            image: Входное изображение
            original_spacing: Исходный воксельный размер
            target_spacing: Целевой воксельный размер
            
        Returns:
            Ресемплированное изображение
        """
        # Вычисляем новый размер
        original_size = np.array(image.shape)
        scale_factor = np.array(original_spacing) / np.array(target_spacing)
        new_size = (original_size * scale_factor).astype(int)
        
        # Ресемплинг
        if len(image.shape) == 3:
            # 3D изображение
            resized = resize(image, new_size, order=3, preserve_range=True, anti_aliasing=True)
        else:
            raise ValueError(f"Неподдерживаемое количество измерений: {len(image.shape)}")
        
        return resized.astype(image.dtype)
    
    def resize_image(self, 
                    image: np.ndarray, 
                    target_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Изменение размера изображения.
        
        Args:
            image: Входное изображение
            target_size: Целевой размер
            
        Returns:
            Измененное изображение
        """
        if len(image.shape) != 3:
            raise ValueError(f"Ожидается 3D изображение, получено {len(image.shape)}D")
        
        resized = resize(image, target_size, order=3, preserve_range=True, anti_aliasing=True)
        return resized.astype(image.dtype)
    
    def preprocess_segmentation(self, 
                               segmentation: np.ndarray, 
                               target_size: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Предобработка сегментации.
        
        Args:
            segmentation: Маска сегментации
            target_size: Целевой размер
            
        Returns:
            Обработанная сегментация
        """
        # Используем nearest neighbor для сегментации
        if target_size is not None:
            resized = resize(segmentation, target_size, order=0, preserve_range=True, anti_aliasing=False)
            return resized.astype(np.uint8)
        
        return segmentation
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Улучшение контраста изображения.
        
        Args:
            image: Входное изображение
            
        Returns:
            Изображение с улучшенным контрастом
        """
        # Гистограммная эквализация
        enhanced = equalize_hist(image)
        return enhanced.astype(np.float32)
    
    def remove_noise(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """
        Удаление шума из изображения.
        
        Args:
            image: Входное изображение
            method: Метод удаления шума ('gaussian', 'median')
            
        Returns:
            Очищенное изображение
        """
        if method == 'gaussian':
            filtered = ndimage.gaussian_filter(image, sigma=1.0)
        elif method == 'median':
            filtered = ndimage.median_filter(image, size=3)
        else:
            raise ValueError(f"Неизвестный метод фильтрации: {method}")
        
        return filtered.astype(image.dtype)
    
    def preprocess_case(self, 
                       image: np.ndarray, 
                       segmentation: np.ndarray,
                       original_spacing: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Полная предобработка случая.
        
        Args:
            image: Входное изображение
            segmentation: Входная сегментация
            original_spacing: Исходный воксельный размер
            
        Returns:
            (processed_image, processed_segmentation, metadata)
        """
        metadata = {
            'original_shape': image.shape,
            'original_spacing': original_spacing,
            'processing_steps': []
        }
        
        # 1. Отсечение HU
        image = self.clip_hounsfield_units(image)
        metadata['processing_steps'].append('clip_hounsfield')
        
        # 2. Ресемплинг если нужно
        if self.target_spacing is not None:
            image = self.resample_image(image, original_spacing, self.target_spacing)
            segmentation = self.preprocess_segmentation(segmentation, image.shape)
            metadata['processing_steps'].append('resample')
            metadata['new_spacing'] = self.target_spacing
        
        # 3. Изменение размера если нужно
        if self.target_size is not None:
            image = self.resize_image(image, self.target_size)
            segmentation = self.preprocess_segmentation(segmentation, self.target_size)
            metadata['processing_steps'].append('resize')
            metadata['new_shape'] = image.shape
        
        # 4. Нормализация
        if self.normalize:
            image = self.normalize_hounsfield_units(image)
            metadata['processing_steps'].append('normalize')
        
        # 5. Удаление шума (опционально)
        # image = self.remove_noise(image)
        # metadata['processing_steps'].append('denoise')
        
        metadata['final_shape'] = image.shape
        metadata['final_dtype'] = str(image.dtype)
        
        return image, segmentation, metadata


class KiTS19Augmentation:
    """
    Класс для аугментации 3D данных KiTS19.
    """
    
    def __init__(self, is_train: bool = True):
        """
        Инициализация аугментаций.
        
        Args:
            is_train: True для обучающих данных, False для валидации
        """
        self.is_train = is_train
        
        if is_train:
            # Аугментации для обучения
            self.transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            ])
        else:
            # Только базовые преобразования для валидации
            self.transforms = A.Compose([])
    
    def augment_3d(self, image: np.ndarray, segmentation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применение аугментаций к 3D данным.
        
        Args:
            image: 3D изображение
            segmentation: 3D сегментация
            
        Returns:
            (augmented_image, augmented_segmentation)
        """
        if not self.is_train:
            return image, segmentation
        
        # Применяем аугментации покадрово (2D) для простоты
        augmented_images = []
        augmented_segmentations = []
        
        for i in range(image.shape[0]):  # По оси Z
            slice_img = image[i]
            slice_seg = segmentation[i]
            
            # Применяем трансформации
            transformed = self.transforms(image=slice_img, mask=slice_seg)
            aug_img = transformed['image']
            aug_seg = transformed['mask']
            
            augmented_images.append(aug_img)
            augmented_segmentations.append(aug_seg)
        
        return np.array(augmented_images), np.array(augmented_segmentations)


class KiTS19Dataset(Dataset):
    """
    PyTorch Dataset для KiTS19 с предобработкой и аугментацией.
    """
    
    def __init__(self, 
                 data_dir: str,
                 case_list: List[str],
                 preprocessor: CTPreprocessor,
                 augmentation: Optional[KiTS19Augmentation] = None,
                 cache_data: bool = False):
        """
        Инициализация датасета.
        
        Args:
            data_dir: Путь к директории data/ KiTS19
            case_list: Список случаев для включения
            preprocessor: Предобработчик
            augmentation: Аугментации (опционально)
            cache_data: Кэшировать данные в памяти
        """
        from .kits19_loader import KiTS19Dataset as KiTS19Loader
        
        self.loader = KiTS19Loader(data_dir)
        self.case_list = case_list
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.cache_data = cache_data
        self.cache = {}
        
        logger.info(f"Dataset создан с {len(case_list)} случаями")
    
    def __len__(self) -> int:
        return len(self.case_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_name = self.case_list[idx]
        
        # Проверяем кэш
        if self.cache_data and case_name in self.cache:
            cached_data = self.cache[case_name]
            if self.augmentation:
                # Применяем аугментации "на лету"
                image, segmentation = self.augmentation.augment_3d(
                    cached_data['image'], cached_data['segmentation']
                )
            else:
                image, segmentation = cached_data['image'], cached_data['segmentation']
        else:
            # Загружаем данные
            image, segmentation, metadata = self.loader.load_case(case_name)
            original_spacing = metadata['voxel_spacing']
            
            # Предобработка
            image, segmentation, _ = self.preprocessor.preprocess_case(
                image, segmentation, original_spacing
            )
            
            # Кэшируем если нужно
            if self.cache_data:
                self.cache[case_name] = {
                    'image': image.copy(),
                    'segmentation': segmentation.copy()
                }
            
            # Аугментации
            if self.augmentation:
                image, segmentation = self.augmentation.augment_3d(image, segmentation)
        
        # Конвертация в тензоры
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # Добавляем канал
        segmentation_tensor = torch.from_numpy(segmentation).long()
        
        return {
            'image': image_tensor,
            'segmentation': segmentation_tensor,
            'case_name': case_name
        }


def create_data_loaders(data_dir: str,
                       train_cases: List[str],
                       val_cases: List[str],
                       test_cases: List[str],
                       batch_size: int = 2,
                       num_workers: int = 4,
                       target_spacing: Tuple[float, float, float] = (1.5, 1.5, 3.0),
                       target_size: Tuple[int, int, int] = (128, 128, 128),
                       cache_data: bool = False) -> Tuple[torch.utils.data.DataLoader, 
                                                        torch.utils.data.DataLoader,
                                                        torch.utils.data.DataLoader]:
    """
    Создание DataLoader'ов для обучения.
    
    Args:
        data_dir: Путь к директории data/ KiTS19
        train_cases: Список обучающих случаев
        val_cases: Список валидационных случаев
        test_cases: Список тестовых случаев
        batch_size: Размер батча
        num_workers: Количество воркеров
        target_spacing: Целевой воксельный размер
        target_size: Целевой размер изображения
        cache_data: Использовать кэширование
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Предобработчик
    preprocessor = CTPreprocessor(
        clip_range=(-1000, 1000),
        target_spacing=target_spacing,
        target_size=target_size,
        normalize=True
    )
    
    # Аугментации
    train_augmentation = KiTS19Augmentation(is_train=True)
    val_augmentation = KiTS19Augmentation(is_train=False)
    
    # Датасеты
    train_dataset = KiTS19Dataset(
        data_dir, train_cases, preprocessor, train_augmentation, cache_data
    )
    val_dataset = KiTS19Dataset(
        data_dir, val_cases, preprocessor, val_augmentation, cache_data
    )
    test_dataset = KiTS19Dataset(
        data_dir, test_cases, preprocessor, val_augmentation, cache_data
    )
    
    # DataLoader'ы
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    logger.info(f"DataLoader'ы созданы:")
    logger.info(f"  Train: {len(train_dataset)} случаев, batch_size={batch_size}")
    logger.info(f"  Val: {len(val_dataset)} случаев, batch_size={batch_size}")
    logger.info(f"  Test: {len(test_dataset)} случаев, batch_size=1")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Пример использования
    print("CT Preprocessing модуль готов к использованию")
