#!/usr/bin/env python3
"""
Унифицированная модель для проекта AR Kidney ML

Поддерживает:
- 3D сегментацию (KiTS19)
- Предсказание координат смещения (табличные данные)
- Transfer learning между источниками
- Multi-task обучение

Автор: AR Kidney ML Project
Версия: 2.0 (Унифицированная)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from enum import Enum

from .unet3d import UNet3D, AttentionUNet3D
from .losses_metrics import DiceLoss, ComboLoss, SegmentationMetrics

logger = logging.getLogger(__name__)


class ModelMode(Enum):
    """Режимы работы модели."""
    SEGMENTATION_ONLY = "segmentation_only"
    COORDINATE_PREDICTION_ONLY = "coordinate_prediction"
    MULTI_TASK = "multi_task"
    TRANSFER_LEARNING = "transfer_learning"


class FeatureExtractor(nn.Module):
    """
    Универсальный экстрактор признаков из изображений.
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 base_channels: int = 64,
                 output_features: int = 512):
        super().__init__()
        
        # Используем энкодер от U-Net
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.fc = nn.Linear(base_channels * 8, output_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Извлечение признаков из 3D изображения.
        
        Args:
            x: Входное изображение [B, C, D, H, W]
            
        Returns:
            Признаки [B, output_features]
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        features = self.fc(x)
        return features


class CoordinatePredictor(nn.Module):
    """
    Предсказатель координат смещения почек.
    """
    
    def __init__(self, 
                 input_features: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 num_coordinates: int = 9):  # 3 точки * 3 координаты
        super().__init__()
        
        layers = []
        prev_dim = input_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_coordinates))
        
        self.predictor = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Предсказание координат смещения.
        
        Args:
            features: Входные признаки [B, input_features]
            
        Returns:
            Координаты смещения [B, num_coordinates]
        """
        return self.predictor(features)


class UnifiedKidneyModel(nn.Module):
    """
    Унифицированная модель для всех задач проекта.
    """
    
    def __init__(self, 
                 mode: ModelMode = ModelMode.MULTI_TASK,
                 input_channels: int = 1,
                 num_classes: int = 3,  # фон, почка, опухоль
                 base_channels: int = 64,
                 feature_dim: int = 512,
                 coordinate_dim: int = 9,
                 dropout: float = 0.1):
        """
        Инициализация унифицированной модели.
        
        Args:
            mode: Режим работы модели
            input_channels: Количество входных каналов
            num_classes: Количество классов сегментации
            base_channels: Базовое количество каналов
            feature_dim: Размерность признаков
            coordinate_dim: Количество координат для предсказания
            dropout: Dropout вероятность
        """
        super().__init__()
        
        self.mode = mode
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.coordinate_dim = coordinate_dim
        
        # Компоненты модели
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            base_channels=base_channels,
            output_features=feature_dim
        )
        
        # Сегментационная ветка (для KiTS19)
        if mode in [ModelMode.SEGMENTATION_ONLY, ModelMode.MULTI_TASK, ModelMode.TRANSFER_LEARNING]:
            self.segmentation_head = UNet3D(
                in_channels=input_channels,
                num_classes=num_classes,
                base_channels=base_channels,
                dropout=dropout
            )
        
        # Предсказание координат (для табличных данных)
        if mode in [ModelMode.COORDINATE_PREDICTION_ONLY, ModelMode.MULTI_TASK]:
            self.coordinate_predictor = CoordinatePredictor(
                input_features=feature_dim,
                num_coordinates=coordinate_dim
            )
        
        # Fusion layer для multi-task
        if mode == ModelMode.MULTI_TASK:
            self.fusion_layer = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        
        # Инициализация весов
        self._initialize_weights()
        
        logger.info(f"UnifiedKidneyModel создана:")
        logger.info(f"  Режим: {mode.value}")
        logger.info(f"  Классы сегментации: {num_classes}")
        logger.info(f"  Размерность признаков: {feature_dim}")
        logger.info(f"  Размерность координат: {coordinate_dim}")
    
    def _initialize_weights(self):
        """Инициализация весов модели."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, 
                image: Optional[torch.Tensor] = None,
                features: Optional[torch.Tensor] = None,
                tabular_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Прямой проход модели.
        
        Args:
            image: 3D изображение [B, C, D, H, W]
            features: Предварительно извлеченные признаки [B, feature_dim]
            tabular_data: Табличные данные [B, tabular_dim]
            
        Returns:
            Словарь с предсказаниями
        """
        outputs = {}
        
        # Извлечение признаков из изображения если есть
        if image is not None:
            extracted_features = self.feature_extractor(image)
            outputs['features'] = extracted_features
            
            # Сегментация
            if hasattr(self, 'segmentation_head'):
                segmentation_logits = self.segmentation_head(image)
                outputs['segmentation'] = segmentation_logits
        elif features is not None:
            extracted_features = features
            outputs['features'] = features
        else:
            raise ValueError("Необходимо предоставить image или features")
        
        # Предсказание координат
        if hasattr(self, 'coordinate_predictor'):
            # Fusion с табличными данными если есть
            if tabular_data is not None and hasattr(self, 'fusion_layer'):
                # Конкатенация признаков изображения и табличных данных
                combined_features = torch.cat([extracted_features, tabular_data], dim=1)
                fused_features = self.fusion_layer(combined_features)
                coordinates = self.coordinate_predictor(fused_features)
            else:
                coordinates = self.coordinate_predictor(extracted_features)
            
            outputs['coordinates'] = coordinates
        
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'UnifiedKidneyModel',
            'mode': self.mode.value,
            'input_channels': getattr(self, 'input_channels', 1),
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'coordinate_dim': self.coordinate_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'components': {
                'feature_extractor': hasattr(self, 'feature_extractor'),
                'segmentation_head': hasattr(self, 'segmentation_head'),
                'coordinate_predictor': hasattr(self, 'coordinate_predictor'),
                'fusion_layer': hasattr(self, 'fusion_layer')
            }
        }


class UnifiedLoss(nn.Module):
    """
    Унифицированная функция потерь для multi-task обучения.
    """
    
    def __init__(self, 
                 mode: ModelMode,
                 segmentation_weight: float = 1.0,
                 coordinate_weight: float = 1.0,
                 dice_weight: float = 0.5,
                 ce_weight: float = 0.5):
        super().__init__()
        
        self.mode = mode
        self.segmentation_weight = segmentation_weight
        self.coordinate_weight = coordinate_weight
        
        # Функции потерь
        if mode in [ModelMode.SEGMENTATION_ONLY, ModelMode.MULTI_TASK]:
            self.segmentation_loss = ComboLoss(
                dice_weight=dice_weight,
                ce_weight=ce_weight
            )
        
        if mode in [ModelMode.COORDINATE_PREDICTION_ONLY, ModelMode.MULTI_TASK]:
            self.coordinate_loss = nn.MSELoss()
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Вычисление потерь.
        
        Args:
            predictions: Предсказания модели
            targets: Истинные значения
            
        Returns:
            Словарь с потерями
        """
        losses = {}
        total_loss = 0.0
        
        # Потери сегментации
        if 'segmentation' in predictions and 'segmentation' in targets:
            seg_loss = self.segmentation_loss(
                predictions['segmentation'], 
                targets['segmentation']
            )
            losses['segmentation'] = seg_loss
            total_loss += self.segmentation_weight * seg_loss
        
        # Потери координат
        if 'coordinates' in predictions and 'coordinates' in targets:
            coord_loss = self.coordinate_loss(
                predictions['coordinates'],
                targets['coordinates']
            )
            losses['coordinates'] = coord_loss
            total_loss += self.coordinate_weight * coord_loss
        
        losses['total'] = total_loss
        return losses


class UnifiedMetrics:
    """
    Унифицированные метрики для всех задач.
    """
    
    def __init__(self, mode: ModelMode, num_classes: int = 3):
        self.mode = mode
        self.num_classes = num_classes
        
        # Метрики сегментации
        if mode in [ModelMode.SEGMENTATION_ONLY, ModelMode.MULTI_TASK]:
            self.segmentation_metrics = SegmentationMetrics(
                num_classes, 
                ['Background', 'Kidney', 'Tumor']
            )
        
        # Метрики координат
        if mode in [ModelMode.COORDINATE_PREDICTION_ONLY, ModelMode.MULTI_TASK]:
            self.coordinate_errors = []
            self.coordinate_mae = []
    
    def update(self, 
               predictions: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor]):
        """Обновление метрик."""
        
        # Метрики сегментации
        if 'segmentation' in predictions and 'segmentation' in targets:
            pred_seg = torch.argmax(predictions['segmentation'], dim=1)
            true_seg = targets['segmentation']
            self.segmentation_metrics.update(pred_seg, true_seg)
        
        # Метрики координат
        if 'coordinates' in predictions and 'coordinates' in targets:
            pred_coords = predictions['coordinates'].cpu().numpy()
            true_coords = targets['coordinates'].cpu().numpy()
            
            # MAE для координат
            mae = np.mean(np.abs(pred_coords - true_coords), axis=1)
            self.coordinate_mae.extend(mae)
            
            # Euclidean distance для 3D точек
            pred_coords_reshaped = pred_coords.reshape(-1, 3, 3)  # [B, 3_points, 3_coords]
            true_coords_reshaped = true_coords.reshape(-1, 3, 3)
            
            for i in range(pred_coords_reshaped.shape[0]):
                point_errors = []
                for j in range(3):  # 3 точки
                    dist = np.linalg.norm(pred_coords_reshaped[i, j] - true_coords_reshaped[i, j])
                    point_errors.append(dist)
                self.coordinate_errors.append(point_errors)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Вычисление итоговых метрик."""
        metrics = {}
        
        # Метрики сегментации
        if hasattr(self, 'segmentation_metrics'):
            seg_metrics = self.segmentation_metrics.compute_average_metrics()
            metrics['segmentation'] = seg_metrics
        
        # Метрики координат
        if self.coordinate_mae:
            metrics['coordinates'] = {
                'mean_mae': np.mean(self.coordinate_mae),
                'std_mae': np.std(self.coordinate_mae)
            }
            
            if self.coordinate_errors:
                errors_array = np.array(self.coordinate_errors)
                metrics['coordinates'].update({
                    'upper_point_error': np.mean(errors_array[:, 0]),
                    'middle_point_error': np.mean(errors_array[:, 1]),
                    'lower_point_error': np.mean(errors_array[:, 2]),
                    'mean_3d_error': np.mean(errors_array)
                })
        
        return metrics
    
    def print_metrics(self):
        """Печать метрик."""
        metrics = self.compute_metrics()
        
        print("\n" + "="*60)
        print(f"UNIFIED METRICS ({self.mode.value})")
        print("="*60)
        
        if 'segmentation' in metrics:
            print("\nSEGMENTATION METRICS:")
            seg_metrics = metrics['segmentation']
            if 'overall' in seg_metrics:
                overall = seg_metrics['overall']
                print(f"  Mean Dice: {overall['mean_dice']:.4f}")
                print(f"  Mean IoU: {overall['mean_iou']:.4f}")
                print(f"  Pixel Accuracy: {overall['pixel_accuracy']:.4f}")
        
        if 'coordinates' in metrics:
            print("\nCOORDINATE METRICS:")
            coord_metrics = metrics['coordinates']
            print(f"  Mean MAE: {coord_metrics['mean_mae']:.4f} mm")
            print(f"  Upper point error: {coord_metrics.get('upper_point_error', 0):.4f} mm")
            print(f"  Middle point error: {coord_metrics.get('middle_point_error', 0):.4f} mm")
            print(f"  Lower point error: {coord_metrics.get('lower_point_error', 0):.4f} mm")
            print(f"  Mean 3D error: {coord_metrics.get('mean_3d_error', 0):.4f} mm")
        
        print("="*60)


def create_unified_model(mode: ModelMode = ModelMode.MULTI_TASK,
                       **kwargs) -> UnifiedKidneyModel:
    """
    Создание унифицированной модели.
    
    Args:
        mode: Режим работы
        **kwargs: Дополнительные параметры
        
    Returns:
        UnifiedKidneyModel
    """
    return UnifiedKidneyModel(mode=mode, **kwargs)


def test_unified_model():
    """Тестирование унифицированной модели."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Тестирование на устройстве: {device}")
    
    # Создаем модель
    model = create_unified_model(
        mode=ModelMode.MULTI_TASK,
        input_channels=1,
        num_classes=3,
        base_channels=32,  # Уменьшаем для теста
        feature_dim=256,
        coordinate_dim=9
    )
    model = model.to(device)
    
    # Тестовые данные
    batch_size = 2
    depth, height, width = 32, 64, 64
    
    # Изображение
    test_image = torch.randn(batch_size, 1, depth, height, width).to(device)
    
    # Табличные данные
    tabular_dim = 10  # Например, возраст, пол, ИМТ и т.д.
    test_tabular = torch.randn(batch_size, tabular_dim).to(device)
    
    print(f"Изображение: {test_image.shape}")
    print(f"Табличные данные: {test_tabular.shape}")
    
    # Прямой проход
    with torch.no_grad():
        outputs = model(image=test_image, tabular_data=test_tabular)
    
    print("\nВыходы модели:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Информация о модели
    model_info = model.get_model_info()
    print(f"\nИнформация о модели:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Тестирование потерь
    criterion = UnifiedLoss(mode=ModelMode.MULTI_TASK).to(device)
    
    # Целевые значения
    targets = {
        'segmentation': torch.randint(0, 3, (batch_size, depth, height, width)).to(device),
        'coordinates': torch.randn(batch_size, 9).to(device)
    }
    
    with torch.no_grad():
        losses = criterion(outputs, targets)
    
    print(f"\nПотери:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n✅ Тестирование унифицированной модели пройдено успешно!")


if __name__ == "__main__":
    test_unified_model()
