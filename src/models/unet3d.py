#!/usr/bin/env python3
"""
3D U-Net Architecture for KiTS19 Segmentation

Реализация 3D U-Net для сегментации почек и опухолей на КТ данных.
Поддерживает multi-class сегментацию (фон, почка, опухоль).

Автор: AR Kidney ML Project
Версия: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DoubleConv3D(nn.Module):
    """
    Двойной сверточный блок 3D U-Net.
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down3D(nn.Module):
    """
    Нисходящий путь с MaxPooling.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """
    Восходящий путь с transpose convolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Обработка разницы в размерах
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # Конкатенация
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """
    Выходной сверточный слой.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net архитектура для медицинской сегментации.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 num_classes: int = 3,  # 0=фон, 1=почка, 2=опухоль
                 base_channels: int = 64,
                 dropout: float = 0.1):
        """
        Инициализация 3D U-Net.
        
        Args:
            in_channels: Количество входных каналов
            num_classes: Количество классов сегментации
            base_channels: Базовое количество каналов
            dropout: Dropout вероятность
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        
        # Энкодер (нисходящий путь)
        self.inc = DoubleConv3D(in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        self.down4 = Down3D(base_channels * 8, base_channels * 16)
        
        # Декодер (восходящий путь)
        self.up1 = Up3D(base_channels * 16, base_channels * 8)
        self.up2 = Up3D(base_channels * 8, base_channels * 4)
        self.up3 = Up3D(base_channels * 4, base_channels * 2)
        self.up4 = Up3D(base_channels * 2, base_channels)
        
        # Выходной слой
        self.outc = OutConv3D(base_channels, num_classes)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout3d(dropout)
        
        # Инициализация весов
        self._initialize_weights()
        
        logger.info(f"3D U-Net создан:")
        logger.info(f"  Входные каналы: {in_channels}")
        logger.info(f"  Классы: {num_classes}")
        logger.info(f"  Базовые каналы: {base_channels}")
        logger.info(f"  Dropout: {dropout}")
    
    def _initialize_weights(self):
        """Инициализация весов сети."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход сети.
        
        Args:
            x: Входной тензор [B, C, D, H, W]
            
        Returns:
            Выходной тензор [B, num_classes, D, H, W]
        """
        # Энкодер
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Dropout в bottleneck
        x5 = self.dropout(x5)
        
        # Декодер
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Выход
        logits = self.outc(x)
        
        return logits
    
    def get_model_info(self) -> dict:
        """
        Получение информации о модели.
        
        Returns:
            Словарь с параметрами модели
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': '3D U-Net',
            'input_channels': self.in_channels,
            'num_classes': self.num_classes,
            'base_channels': self.base_channels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }


class AttentionUNet3D(nn.Module):
    """
    3D U-Net с attention механизмом.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 num_classes: int = 3,
                 base_channels: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        
        # Энкодер
        self.inc = DoubleConv3D(in_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        self.down4 = Down3D(base_channels * 8, base_channels * 16)
        
        # Attention gates (упрощенная версия)
        self.att1 = AttentionGate(base_channels * 16, base_channels * 8, base_channels * 4)
        self.att2 = AttentionGate(base_channels * 8, base_channels * 4, base_channels * 2)
        self.att3 = AttentionGate(base_channels * 4, base_channels * 2, base_channels)
        self.att4 = AttentionGate(base_channels * 2, base_channels, base_channels // 2)
        
        # Декодер
        self.up1 = Up3D(base_channels * 16, base_channels * 8)
        self.up2 = Up3D(base_channels * 8, base_channels * 4)
        self.up3 = Up3D(base_channels * 4, base_channels * 2)
        self.up4 = Up3D(base_channels * 2, base_channels)
        
        # Выход
        self.outc = OutConv3D(base_channels, num_classes)
        self.dropout = nn.Dropout3d(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Энкодер
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = self.dropout(x5)
        
        # Декодер с attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


class AttentionGate(nn.Module):
    """
    Attention Gate для Attention U-Net.
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


def create_unet3d(model_type: str = 'unet',
                  in_channels: int = 1,
                  num_classes: int = 3,
                  base_channels: int = 64,
                  dropout: float = 0.1) -> nn.Module:
    """
    Создание 3D U-Net модели.
    
    Args:
        model_type: Тип модели ('unet', 'attention_unet')
        in_channels: Количество входных каналов
        num_classes: Количество классов
        base_channels: Базовое количество каналов
        dropout: Dropout вероятность
        
    Returns:
        Модель PyTorch
    """
    if model_type == 'unet':
        model = UNet3D(in_channels, num_classes, base_channels, dropout)
    elif model_type == 'attention_unet':
        model = AttentionUNet3D(in_channels, num_classes, base_channels, dropout)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    return model


def test_unet3d():
    """
    Тестирование 3D U-Net модели.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Тестирование на устройстве: {device}")
    
    # Создаем модель
    model = create_unet3d(
        model_type='unet',
        in_channels=1,
        num_classes=3,
        base_channels=32,  # Уменьшаем для теста
        dropout=0.1
    )
    model = model.to(device)
    
    # Тестовый вход
    batch_size = 2
    depth, height, width = 64, 128, 128
    test_input = torch.randn(batch_size, 1, depth, height, width).to(device)
    
    print(f"Входной размер: {test_input.shape}")
    
    # Прямой проход
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Выходной размер: {output.shape}")
    
    # Информация о модели
    model_info = model.get_model_info()
    print("\nИнформация о модели:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Проверка выхода
    assert output.shape[0] == batch_size
    assert output.shape[1] == 3  # 3 класса
    assert output.shape[2:] == test_input.shape[2:]
    
    print("\n✅ Тестирование 3D U-Net пройдено успешно!")


if __name__ == "__main__":
    test_unet3d()
