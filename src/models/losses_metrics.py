#!/usr/bin/env python3
"""
Loss Functions and Metrics for 3D Segmentation

Функции потерь и метрики для 3D сегментации KiTS19.
Включает Dice Loss, Cross-Entropy, Combo Loss и метрики качества.

Автор: AR Kidney ML Project
Версия: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Dice Loss для 3D сегментации.
    """
    
    def __init__(self, 
                 epsilon: float = 1e-6,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Вычисление Dice Loss.
        
        Args:
            input: Предсказания [B, C, D, H, W]
            target: Истинные метки [B, D, H, W]
            
        Returns:
            Dice Loss
        """
        # Конвертация target в one-hot
        num_classes = input.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Softmax предсказаний
        input_probs = F.softmax(input, dim=1)
        
        # Вычисление Dice для каждого класса
        dice_scores = []
        for class_idx in range(num_classes):
            input_class = input_probs[:, class_idx]
            target_class = target_one_hot[:, class_idx]
            
            intersection = (input_class * target_class).sum(dim=(1, 2, 3, 4))
            union = input_class.sum(dim=(1, 2, 3, 4)) + target_class.sum(dim=(1, 2, 3, 4))
            
            dice_class = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
            dice_scores.append(1.0 - dice_class)  # Loss = 1 - Dice
        
        dice_loss = torch.stack(dice_scores, dim=1)  # [B, C]
        
        # Применение весов если есть
        if self.weight is not None:
            dice_loss = dice_loss * self.weight.unsqueeze(0)
        
        # Reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с дисбалансом классов.
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Вычисление Focal Loss.
        """
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ComboLoss(nn.Module):
    """
    Комбинация Dice Loss и Cross-Entropy Loss.
    """
    
    def __init__(self, 
                 dice_weight: float = 0.5,
                 ce_weight: float = 0.5,
                 focal_gamma: float = 0.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        if focal_gamma > 0:
            self.focal_loss = FocalLoss(gamma=focal_gamma)
        else:
            self.focal_loss = None
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Вычисление комбинированной потери.
        """
        dice = self.dice_loss(input, target)
        ce = self.ce_loss(input, target)
        
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        
        if self.focal_loss is not None:
            focal = self.focal_loss(input, target)
            total_loss = total_loss * 0.7 + focal * 0.3
        
        return total_loss


class SegmentationMetrics:
    """
    Класс для вычисления метрик сегментации.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
        # Метрики для накопления
        self.reset()
    
    def reset(self):
        """Сброс накопленных метрик."""
        self.dice_scores = []
        self.iou_scores = []
        self.pixel_accuracies = []
        self.sensitivities = []
        self.specificities = []
    
    def compute_dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Вычисление Dice коэффициента.
        """
        smooth = 1e-6
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()
    
    def compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Вычисление Intersection over Union (IoU).
        """
        smooth = 1e-6
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    def compute_pixel_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Вычисление точности пикселей.
        """
        correct = (pred == target).sum()
        total = target.numel()
        return (correct.float() / total).item()
    
    def compute_sensitivity_specificity(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
        """
        Вычисление чувствительности и специфичности.
        """
        # True Positive, False Positive, True Negative, False Negative
        tp = ((pred == 1) & (target == 1)).sum()
        fp = ((pred == 1) & (target == 0)).sum()
        tn = ((pred == 0) & (target == 0)).sum()
        fn = ((pred == 0) & (target == 1)).sum()
        
        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn + 1e-6)
        
        # Specificity
        specificity = tn / (tn + fp + 1e-6)
        
        return sensitivity.item(), specificity.item()
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Обновление метрик для одного батча.
        """
        # Конвертация в one-hot если нужно
        if pred.dim() == 4:  # [B, D, H, W] с классами
            pred_one_hot = F.one_hot(pred, self.num_classes).permute(0, 4, 1, 2, 3).float()
        else:  # [B, C, D, H, W] с вероятностями
            pred_one_hot = pred
        
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Метрики для каждого класса
        batch_dice = []
        batch_iou = []
        batch_sens = []
        batch_spec = []
        
        for class_idx in range(self.num_classes):
            pred_class = pred_one_hot[:, class_idx] > 0.5  # Бинаризация
            target_class = target_one_hot[:, class_idx]
            
            # Dice
            dice = self.compute_dice_coefficient(pred_class, target_class)
            batch_dice.append(dice)
            
            # IoU
            iou = self.compute_iou(pred_class, target_class)
            batch_iou.append(iou)
            
            # Sensitivity и Specificity
            sens, spec = self.compute_sensitivity_specificity(pred_class, target_class)
            batch_sens.append(sens)
            batch_spec.append(spec)
        
        # Pixel accuracy (для всех классов)
        pred_classes = torch.argmax(pred_one_hot, dim=1)
        target_classes = torch.argmax(target_one_hot, dim=1)
        pixel_acc = self.compute_pixel_accuracy(pred_classes, target_classes)
        
        # Накопление
        self.dice_scores.append(batch_dice)
        self.iou_scores.append(batch_iou)
        self.pixel_accuracies.append(pixel_acc)
        self.sensitivities.append(batch_sens)
        self.specificities.append(batch_spec)
    
    def compute_average_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Вычисление средних метрик.
        """
        if not self.dice_scores:
            return {}
        
        # Конвертация в numpy
        dice_array = np.array(self.dice_scores)  # [batch, classes]
        iou_array = np.array(self.iou_scores)
        sens_array = np.array(self.sensitivities)
        spec_array = np.array(self.specificities)
        
        metrics = {}
        
        # Метрики по классам
        for i, class_name in enumerate(self.class_names):
            metrics[class_name] = {
                'dice': float(np.mean(dice_array[:, i])),
                'iou': float(np.mean(iou_array[:, i])),
                'sensitivity': float(np.mean(sens_array[:, i])),
                'specificity': float(np.mean(spec_array[:, i]))
            }
        
        # Общие метрики
        metrics['overall'] = {
            'mean_dice': float(np.mean(dice_array)),
            'mean_iou': float(np.mean(iou_array)),
            'pixel_accuracy': float(np.mean(self.pixel_accuracies))
        }
        
        return metrics
    
    def print_metrics(self):
        """Печать метрик."""
        metrics = self.compute_average_metrics()
        
        if not metrics:
            print("Нет данных для вычисления метрик")
            return
        
        print("\n" + "="*60)
        print("METRICS REPORT")
        print("="*60)
        
        # Метрики по классам
        for class_name in self.class_names:
            class_metrics = metrics[class_name]
            print(f"\n{class_name}:")
            print(f"  Dice: {class_metrics['dice']:.4f}")
            print(f"  IoU: {class_metrics['iou']:.4f}")
            print(f"  Sensitivity: {class_metrics['sensitivity']:.4f}")
            print(f"  Specificity: {class_metrics['specificity']:.4f}")
        
        # Общие метрики
        overall = metrics['overall']
        print(f"\nOverall:")
        print(f"  Mean Dice: {overall['mean_dice']:.4f}")
        print(f"  Mean IoU: {overall['mean_iou']:.4f}")
        print(f"  Pixel Accuracy: {overall['pixel_accuracy']:.4f}")
        print("="*60)


def test_loss_functions():
    """
    Тестирование функций потерь.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Тестовые данные
    batch_size = 2
    num_classes = 3
    depth, height, width = 32, 64, 64
    
    # Предсказания (логиты)
    pred = torch.randn(batch_size, num_classes, depth, height, width).to(device)
    
    # Истинные метки
    target = torch.randint(0, num_classes, (batch_size, depth, height, width)).to(device)
    
    print(f"Предсказания: {pred.shape}")
    print(f"Цели: {target.shape}")
    
    # Тестирование функций потерь
    dice_loss = DiceLoss().to(device)
    focal_loss = FocalLoss().to(device)
    combo_loss = ComboLoss().to(device)
    
    with torch.no_grad():
        dice = dice_loss(pred, target)
        focal = focal_loss(pred, target)
        combo = combo_loss(pred, target)
        
        print(f"\nDice Loss: {dice.item():.4f}")
        print(f"Focal Loss: {focal.item():.4f}")
        print(f"Combo Loss: {combo.item():.4f}")
    
    # Тестирование метрик
    metrics = SegmentationMetrics(num_classes, ['Background', 'Kidney', 'Tumor'])
    
    # Обновление метрик
    pred_classes = torch.argmax(pred, dim=1)
    metrics.update(pred_classes, target)
    
    # Печать метрик
    metrics.print_metrics()
    
    print("\n✅ Тестирование функций потерь и метрик пройдено успешно!")


if __name__ == "__main__":
    test_loss_functions()
