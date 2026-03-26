"""Clinical metrics for kidney displacement prediction"""

import numpy as np
from typing import Dict, List, Optional

class ClinicalMetrics:
    """Класс для расчета клинических метрик"""
    
    @staticmethod
    def calculate_mae(predictions: List[float], targets: List[float]) -> float:
        """Расчет Mean Absolute Error"""
        return np.mean(np.abs(np.array(predictions) - np.array(targets)))
    
    @staticmethod
    def calculate_accuracy_within_threshold(
        predictions: List[float], 
        targets: List[float], 
        threshold: float
    ) -> float:
        """Расчет точности в пределах порога"""
        errors = np.abs(np.array(predictions) - np.array(targets))
        return np.mean(errors <= threshold) * 100
    
    @staticmethod
    def calculate_clinical_metrics(predictions: List[float], targets: List[float]) -> Dict:
        """Расчет полного набора клинических метрик"""
        return {
            'mae': ClinicalMetrics.calculate_mae(predictions, targets),
            'accuracy_5mm': ClinicalMetrics.calculate_accuracy_within_threshold(predictions, targets, 5.0),
            'accuracy_10mm': ClinicalMetrics.calculate_accuracy_within_threshold(predictions, targets, 10.0)
        }
