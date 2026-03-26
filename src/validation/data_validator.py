import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidator:
    """Валидатор входных данных"""
    
    def __init__(self):
        self.ranges = {
            'age': (0, 100),
            'bmi': (10, 60),
            'weight_kg': (30, 200),
            'height_m': (1.0, 2.5)
        }
    
    def validate_patient_data(self, patient_data: Dict) -> Dict:
        """Валидация данных пациента"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 1. Проверка обязательных полей
        required_fields = [
            'age', 'bmi', 'sex_encoded',
            'kidney_left_center_x_mm', 'kidney_left_center_y_mm', 'kidney_left_center_z_mm',
            'kidney_right_center_x_mm', 'kidney_right_center_y_mm', 'kidney_right_center_z_mm'
        ]
        
        for field in required_fields:
            if field not in patient_data:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['is_valid'] = False
        
        # 2. Проверка диапазонов
        for field, (min_val, max_val) in self.ranges.items():
            if field in patient_data:
                value = patient_data[field]
                if not (min_val <= value <= max_val):
                    validation_result['errors'].append(
                        f"{field} out of range: {value} (expected {min_val}-{max_val})"
                    )
                    validation_result['is_valid'] = False
        
        # 3. Проверка логики почек
        if not self._validate_kidney_logic(patient_data):
            validation_result['warnings'].append("Kidney position logic may be incorrect")
        
        return validation_result
    
    def _validate_kidney_logic(self, patient_data: Dict) -> bool:
        """Проверка логики положения почек"""
        try:
            left_x = patient_data.get('kidney_left_center_x_mm', 0)
            right_x = patient_data.get('kidney_right_center_x_mm', 0)
            
            # Левая почка должна быть слева (отрицательный X)
            # Правая почка должна быть справа (положительный X)
            return left_x < 0 and right_x > 0
        except:
            return False

class ClinicalMetrics:
    """Клинические метрики"""
    
    def __init__(self):
        self.thresholds = [5, 10, 15]  # мм
        self.predictions_history = []
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Расчет клинических метрик"""
        metrics = {}
        
        # Общие метрики
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # Процент в пределах порогов
        for threshold in self.thresholds:
            metrics[f'within_{threshold}mm'] = np.mean(np.abs(y_true - y_pred) < threshold) * 100
        
        # Раздельно для почек
        if y_true.shape[1] >= 6:  # левая + правая почка
            metrics['left_kidney'] = self._kidney_metrics(y_true[:, :3], y_pred[:, :3])
            metrics['right_kidney'] = self._kidney_metrics(y_true[:, 3:], y_pred[:, 3:])
        
        return metrics
    
    def _kidney_metrics(self, y_true_kidney: np.ndarray, y_pred_kidney: np.ndarray) -> Dict:
        """Метрики для одной почки"""
        return {
            'mae': float(np.mean(np.abs(y_true_kidney - y_pred_kidney))),
            'within_5mm': float(np.mean(np.abs(y_true_kidney - y_pred_kidney) < 5) * 100)
        }
    
    def add_prediction(self, prediction: Dict, ground_truth: Optional[Dict] = None):
        """Добавление предсказания в историю"""
        self.predictions_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'ground_truth': ground_truth
        })
        
        # Ограничиваем размер истории
        if len(self.predictions_history) > 1000:
            self.predictions_history = self.predictions_history[-1000:]
    
    def get_summary_metrics(self) -> Dict:
        """Получение сводных метрик"""
        if not self.predictions_history:
            return {
                'total_predictions': 0,
                'average_confidence': 0.0,
                'success_rate': 0.0
            }
        
        total = len(self.predictions_history)
        successful = sum(1 for p in self.predictions_history if p['prediction'].get('success', False))
        confidences = [p['prediction'].get('confidence', 0) for p in self.predictions_history]
        
        return {
            'total_predictions': total,
            'successful_predictions': successful,
            'success_rate': (successful / total) * 100,
            'average_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }

class SystemLogger:
    """Логирование работы системы"""
    
    def __init__(self, log_file: str = "logs/kidney_ar_system.log"):
        self.log_file = log_file
        self.setup_logging()
        
        # Создаем директорию для логов
        Path(log_file).parent.mkdir(exist_ok=True)
    
    def setup_logging(self):
        """Настройка логирования"""
        # Создаем директорию для логов
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Создаем logger для файла
        self.file_logger = logging.getLogger('kidney_ar_file')
        self.file_logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Добавляем handler
        if not self.file_logger.handlers:
            self.file_logger.addHandler(file_handler)
    
    def log_input_data(self, patient_id: str, features: Dict, processing_time: float):
        """Логирование входных данных"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'type': 'INPUT',
            'patient_id': patient_id,
            'features_summary': self._summarize_features(features),
            'processing_time_ms': processing_time * 1000
        }
        
        self._write_log(log_data)
    
    def log_prediction(self, patient_id: str, prediction: np.ndarray, 
                      confidence: float, constraints_applied: bool):
        """Логирование предсказания"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'type': 'PREDICTION',
            'patient_id': patient_id,
            'prediction': prediction.tolist(),
            'confidence': confidence,
            'constraints_applied': constraints_applied
        }
        
        self._write_log(log_data)
    
    def log_error(self, patient_id: str, error_type: str, details: List[str]):
        """Логирование ошибок"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'type': 'ERROR',
            'patient_id': patient_id,
            'error_type': error_type,
            'details': details
        }
        
        self._write_log(log_data)
    
    def log_system_event(self, event_type: str, details: Dict):
        """Логирование системных событий"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'type': 'SYSTEM',
            'event_type': event_type,
            'details': details
        }
        
        self._write_log(log_data)
    
    def _summarize_features(self, features: Dict) -> Dict:
        """Суммаризация признаков для логирования"""
        summary = {}
        
        # Ключевые демографические данные
        for key in ['age', 'bmi', 'sex_encoded']:
            if key in features:
                summary[key] = features[key]
        
        # Позиция почек
        for kidney in ['left', 'right']:
            for axis in ['x', 'y', 'z']:
                key = f'kidney_{kidney}_center_{axis}_mm'
                if key in features:
                    summary[key] = features[key]
        
        return summary
    
    def _write_log(self, log_data: Dict):
        """Запись лога в файл"""
        try:
            self.file_logger.info(json.dumps(log_data, ensure_ascii=False))
        except Exception as e:
            print(f"Ошибка записи лога: {e}")
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict]:
        """Получение последних логов"""
        try:
            if not Path(self.log_file).exists():
                return []
            
            logs = []
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Парсим JSON из строки лога
                        parts = line.strip().split(' - ', 3)
                        if len(parts) >= 4:
                            log_json = parts[-1]
                            log_data = json.loads(log_json)
                            logs.append(log_data)
                            
                            if len(logs) >= limit:
                                break
                    except:
                        continue
            
            return logs[-limit:]
            
        except Exception as e:
            logger.error(f"Ошибка чтения логов: {e}")
            return []

if __name__ == "__main__":
    # Тестирование компонентов
    logging.basicConfig(level=logging.INFO)
    logger.info("Тестирование компонентов валидации и логирования")
    
    # Тестирование DataValidator
    validator = DataValidator()
    
    test_patient = {
        'age': 45,
        'bmi': 24.5,
        'sex_encoded': 1,
        'kidney_left_center_x_mm': -45.2,
        'kidney_left_center_y_mm': 18.5,
        'kidney_left_center_z_mm': 95.3,
        'kidney_right_center_x_mm': 52.1,
        'kidney_right_center_y_mm': 19.8,
        'kidney_right_center_z_mm': 96.7
    }
    
    validation = validator.validate_patient_data(test_patient)
    print(f"Валидация: {validation}")
    
    # Тестирование ClinicalMetrics
    metrics = ClinicalMetrics()
    
    # Добавляем тестовое предсказание
    test_prediction = {
        'success': True,
        'confidence': 0.85,
        'left_kidney': {'displacement': [5.0, -3.0, 2.0]},
        'right_kidney': {'displacement': [4.8, -2.9, 2.1]}
    }
    
    metrics.add_prediction(test_prediction)
    summary = metrics.get_summary_metrics()
    print(f"Метрики: {summary}")
    
    # Тестирование SystemLogger
    system_logger = SystemLogger("logs/test.log")
    
    system_logger.log_input_data("test_patient", test_patient, 0.15)
    system_logger.log_prediction("test_patient", np.array([5.0, -3.0, 2.0]), 0.85, True)
    system_logger.log_system_event("test_event", {"status": "success"})
    
    recent_logs = system_logger.get_recent_logs(5)
    print(f"Последние логи: {len(recent_logs)} записей")
    
    logger.info("Компоненты протестированы успешно")
