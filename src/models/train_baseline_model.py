#!/usr/bin/env python3
"""
БЛОК 2: BASELINE МОДЕЛЬ
Обучение Linear Regression как baseline
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pickle
import json
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaselineModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.feature_names = []
        self.target_names = []
        self.results = {}
        
    def load_data(self):
        """Загрузить train.csv и validation.csv"""
        logger.info("Загрузка данных...")
        
        train_df = pd.read_csv('data/processed/train.csv')
        val_df = pd.read_csv('data/processed/validation.csv')
        
        logger.info(f"Train: {len(train_df)} строк")
        logger.info(f"Validation: {len(val_df)} строк")
        
        return train_df, val_df
    
    def define_features_targets(self, df):
        """Определить список входных признаков и целевых переменных"""
        logger.info("Определение признаков и целей...")
        
        # Входные признаки (демография и исходные координаты на спине)
        self.feature_names = [
            'sex', 'age', 'bmi', 'body_type',
            'Y_upper_supine', 'Z_upper_supine'
        ]
        
        # Целевые переменные (координаты на боку)
        self.target_names = ['Y_upper_lateral', 'Z_upper_lateral']
        
        logger.info(f"Признаки: {len(self.feature_names)}")
        logger.info(f"Цели: {len(self.target_names)}")
        
        # Проверка наличия всех колонок
        missing_features = [col for col in self.feature_names if col not in df.columns]
        missing_targets = [col for col in self.target_names if col not in df.columns]
        
        if missing_features:
            raise ValueError(f"Отсутствуют признаки: {missing_features}")
        if missing_targets:
            raise ValueError(f"Отсутствуют цели: {missing_targets}")
        
        return self.feature_names, self.target_names
    
    def split_data(self, train_df, val_df):
        """Разделить на X_train, y_train, X_val, y_val"""
        logger.info("Разделение данных...")
        
        # Train
        X_train = train_df[self.feature_names]
        y_train = train_df[self.target_names]
        
        # Validation
        X_val = val_df[self.feature_names]
        y_val = val_df[self.target_names]
        
        logger.info(f"X_train: {X_train.shape}")
        logger.info(f"y_train: {y_train.shape}")
        logger.info(f"X_val: {X_val.shape}")
        logger.info(f"y_val: {y_val.shape}")
        
        return X_train, y_train, X_val, y_val
    
    def normalize_features(self, X_train, X_val):
        """Нормализовать признаки"""
        logger.info("Нормализация признаков...")
        
        # Обучение scaler на train
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Применение к validation
        X_val_scaled = self.scaler.transform(X_val)
        
        # Возврат в DataFrame для удобства
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=self.feature_names)
        
        logger.info("Нормализация завершена")
        
        return X_train_scaled, X_val_scaled
    
    def train_model(self, X_train, y_train):
        """Обучить Linear Regression"""
        logger.info("Обучение Linear Regression...")
        
        self.model.fit(X_train, y_train)
        
        logger.info("Модель обучена")
        
    def make_predictions(self, X_train, X_val):
        """Сделать предсказания"""
        logger.info("Предсказания...")
        
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Конвертация в DataFrame
        train_pred = pd.DataFrame(train_pred, columns=self.target_names)
        val_pred = pd.DataFrame(val_pred, columns=self.target_names)
        
        return train_pred, val_pred
    
    def calculate_metrics(self, y_train, train_pred, y_val, val_pred):
        """Вычислить метрики"""
        logger.info("Расчет метрик...")
        
        # MAE на train
        train_mae = mean_absolute_error(y_train, train_pred)
        train_mae_per_axis = {
            axis: mean_absolute_error(y_train[axis], train_pred[axis]) 
            for axis in self.target_names
        }
        
        # MAE на validation
        val_mae = mean_absolute_error(y_val, val_pred)
        val_mae_per_axis = {
            axis: mean_absolute_error(y_val[axis], val_pred[axis]) 
            for axis in self.target_names
        }
        
        # Сохранение результатов
        self.results = {
            'train_mae': float(train_mae),
            'train_mae_per_axis': train_mae_per_axis,
            'val_mae': float(val_mae),
            'val_mae_per_axis': val_mae_per_axis,
            'n_features': len(self.feature_names),
            'n_targets': len(self.target_names),
            'n_train_samples': len(y_train),
            'n_val_samples': len(y_val)
        }
        
        # Логирование
        logger.info(f"Train MAE: {train_mae:.3f} мм")
        logger.info(f"Validation MAE: {val_mae:.3f} мм")
        
        for axis in self.target_names:
            logger.info(f"  {axis}: Train={train_mae_per_axis[axis]:.3f}, Val={val_mae_per_axis[axis]:.3f}")
        
        return self.results
    
    def save_results(self):
        """Сохранить результаты и модель"""
        logger.info("Сохранение результатов...")
        
        # Сохранить scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info("✅ scaler.pkl сохранен")
        
        # Сохранить модель
        with open('models/baseline_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        logger.info("✅ baseline_model.pkl сохранен")
        
        # Сохранить метрики
        with open('models/baseline_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info("✅ baseline_results.json сохранен")
        
        # Сохранить названия признаков
        with open('models/feature_names.json', 'w', encoding='utf-8') as f:
            json.dump(self.feature_names, f, indent=2, ensure_ascii=False)
        logger.info("✅ feature_names.json сохранен")
        
        # Сохранить названия целей
        with open('models/target_names.json', 'w', encoding='utf-8') as f:
            json.dump(self.target_names, f, indent=2, ensure_ascii=False)
        logger.info("✅ target_names.json сохранен")
    
    def run(self):
        """Запустить весь процесс"""
        logger.info("=" * 80)
        logger.info("БЛОК 2: BASELINE МОДЕЛЬ")
        logger.info("=" * 80)
        
        try:
            # 1. Загрузка данных
            train_df, val_df = self.load_data()
            self.todo_list_update("block2_1", "completed")
            
            # 2. Определение признаков и целей
            self.define_features_targets(train_df)
            self.todo_list_update("block2_2", "completed")
            self.todo_list_update("block2_3", "completed")
            
            # 3. Разделение данных
            X_train, y_train, X_val, y_val = self.split_data(train_df, val_df)
            self.todo_list_update("block2_4", "completed")
            
            # 4. Нормализация
            X_train_scaled, X_val_scaled = self.normalize_features(X_train, X_val)
            self.todo_list_update("block2_5", "completed")
            self.todo_list_update("block2_6", "completed")
            
            # 5. Обучение модели
            self.train_model(X_train_scaled, y_train)
            self.todo_list_update("block2_7", "completed")
            
            # 6. Предсказания
            train_pred, val_pred = self.make_predictions(X_train_scaled, X_val_scaled)
            self.todo_list_update("block2_8", "completed")
            self.todo_list_update("block2_9", "completed")
            
            # 7. Метрики
            results = self.calculate_metrics(y_train, train_pred, y_val, val_pred)
            self.todo_list_update("block2_10", "completed")
            self.todo_list_update("block2_11", "completed")
            self.todo_list_update("block2_12", "completed")
            
            # 8. Сохранение
            self.save_results()
            self.todo_list_update("block2_13", "completed")
            self.todo_list_update("block2_14", "completed")
            
            # Итоговая сводка
            logger.info("=" * 80)
            logger.info("📊 ИТОГОВАЯ СВОДКА BASELINE МОДЕЛИ")
            logger.info("=" * 80)
            logger.info(f"✅ Train MAE: {results['train_mae']:.3f} мм")
            logger.info(f"✅ Validation MAE: {results['val_mae']:.3f} мм")
            logger.info(f"✅ Признаков: {results['n_features']}")
            logger.info(f"✅ Целевых переменных: {results['n_targets']}")
            logger.info("=" * 80)
            logger.info("✅ BASELINE МОДЕЛЬ ГОТОВА!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            raise
    
    def todo_list_update(self, task_id, status):
        """Обновить статус задачи (заглушка для интеграции)"""
        pass

def main():
    """Главная функция"""
    # Создание папки для моделей
    import os
    os.makedirs('models', exist_ok=True)
    
    # Запуск обучения
    trainer = BaselineModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main()
