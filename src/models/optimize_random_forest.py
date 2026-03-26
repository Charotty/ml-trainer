#!/usr/bin/env python3
"""
ОПТИМИЗАЦИЯ RANDOM FOREST
Улучшение лучшей модели для достижения максимальной точности
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
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

class RandomForestOptimizer:
    def __init__(self):
        self.scaler = None
        self.feature_names = []
        self.target_names = []
        self.best_model = None
        self.best_params = None
        self.results = {}
        
    def load_data_and_scaler(self):
        """Загрузить данные и scaler"""
        logger.info("Загрузка данных и scaler...")
        
        train_df = pd.read_csv('data/processed/train.csv')
        val_df = pd.read_csv('data/processed/validation.csv')
        
        with open('models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open('models/feature_names.json', 'r', encoding='utf-8') as f:
            self.feature_names = json.load(f)
        
        with open('models/target_names.json', 'r', encoding='utf-8') as f:
            self.target_names = json.load(f)
        
        return train_df, val_df
    
    def prepare_data(self, train_df, val_df):
        """Подготовить данные"""
        X_train = train_df[self.feature_names]
        y_train = train_df[self.target_names]
        X_val = val_df[self.feature_names]
        y_val = val_df[self.target_names]
        
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        return X_train_scaled, y_train, X_val_scaled, y_val
    
    def create_extensive_param_grid(self):
        """Создать расширенную сетку параметров"""
        logger.info("Создание расширенной сетки параметров...")
        
        param_grid = {
            'estimator__n_estimators': [50, 100, 150, 200, 300, 400, 500],
            'estimator__max_depth': [3, 5, 7, 10, 15, 20, None],
            'estimator__min_samples_split': [2, 5, 10, 15, 20],
            'estimator__min_samples_leaf': [1, 2, 4, 6, 8],
            'estimator__max_features': ['sqrt', 'log2', 0.6, 0.8, 1.0],
            'estimator__bootstrap': [True, False],
            'estimator__oob_score': [True, False],
            'estimator__max_samples': [None, 0.6, 0.8, 0.9]
        }
        
        return param_grid
    
    def create_conservative_param_grid(self):
        """Создать консервативную сетку для маленького датасета"""
        logger.info("Создание консервативной сетки параметров...")
        
        param_grid = {
            'estimator__n_estimators': [100, 200, 300, 400],
            'estimator__max_depth': [5, 7, 10, 15, 20],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__max_features': ['sqrt', 'log2', 0.8],
            'estimator__bootstrap': [True],
            'estimator__oob_score': [True]
        }
        
        return param_grid
    
    def optimize_with_random_search(self, X_train, y_train, X_val, y_val, param_grid):
        """Оптимизация с RandomizedSearchCV"""
        logger.info("Оптимизация с RandomizedSearchCV...")
        
        # Базовая модель
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        multi_rf = MultiOutputRegressor(rf)
        
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=multi_rf,
            param_distributions=param_grid,
            n_iter=100,  # Много итераций для поиска
            scoring='neg_mean_absolute_error',
            cv=5,  # 5-fold кросс-валидация
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Запуск RandomizedSearchCV...")
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        
        logger.info(f"Лучшие параметры: {best_params}")
        logger.info(f"Лучший score: {-random_search.best_score_:.4f}")
        
        return best_model, best_params
    
    def optimize_with_grid_search(self, X_train, y_train, X_val, y_val, param_grid):
        """Оптимизация с GridSearchCV (для финальной точности)"""
        logger.info("Финальная оптимизация с GridSearchCV...")
        
        # Уменьшенная сетка для GridSearch
        final_param_grid = {
            'estimator__n_estimators': [150, 200, 250, 300],
            'estimator__max_depth': [7, 10, 15],
            'estimator__min_samples_split': [2, 5],
            'estimator__min_samples_leaf': [1, 2],
            'estimator__max_features': ['sqrt', 0.8],
            'estimator__bootstrap': [True],
            'estimator__oob_score': [True]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        multi_rf = MultiOutputRegressor(rf)
        
        grid_search = GridSearchCV(
            estimator=multi_rf,
            param_grid=final_param_grid,
            scoring='neg_mean_absolute_error',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Запуск GridSearchCV...")
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Финальные лучшие параметры: {best_params}")
        logger.info(f"Финальный лучший score: {-grid_search.best_score_:.4f}")
        
        return best_model, best_params
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val):
        """Оценить модель"""
        logger.info("Оценка модели...")
        
        # Предсказания
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Метрики
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        # MAE по осям
        train_mae_per_axis = {}
        val_mae_per_axis = {}
        
        for i, target in enumerate(self.target_names):
            train_mae_per_axis[target] = mean_absolute_error(y_train[target], train_pred[:, i])
            val_mae_per_axis[target] = mean_absolute_error(y_val[target], val_pred[:, i])
        
        logger.info(f"Train MAE: {train_mae:.3f} мм")
        logger.info(f"Validation MAE: {val_mae:.3f} мм")
        
        for target in self.target_names:
            logger.info(f"  {target}: Train={train_mae_per_axis[target]:.3f}, Val={val_mae_per_axis[target]:.3f}")
        
        return {
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_mae_per_axis': train_mae_per_axis,
            'val_mae_per_axis': val_mae_per_axis
        }
    
    def compare_with_previous(self, val_mae):
        """Сравнить с предыдущими результатами"""
        logger.info("Сравнение с предыдущими результатами...")
        
        with open('models/baseline_results.json', 'r', encoding='utf-8') as f:
            baseline_results = json.load(f)
        
        with open('models/random_forest_results.json', 'r', encoding='utf-8') as f:
            old_rf_results = json.load(f)
        
        improvement_vs_baseline = ((baseline_results['val_mae'] - val_mae) / baseline_results['val_mae']) * 100
        improvement_vs_old_rf = ((old_rf_results['val_mae'] - val_mae) / old_rf_results['val_mae']) * 100
        
        logger.info(f"Улучшение vs Baseline: {improvement_vs_baseline:+.1f}%")
        logger.info(f"Улучшение vs старый Random Forest: {improvement_vs_old_rf:+.1f}%")
        
        return improvement_vs_baseline, improvement_vs_old_rf
    
    def save_results(self, metrics):
        """Сохранить результаты"""
        logger.info("Сохранение результатов...")
        
        # Сохранить модель
        with open('models/model_rf_optimized.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        logger.info("✅ model_rf_optimized.pkl сохранен")
        
        # Сохранить параметры
        with open('models/rf_optimized_params.json', 'w', encoding='utf-8') as f:
            json.dump(self.best_params, f, indent=2, ensure_ascii=False)
        logger.info("✅ rf_optimized_params.json сохранен")
        
        # Сохранить результаты
        self.results = {
            **metrics,
            'best_params': self.best_params,
            'n_features': len(self.feature_names),
            'n_targets': len(self.target_names),
            'optimization_method': 'RandomizedSearchCV + GridSearchCV',
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/rf_optimized_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info("✅ rf_optimized_results.json сохранен")
    
    def update_comparison_table(self, metrics):
        """Обновить таблицу сравнения"""
        logger.info("Обновление таблицы сравнения...")
        
        comparison_df = pd.read_csv('models/model_comparison.csv')
        
        new_row = {
            'Model': 'Random Forest Optimized',
            'Train MAE (mm)': metrics['train_mae'],
            'Val MAE (mm)': metrics['val_mae'],
            'Y_upper_lateral (mm)': metrics['val_mae_per_axis']['Y_upper_lateral'],
            'Z_upper_lateral (mm)': metrics['val_mae_per_axis']['Z_upper_lateral']
        }
        
        comparison_df = pd.concat([comparison_df, pd.DataFrame([new_row])], ignore_index=True)
        comparison_df.to_csv('models/model_comparison.csv', index=False)
        
        logger.info("📊 ФИНАЛЬНАЯ ТАБЛИЦА СРАВНЕНИЯ:")
        logger.info(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def run(self):
        """Запустить оптимизацию Random Forest"""
        logger.info("=" * 80)
        logger.info("ОПТИМИЗАЦИЯ RANDOM FOREST")
        logger.info("=" * 80)
        
        try:
            # 1. Загрузка данных
            train_df, val_df = self.load_data_and_scaler()
            
            # 2. Подготовка данных
            X_train, y_train, X_val, y_val = self.prepare_data(train_df, val_df)
            
            # 3. Создание сетки параметров
            param_grid = self.create_conservative_param_grid()
            
            # 4. Оптимизация с RandomizedSearchCV
            best_model_random, best_params_random = self.optimize_with_random_search(
                X_train, y_train, X_val, y_val, param_grid
            )
            
            # 5. Финальная оптимизация с GridSearchCV
            best_model_grid, best_params_grid = self.optimize_with_grid_search(
                X_train, y_train, X_val, y_val, param_grid
            )
            
            # 6. Выбор лучшей модели
            self.best_model = best_model_grid
            self.best_params = best_params_grid
            
            # 7. Оценка модели
            metrics = self.evaluate_model(self.best_model, X_train, y_train, X_val, y_val)
            
            # 8. Сравнение с предыдущими
            imp_baseline, imp_old_rf = self.compare_with_previous(metrics['val_mae'])
            
            # 9. Сохранение результатов
            self.save_results(metrics)
            
            # 10. Обновление таблицы
            comparison_df = self.update_comparison_table(metrics)
            
            # Итоговая сводка
            logger.info("=" * 80)
            logger.info("📊 ИТОГИ ОПТИМИЗАЦИИ RANDOM FOREST")
            logger.info("=" * 80)
            logger.info(f"✅ Train MAE: {metrics['train_mae']:.3f} мм")
            logger.info(f"✅ Validation MAE: {metrics['val_mae']:.3f} мм")
            logger.info(f"✅ Улучшение vs старый Random Forest: {imp_old_rf:+.1f}%")
            
            if imp_old_rf > 0:
                logger.info("🎉 Random Forest УЛУЧШЕН!")
                logger.info("🏆 ЭТО НОВАЯ ЛУЧШАЯ МОДЕЛЬ!")
            else:
                logger.info(f"⚠️ Улучшить не удалось, разница: {imp_old_rf:.1f}%")
            
            logger.info("=" * 80)
            logger.info("✅ ОПТИМИЗАЦИЯ RANDOM FOREST ЗАВЕРШЕНА!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            raise

def main():
    optimizer = RandomForestOptimizer()
    optimizer.run()

if __name__ == "__main__":
    main()
