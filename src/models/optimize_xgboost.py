#!/usr/bin/env python3
"""
ОПТИМИЗАЦИЯ XGBOOST ПАРАМЕТРОВ
Подбор оптимальных гиперпараметров для улучшения результатов
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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

class XGBoostOptimizer:
    def __init__(self):
        self.scaler = None
        self.feature_names = []
        self.target_names = []
        self.best_models = {}
        self.best_params = {}
        self.results = {}
        
    def load_data_and_scaler(self):
        """Загрузить данные и scaler"""
        logger.info("Загрузка данных и scaler...")
        
        # Загрузка данных
        train_df = pd.read_csv('data/processed/train.csv')
        val_df = pd.read_csv('data/processed/validation.csv')
        
        # Загрузка scaler
        with open('models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Загрузка названий признаков и целей
        with open('models/feature_names.json', 'r', encoding='utf-8') as f:
            self.feature_names = json.load(f)
        
        with open('models/target_names.json', 'r', encoding='utf-8') as f:
            self.target_names = json.load(f)
        
        logger.info(f"Train: {len(train_df)} строк")
        logger.info(f"Validation: {len(val_df)} строк")
        
        return train_df, val_df
    
    def prepare_data(self, train_df, val_df):
        """Подготовить данные"""
        logger.info("Подготовка данных...")
        
        # Разделение
        X_train = train_df[self.feature_names]
        y_train = train_df[self.target_names]
        X_val = val_df[self.feature_names]
        y_val = val_df[self.target_names]
        
        # Нормализация
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        return X_train_scaled, y_train, X_val_scaled, y_val
    
    def create_param_grid(self):
        """Создать сетку параметров для поиска"""
        logger.info("Создание сетки параметров...")
        
        # Более консервативные параметры для маленького датасета
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 8],
            'min_child_weight': [1, 2, 3, 5],
            'subsample': [0.6, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1.0],  # L1 регуляризация
            'reg_lambda': [0.5, 1.0, 2.0, 5.0]  # L2 регуляризация
        }
        
        return param_grid
    
    def optimize_single_target(self, X_train, y_train, X_val, y_val, target):
        """Оптимизировать параметры для одной целевой переменной"""
        logger.info(f"Оптимизация параметров для {target}...")
        
        # Базовая модель
        base_model = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )
        
        # Параметры для поиска
        param_grid = self.create_param_grid()
        
        # RandomizedSearchCV для быстрого поиска
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=50,  # Количество комбинаций для проб
            scoring='neg_mean_absolute_error',
            cv=3,  # 3-fold кросс-валидация
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # Поиск лучших параметров
        random_search.fit(X_train, y_train[target])
        
        # Лучшая модель
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        
        # Предсказания и оценка
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train[target], train_pred)
        val_mae = mean_absolute_error(y_val[target], val_pred)
        
        logger.info(f"  Лучшие параметры: {best_params}")
        logger.info(f"  Train MAE: {train_mae:.3f} мм")
        logger.info(f"  Val MAE: {val_mae:.3f} мм")
        
        return best_model, best_params, train_mae, val_mae
    
    def optimize_all_targets(self, X_train, y_train, X_val, y_val):
        """Оптимизировать параметры для всех целей"""
        logger.info("Оптимизация параметров для всех целей...")
        
        all_train_mae = []
        all_val_mae = []
        
        for target in self.target_names:
            best_model, best_params, train_mae, val_mae = self.optimize_single_target(
                X_train, y_train, X_val, y_val, target
            )
            
            self.best_models[target] = best_model
            self.best_params[target] = best_params
            all_train_mae.append(train_mae)
            all_val_mae.append(val_mae)
        
        # Общие метрики
        total_train_mae = np.mean(all_train_mae)
        total_val_mae = np.mean(all_val_mae)
        
        logger.info(f"✅ Общий Train MAE: {total_train_mae:.3f} мм")
        logger.info(f"✅ Общий Val MAE: {total_val_mae:.3f} мм")
        
        return total_train_mae, total_val_mae
    
    def compare_with_previous(self, val_mae):
        """Сравнить с предыдущими результатами"""
        logger.info("Сравнение с предыдущими результатами...")
        
        # Загрузка предыдущих результатов
        with open('models/xgboost_results.json', 'r', encoding='utf-8') as f:
            old_xgb_results = json.load(f)
        
        with open('models/random_forest_results.json', 'r', encoding='utf-8') as f:
            rf_results = json.load(f)
        
        # Сравнения
        improvement_vs_old_xgb = ((old_xgb_results['val_mae'] - val_mae) / old_xgb_results['val_mae']) * 100
        improvement_vs_rf = ((rf_results['val_mae'] - val_mae) / rf_results['val_mae']) * 100
        
        logger.info(f"Улучшение vs старый XGBoost: {improvement_vs_old_xgb:+.1f}%")
        logger.info(f"Улучшение vs Random Forest: {improvement_vs_rf:+.1f}%")
        
        return improvement_vs_old_xgb, improvement_vs_rf
    
    def save_results(self, train_mae, val_mae):
        """Сохранить результаты"""
        logger.info("Сохранение результатов...")
        
        # Сохранить оптимизированные модели
        for target in self.target_names:
            model_filename = f'models/model_xgb_optimized_{target}.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump(self.best_models[target], f)
            logger.info(f"✅ {model_filename} сохранен")
        
        # Сохранить параметры
        with open('models/xgb_optimized_params.json', 'w', encoding='utf-8') as f:
            json.dump(self.best_params, f, indent=2, ensure_ascii=False)
        logger.info("✅ xgb_optimized_params.json сохранен")
        
        # Сохранить результаты
        self.results = {
            'train_mae': float(train_mae),
            'val_mae': float(val_mae),
            'best_params': self.best_params,
            'n_features': len(self.feature_names),
            'n_targets': len(self.target_names),
            'optimization_method': 'RandomizedSearchCV',
            'n_iter': 50,
            'cv_folds': 3
        }
        
        with open('models/xgb_optimized_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info("✅ xgb_optimized_results.json сохранен")
    
    def update_comparison_table(self, val_mae):
        """Обновить таблицу сравнения"""
        logger.info("Обновление таблицы сравнения...")
        
        # Загрузка текущей таблицы
        comparison_df = pd.read_csv('models/model_comparison.csv')
        
        # Добавление оптимизированного XGBoost
        new_row = {
            'Model': 'XGBoost Optimized',
            'Train MAE (mm)': self.results['train_mae'],
            'Val MAE (mm)': val_mae,
            'Y_upper_lateral (mm)': mean_absolute_error(
                pd.read_csv('data/processed/validation.csv')['Y_upper_lateral'],
                self.best_models['Y_upper_lateral'].predict(
                    self.scaler.transform(
                        pd.read_csv('data/processed/validation.csv')[self.feature_names]
                    )
                )
            ),
            'Z_upper_lateral (mm)': mean_absolute_error(
                pd.read_csv('data/processed/validation.csv')['Z_upper_lateral'],
                self.best_models['Z_upper_lateral'].predict(
                    self.scaler.transform(
                        pd.read_csv('data/processed/validation.csv')[self.feature_names]
                    )
                )
            )
        }
        
        # Добавление строки
        comparison_df = pd.concat([comparison_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Сохранение обновленной таблицы
        comparison_df.to_csv('models/model_comparison.csv', index=False)
        
        # Логирование
        logger.info("📊 ОБНОВЛЕННАЯ ТАБЛИЦА СРАВНЕНИЯ:")
        logger.info(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def run(self):
        """Запустить оптимизацию"""
        logger.info("=" * 80)
        logger.info("ОПТИМИЗАЦИЯ XGBOOST ПАРАМЕТРОВ")
        logger.info("=" * 80)
        
        try:
            # 1. Загрузка данных
            train_df, val_df = self.load_data_and_scaler()
            
            # 2. Подготовка данных
            X_train, y_train, X_val, y_val = self.prepare_data(train_df, val_df)
            
            # 3. Оптимизация параметров
            train_mae, val_mae = self.optimize_all_targets(X_train, y_train, X_val, y_val)
            
            # 4. Сравнение с предыдущими результатами
            improvement_vs_old_xgb, improvement_vs_rf = self.compare_with_previous(val_mae)
            
            # 5. Сохранение результатов
            self.save_results(train_mae, val_mae)
            
            # 6. Обновление таблицы сравнения
            comparison_df = self.update_comparison_table(val_mae)
            
            # Итоговая сводка
            logger.info("=" * 80)
            logger.info("📊 ИТОГИ ОПТИМИЗАЦИИ XGBOOST")
            logger.info("=" * 80)
            logger.info(f"✅ Train MAE: {train_mae:.3f} мм")
            logger.info(f"✅ Validation MAE: {val_mae:.3f} мм")
            logger.info(f"✅ Улучшение vs старый XGBoost: {improvement_vs_old_xgb:+.1f}%")
            logger.info(f"✅ Улучшение vs Random Forest: {improvement_vs_rf:+.1f}%")
            
            if improvement_vs_rf > 0:
                logger.info("🎉 XGBoost теперь лучше Random Forest!")
            else:
                logger.info("⚠️ Random Forest все еще лучше")
            
            logger.info("=" * 80)
            logger.info("✅ ОПТИМИЗАЦИЯ XGBOOST ЗАВЕРШЕНА!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            raise

def main():
    """Главная функция"""
    optimizer = XGBoostOptimizer()
    optimizer.run()

if __name__ == "__main__":
    main()
