#!/usr/bin/env python3
"""
ПРОДВИНУТАЯ ОПТИМИЗАЦИЯ XGBOOST
Множественные подходы для достижения лучших результатов
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import VotingRegressor
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

class AdvancedXGBoostOptimizer:
    def __init__(self):
        self.scaler = None
        self.feature_names = []
        self.target_names = []
        self.best_models = {}
        self.ensemble_models = {}
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
        
        return train_df, val_df
    
    def prepare_data(self, train_df, val_df):
        """Подготовить данные"""
        X_train = train_df[self.feature_names]
        y_train = train_df[self.target_names]
        X_val = val_df[self.feature_names]
        y_val = val_df[self.target_names]
        
        # Нормализация
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        return X_train_scaled, y_train, X_val_scaled, y_val
    
    def approach1_conservative(self, X_train, y_train, X_val, y_val, target):
        """Подход 1: Консервативные параметры для маленького датасета"""
        logger.info(f"  Подход 1 (консервативный) для {target}...")
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [2, 3, 4],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0]
        }
        
        model = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=30,
            scoring='neg_mean_absolute_error',
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train[target])
        
        best_model = search.best_estimator_
        val_pred = best_model.predict(X_val)
        val_mae = mean_absolute_error(y_val[target], val_pred)
        
        logger.info(f"    Val MAE: {val_mae:.3f} мм")
        return best_model, val_mae, search.best_params_
    
    def approach2_aggressive(self, X_train, y_train, X_val, y_val, target):
        """Подход 2: Агрессивные параметры с сильной регуляризацией"""
        logger.info(f"  Подход 2 (агрессивный) для {target}...")
        
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.03, 0.05],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 2, 4],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1.0, 10.0],
            'reg_lambda': [0.1, 0.5, 1.0, 5.0, 10.0],
            'gamma': [0, 0.1, 0.5, 1.0]  # Минимальная потеря для сплита
        }
        
        model = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=50,
            scoring='neg_mean_absolute_error',
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train[target])
        
        best_model = search.best_estimator_
        val_pred = best_model.predict(X_val)
        val_mae = mean_absolute_error(y_train[target], best_model.predict(X_train))  # Train MAE для контроля переобучения
        val_mae_test = mean_absolute_error(y_val[target], val_pred)
        
        logger.info(f"    Train MAE: {val_mae:.3f} мм, Val MAE: {val_mae_test:.3f} мм")
        return best_model, val_mae_test, search.best_params_
    
    def approach3_ensemble(self, models, X_val):
        """Подход 3: Ансамбль из нескольких моделей"""
        logger.info("  Подход 3 (ансамбль)...")
        
        # Создание ансамбля из разных моделей
        ensemble_models = []
        for name, model in models.items():
            ensemble_models.append((name, model))
        
        if len(ensemble_models) > 1:
            voting_regressor = VotingRegressor(ensemble_models)
            return voting_regressor
        else:
            return list(models.values())[0]
    
    def approach4_target_specific(self, X_train, y_train, X_val, y_val, target):
        """Подход 4: Специфичные параметры для каждой цели"""
        logger.info(f"  Подход 4 (target-specific) для {target}...")
        
        if 'Y_upper_lateral' in target:
            # Для Y-координаты - более сложная модель
            param_grid = {
                'n_estimators': [200, 300, 400],
                'learning_rate': [0.01, 0.03, 0.05],
                'max_depth': [4, 5, 6, 8],
                'min_child_weight': [1, 2, 3],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [0.5, 1.0, 2.0]
            }
        else:
            # Для Z-координаты - простая модель с сильной регуляризацией
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.1, 0.2, 0.3],
                'max_depth': [2, 3, 4],
                'min_child_weight': [1, 2, 5],
                'subsample': [0.5, 0.6, 0.7],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0.1, 1.0, 10.0],
                'reg_lambda': [1.0, 5.0, 10.0],
                'gamma': [0, 0.1, 0.5]
            }
        
        model = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=40,
            scoring='neg_mean_absolute_error',
            cv=4,  # Больше фолдов для маленького датасета
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train[target])
        
        best_model = search.best_estimator_
        val_pred = best_model.predict(X_val)
        val_mae = mean_absolute_error(y_val[target], val_pred)
        
        logger.info(f"    Val MAE: {val_mae:.3f} мм")
        return best_model, val_mae, search.best_params_
    
    def optimize_all_approaches(self, X_train, y_train, X_val, y_val):
        """Оптимизация всеми подходами"""
        logger.info("Многоподходная оптимизация...")
        
        all_results = {}
        
        for target in self.target_names:
            logger.info(f"Оптимизация {target}...")
            
            # Подход 1: Консервативный
            model1, mae1, params1 = self.approach1_conservative(X_train, y_train, X_val, y_val, target)
            
            # Подход 2: Агрессивный
            model2, mae2, params2 = self.approach2_aggressive(X_train, y_train, X_val, y_val, target)
            
            # Подход 4: Target-specific
            model4, mae4, params4 = self.approach4_target_specific(X_train, y_train, X_val, y_val, target)
            
            # Выбор лучшей модели
            models = {
                'conservative': (model1, mae1, params1),
                'aggressive': (model2, mae2, params2),
                'target_specific': (model4, mae4, params4)
            }
            
            best_approach = min(models, key=lambda x: models[x][1])
            best_model, best_mae, best_params = models[best_approach]
            
            logger.info(f"  Лучший подход: {best_approach} с MAE: {best_mae:.3f} мм")
            
            all_results[target] = {
                'best_model': best_model,
                'best_mae': best_mae,
                'best_approach': best_approach,
                'best_params': best_params,
                'all_maes': {name: mae for name, (_, mae, _) in models.items()}
            }
            
            self.best_models[target] = best_model
        
        return all_results
    
    def create_meta_ensemble(self, X_train, y_train, X_val, y_val):
        """Создать мета-ансамбль из лучших моделей"""
        logger.info("Создание мета-ансамбля...")
        
        # Для каждой цели создаем ансамбль из топ-3 подходов
        meta_models = {}
        
        for target in self.target_names:
            # Обучаем несколько моделей с разными seed
            models = []
            for seed in [42, 123, 456, 789, 999]:
                model = XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4 if 'Y' in target else 3,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=-1,
                    objective='reg:squarederror'
                )
                model.fit(X_train, y_train[target])
                models.append((f'xgb_{seed}', model))
            
            # Создаем VotingRegressor
            if len(models) > 1:
                meta_models[target] = VotingRegressor(models)
            else:
                meta_models[target] = models[0][1]
        
        return meta_models
    
    def evaluate_models(self, X_val, y_val):
        """Оценить все модели"""
        logger.info("Оценка моделей...")
        
        results = {}
        total_train_mae = []
        total_val_mae = []
        
        for target in self.target_names:
            # Предсказания
            val_pred = self.best_models[target].predict(X_val)
            
            # Метрики
            val_mae = mean_absolute_error(y_val[target], val_pred)
            
            results[target] = {
                'val_mae': val_mae,
                'val_pred': val_pred
            }
            
            total_val_mae.append(val_mae)
            
            logger.info(f"  {target}: Val MAE = {val_mae:.3f} мм")
        
        # Общие метрики
        overall_val_mae = np.mean(total_val_mae)
        
        logger.info(f"Общий Val MAE: {overall_val_mae:.3f} мм")
        
        return results, overall_val_mae
    
    def compare_with_all(self, val_mae):
        """Сравнить со всеми предыдущими результатами"""
        logger.info("Сравнение со всеми моделями...")
        
        # Загрузка всех результатов
        with open('models/baseline_results.json', 'r', encoding='utf-8') as f:
            baseline_results = json.load(f)
        
        with open('models/random_forest_results.json', 'r', encoding='utf-8') as f:
            rf_results = json.load(f)
        
        with open('models/xgb_optimized_results.json', 'r', encoding='utf-8') as f:
            old_xgb_results = json.load(f)
        
        # Сравнения
        improvement_vs_baseline = ((baseline_results['val_mae'] - val_mae) / baseline_results['val_mae']) * 100
        improvement_vs_rf = ((rf_results['val_mae'] - val_mae) / rf_results['val_mae']) * 100
        improvement_vs_old_xgb = ((old_xgb_results['val_mae'] - val_mae) / old_xgb_results['val_mae']) * 100
        
        logger.info(f"Улучшение vs Baseline: {improvement_vs_baseline:+.1f}%")
        logger.info(f"Улучшение vs Random Forest: {improvement_vs_rf:+.1f}%")
        logger.info(f"Улучшение vs старый XGBoost: {improvement_vs_old_xgb:+.1f}%")
        
        return improvement_vs_baseline, improvement_vs_rf, improvement_vs_old_xgb
    
    def save_results(self, val_mae):
        """Сохранить результаты"""
        logger.info("Сохранение результатов...")
        
        # Сохранить модели
        for target in self.target_names:
            model_filename = f'models/model_xgb_advanced_{target}.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump(self.best_models[target], f)
            logger.info(f"✅ {model_filename} сохранен")
        
        # Сохранить результаты
        self.results = {
            'val_mae': float(val_mae),
            'n_features': len(self.feature_names),
            'n_targets': len(self.target_names),
            'optimization_method': 'Multi-approach advanced',
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/xgb_advanced_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info("✅ xgb_advanced_results.json сохранен")
    
    def update_comparison_table(self, val_mae):
        """Обновить таблицу сравнения"""
        # Загрузка таблицы
        comparison_df = pd.read_csv('models/model_comparison.csv')
        
        # Расчет MAE по осям
        val_df = pd.read_csv('data/processed/validation.csv')
        X_val_scaled = self.scaler.transform(val_df[self.feature_names])
        
        y_mae = mean_absolute_error(
            val_df['Y_upper_lateral'],
            self.best_models['Y_upper_lateral'].predict(X_val_scaled)
        )
        
        z_mae = mean_absolute_error(
            val_df['Z_upper_lateral'],
            self.best_models['Z_upper_lateral'].predict(X_val_scaled)
        )
        
        # Добавление новой модели
        new_row = {
            'Model': 'XGBoost Advanced',
            'Train MAE (mm)': 0.0,  # Будет рассчитано
            'Val MAE (mm)': val_mae,
            'Y_upper_lateral (mm)': y_mae,
            'Z_upper_lateral (mm)': z_mae
        }
        
        comparison_df = pd.concat([comparison_df, pd.DataFrame([new_row])], ignore_index=True)
        comparison_df.to_csv('models/model_comparison.csv', index=False)
        
        logger.info("📊 ФИНАЛЬНАЯ ТАБЛИЦА СРАВНЕНИЯ:")
        logger.info(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def run(self):
        """Запустить продвинутую оптимизацию"""
        logger.info("=" * 80)
        logger.info("ПРОДВИНУТАЯ ОПТИМИЗАЦИЯ XGBOOST")
        logger.info("=" * 80)
        
        try:
            # 1. Загрузка данных
            train_df, val_df = self.load_data_and_scaler()
            
            # 2. Подготовка данных
            X_train, y_train, X_val, y_val = self.prepare_data(train_df, val_df)
            
            # 3. Многоподходная оптимизация
            all_results = self.optimize_all_approaches(X_train, y_train, X_val, y_val)
            
            # 4. Создание мета-ансамбля (опционально)
            # meta_models = self.create_meta_ensemble(X_train, y_train, X_val, y_val)
            
            # 5. Оценка моделей
            results, val_mae = self.evaluate_models(X_val, y_val)
            
            # 6. Сравнение со всеми
            imp_baseline, imp_rf, imp_old_xgb = self.compare_with_all(val_mae)
            
            # 7. Сохранение результатов
            self.save_results(val_mae)
            
            # 8. Обновление таблицы
            comparison_df = self.update_comparison_table(val_mae)
            
            # Итоговая сводка
            logger.info("=" * 80)
            logger.info("📊 ИТОГИ ПРОДВИНУТОЙ ОПТИМИЗАЦИИ")
            logger.info("=" * 80)
            logger.info(f"✅ Val MAE: {val_mae:.3f} мм")
            logger.info(f"✅ Улучшение vs Random Forest: {imp_rf:+.1f}%")
            
            if imp_rf > 0:
                logger.info("🎉 XGBoost ПОБЕДИЛ RANDOM FOREST!")
            else:
                logger.info(f"⚠️ Random Forest все еще лучше на {abs(imp_rf):.1f}%")
            
            logger.info("=" * 80)
            logger.info("✅ ПРОДВИНУТАЯ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            raise

def main():
    optimizer = AdvancedXGBoostOptimizer()
    optimizer.run()

if __name__ == "__main__":
    main()
