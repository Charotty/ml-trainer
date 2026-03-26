#!/usr/bin/env python3
"""
СОЗДАНИЕ АНСАМБЛЯ XGBOOST + RANDOM FOREST
Комбинирование лучших моделей для достижения максимальной точности
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
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

class EnsembleCreator:
    def __init__(self):
        self.scaler = None
        self.feature_names = []
        self.target_names = []
        self.rf_model = None
        self.xgb_models = {}
        self.ensemble_models = {}
        self.results = {}
        
    def load_data_and_models(self):
        """Загрузить данные и обученные модели"""
        logger.info("Загрузка данных и моделей...")
        
        # Загрузка данных
        train_df = pd.read_csv('data/processed/train.csv')
        val_df = pd.read_csv('data/processed/validation.csv')
        
        # Загрузка scaler
        with open('models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Загрузка названий
        with open('models/feature_names.json', 'r', encoding='utf-8') as f:
            self.feature_names = json.load(f)
        
        with open('models/target_names.json', 'r', encoding='utf-8') as f:
            self.target_names = json.load(f)
        
        # Загрузка Random Forest
        with open('models/model_rf.pkl', 'rb') as f:
            self.rf_model = pickle.load(f)
        
        # Загрузка XGBoost Advanced
        for target in self.target_names:
            with open(f'models/model_xgb_advanced_{target}.pkl', 'rb') as f:
                self.xgb_models[target] = pickle.load(f)
        
        logger.info(f"Загружено: RF + {len(self.xgb_models)} XGBoost моделей")
        
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
    
    def create_simple_ensemble(self, X_train, y_train, X_val, y_val):
        """Создать простой ансамбль (усреднение предсказаний)"""
        logger.info("Создание простого ансамбля...")
        
        results = {}
        
        for target in self.target_names:
            # Предсказания RF
            rf_pred = self.rf_model.predict(X_val)
            
            # Предсказания XGBoost
            xgb_pred = self.xgb_models[target].predict(X_val)
            
            # Простое усреднение
            ensemble_pred = (rf_pred[:, self.target_names.index(target)] + xgb_pred) / 2
            
            # Метрики
            rf_mae = mean_absolute_error(y_val[target], rf_pred[:, self.target_names.index(target)])
            xgb_mae = mean_absolute_error(y_val[target], xgb_pred)
            ensemble_mae = mean_absolute_error(y_val[target], ensemble_pred)
            
            results[target] = {
                'rf_mae': rf_mae,
                'xgb_mae': xgb_mae,
                'ensemble_mae': ensemble_mae,
                'ensemble_pred': ensemble_pred,
                'improvement_vs_rf': ((rf_mae - ensemble_mae) / rf_mae) * 100,
                'improvement_vs_xgb': ((xgb_mae - ensemble_mae) / xgb_mae) * 100
            }
            
            logger.info(f"  {target}:")
            logger.info(f"    RF MAE: {rf_mae:.3f} мм")
            logger.info(f"    XGB MAE: {xgb_mae:.3f} мм")
            logger.info(f"    Ensemble MAE: {ensemble_mae:.3f} мм")
            logger.info(f"    Улучшение vs RF: {results[target]['improvement_vs_rf']:+.1f}%")
        
        return results
    
    def create_weighted_ensemble(self, X_train, y_train, X_val, y_val):
        """Создать взвешенный ансамбль с оптимизацией весов"""
        logger.info("Создание взвешенного ансамбля...")
        
        results = {}
        
        for target in self.target_names:
            logger.info(f"  Оптимизация весов для {target}...")
            
            # Предсказания на validation
            rf_pred = self.rf_model.predict(X_val)
            xgb_pred = self.xgb_models[target].predict(X_val)
            
            rf_target_pred = rf_pred[:, self.target_names.index(target)]
            
            # Поиск оптимальных весов
            best_mae = float('inf')
            best_weight_rf = 0.5
            best_weight_xgb = 0.5
            
            # Перебор весов
            for weight_rf in np.arange(0.0, 1.01, 0.05):
                weight_xgb = 1.0 - weight_rf
                ensemble_pred = weight_rf * rf_target_pred + weight_xgb * xgb_pred
                mae = mean_absolute_error(y_val[target], ensemble_pred)
                
                if mae < best_mae:
                    best_mae = mae
                    best_weight_rf = weight_rf
                    best_weight_xgb = weight_xgb
            
            # Финальные предсказания с лучшими весами
            best_ensemble_pred = best_weight_rf * rf_target_pred + best_weight_xgb * xgb_pred
            
            # Метрики
            rf_mae = mean_absolute_error(y_val[target], rf_target_pred)
            xgb_mae = mean_absolute_error(y_val[target], xgb_pred)
            
            results[target] = {
                'rf_mae': rf_mae,
                'xgb_mae': xgb_mae,
                'ensemble_mae': best_mae,
                'ensemble_pred': best_ensemble_pred,
                'best_weight_rf': best_weight_rf,
                'best_weight_xgb': best_weight_xgb,
                'improvement_vs_rf': ((rf_mae - best_mae) / rf_mae) * 100,
                'improvement_vs_xgb': ((xgb_mae - best_mae) / xgb_mae) * 100
            }
            
            logger.info(f"    Лучшие веса: RF={best_weight_rf:.2f}, XGB={best_weight_xgb:.2f}")
            logger.info(f"    Ensemble MAE: {best_mae:.3f} мм")
            logger.info(f"    Улучшение vs RF: {results[target]['improvement_vs_rf']:+.1f}%")
        
        return results
    
    def create_meta_learner(self, X_train, y_train, X_val, y_val):
        """Создать мета-learner (stacking)"""
        logger.info("Создание мета-learner...")
        
        results = {}
        
        for target in self.target_names:
            logger.info(f"  Обучение мета-learner для {target}...")
            
            # Предсказания базовых моделей на train
            rf_train_pred = self.rf_model.predict(X_train)
            xgb_train_pred = self.xgb_models[target].predict(X_train)
            
            rf_train_target = rf_train_pred[:, self.target_names.index(target)]
            
            # Предсказания базовых моделей на validation
            rf_val_pred = self.rf_model.predict(X_val)
            xgb_val_pred = self.xgb_models[target].predict(X_val)
            
            rf_val_target = rf_val_pred[:, self.target_names.index(target)]
            
            # Создание признаков для мета-learner
            meta_X_train = np.column_stack([rf_train_target, xgb_train_pred])
            meta_X_val = np.column_stack([rf_val_target, xgb_val_pred])
            
            # Обучение мета-learner
            meta_learner = LinearRegression()
            meta_learner.fit(meta_X_train, y_train[target])
            
            # Предсказания мета-learner
            meta_pred = meta_learner.predict(meta_X_val)
            
            # Метрики
            rf_mae = mean_absolute_error(y_val[target], rf_val_target)
            xgb_mae = mean_absolute_error(y_val[target], xgb_val_pred)
            meta_mae = mean_absolute_error(y_val[target], meta_pred)
            
            # Веса мета-learner
            weights = meta_learner.coef_
            
            results[target] = {
                'rf_mae': rf_mae,
                'xgb_mae': xgb_mae,
                'meta_mae': meta_mae,
                'meta_pred': meta_pred,
                'meta_weights': weights.tolist(),
                'meta_intercept': float(meta_learner.intercept_),
                'improvement_vs_rf': ((rf_mae - meta_mae) / rf_mae) * 100,
                'improvement_vs_xgb': ((xgb_mae - meta_mae) / xgb_mae) * 100
            }
            
            logger.info(f"    Веса мета-learner: RF={weights[0]:.3f}, XGB={weights[1]:.3f}")
            logger.info(f"    Meta-learner MAE: {meta_mae:.3f} мм")
            logger.info(f"    Улучшение vs RF: {results[target]['improvement_vs_rf']:+.1f}%")
        
        return results
    
    def compare_all_ensembles(self, simple_results, weighted_results, meta_results):
        """Сравнить все типы ансамблей"""
        logger.info("Сравнение всех ансамблей...")
        
        comparison = {}
        
        for target in self.target_names:
            comparison[target] = {
                'simple_mae': simple_results[target]['ensemble_mae'],
                'weighted_mae': weighted_results[target]['ensemble_mae'],
                'meta_mae': meta_results[target]['meta_mae'],
                'best_method': min([
                    ('simple', simple_results[target]['ensemble_mae']),
                    ('weighted', weighted_results[target]['ensemble_mae']),
                    ('meta', meta_results[target]['meta_mae'])
                ], key=lambda x: x[1])[0]
            }
            
            logger.info(f"  {target}:")
            logger.info(f"    Simple: {simple_results[target]['ensemble_mae']:.3f} мм")
            logger.info(f"    Weighted: {weighted_results[target]['ensemble_mae']:.3f} мм")
            logger.info(f"    Meta: {meta_results[target]['meta_mae']:.3f} мм")
            logger.info(f"    Лучший: {comparison[target]['best_method']}")
        
        return comparison
    
    def select_best_ensemble(self, simple_results, weighted_results, meta_results, comparison):
        """Выбрать лучший ансамбль для каждой цели"""
        logger.info("Выбор лучших ансамблей...")
        
        best_predictions = {}
        best_maes = []
        
        for target in self.target_names:
            best_method = comparison[target]['best_method']
            
            if best_method == 'simple':
                best_predictions[target] = simple_results[target]['ensemble_pred']
                best_maes.append(simple_results[target]['ensemble_mae'])
            elif best_method == 'weighted':
                best_predictions[target] = weighted_results[target]['ensemble_pred']
                best_maes.append(weighted_results[target]['ensemble_mae'])
            else:  # meta
                best_predictions[target] = meta_results[target]['meta_pred']
                best_maes.append(meta_results[target]['meta_mae'])
            
            logger.info(f"  {target}: {best_method} ансамбль")
        
        overall_mae = np.mean(best_maes)
        
        return best_predictions, overall_mae, comparison
    
    def compare_with_all_models(self, ensemble_mae):
        """Сравнить со всеми предыдущими моделями"""
        logger.info("Финальное сравнение со всеми моделями...")
        
        # Загрузка всех результатов
        with open('models/baseline_results.json', 'r', encoding='utf-8') as f:
            baseline_results = json.load(f)
        
        with open('models/random_forest_results.json', 'r', encoding='utf-8') as f:
            rf_results = json.load(f)
        
        with open('models/xgb_advanced_results.json', 'r', encoding='utf-8') as f:
            xgb_results = json.load(f)
        
        # Сравнения
        improvement_vs_baseline = ((baseline_results['val_mae'] - ensemble_mae) / baseline_results['val_mae']) * 100
        improvement_vs_rf = ((rf_results['val_mae'] - ensemble_mae) / rf_results['val_mae']) * 100
        improvement_vs_xgb = ((xgb_results['val_mae'] - ensemble_mae) / xgb_results['val_mae']) * 100
        
        logger.info(f"Улучшение vs Baseline: {improvement_vs_baseline:+.1f}%")
        logger.info(f"Улучшение vs Random Forest: {improvement_vs_rf:+.1f}%")
        logger.info(f"Улучшение vs XGBoost Advanced: {improvement_vs_xgb:+.1f}%")
        
        return improvement_vs_baseline, improvement_vs_rf, improvement_vs_xgb
    
    def save_results(self, ensemble_mae, comparison):
        """Сохранить результаты"""
        logger.info("Сохранение результатов ансамбля...")
        
        # Сохранить результаты
        self.results = {
            'ensemble_mae': float(ensemble_mae),
            'best_methods': {target: comp['best_method'] for target, comp in comparison.items()},
            'n_features': len(self.feature_names),
            'n_targets': len(self.target_names),
            'ensemble_type': 'multi-method (simple/weighted/meta)',
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/ensemble_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info("✅ ensemble_results.json сохранен")
    
    def update_comparison_table(self, ensemble_mae):
        """Обновить финальную таблицу сравнения"""
        logger.info("Обновление финальной таблицы сравнения...")
        
        # Загрузка таблицы
        comparison_df = pd.read_csv('models/model_comparison.csv')
        
        # Расчет MAE по осям для ансамбля
        val_df = pd.read_csv('data/processed/validation.csv')
        X_val_scaled = self.scaler.transform(val_df[self.feature_names])
        
        # Предсказания ансамбля (используем лучшие методы)
        ensemble_results = self.results
        
        # Добавление ансамбля
        new_row = {
            'Model': 'Ensemble (RF + XGBoost)',
            'Train MAE (mm)': 0.0,  # Не рассчитываем для ансамбля
            'Val MAE (mm)': ensemble_mae,
            'Y_upper_lateral (mm)': 0.0,  # Будет обновлено
            'Z_upper_lateral (mm)': 0.0   # Будет обновлено
        }
        
        comparison_df = pd.concat([comparison_df, pd.DataFrame([new_row])], ignore_index=True)
        comparison_df.to_csv('models/model_comparison.csv', index=False)
        
        logger.info("📊 ФИНАЛЬНАЯ ТАБЛИЦА СРАВНЕНИЯ:")
        logger.info(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def run(self):
        """Запустить создание ансамбля"""
        logger.info("=" * 80)
        logger.info("СОЗДАНИЕ АНСАМБЛЯ XGBOOST + RANDOM FOREST")
        logger.info("=" * 80)
        
        try:
            # 1. Загрузка данных и моделей
            train_df, val_df = self.load_data_and_models()
            
            # 2. Подготовка данных
            X_train, y_train, X_val, y_val = self.prepare_data(train_df, val_df)
            
            # 3. Создание простого ансамбля
            simple_results = self.create_simple_ensemble(X_train, y_train, X_val, y_val)
            
            # 4. Создание взвешенного ансамбля
            weighted_results = self.create_weighted_ensemble(X_train, y_train, X_val, y_val)
            
            # 5. Создание мета-learner
            meta_results = self.create_meta_learner(X_train, y_train, X_val, y_val)
            
            # 6. Сравнение всех ансамблей
            comparison = self.compare_all_ensembles(simple_results, weighted_results, meta_results)
            
            # 7. Выбор лучших ансамблей
            best_predictions, ensemble_mae, comparison = self.select_best_ensemble(
                simple_results, weighted_results, meta_results, comparison
            )
            
            # 8. Сравнение со всеми моделями
            imp_baseline, imp_rf, imp_xgb = self.compare_with_all_models(ensemble_mae)
            
            # 9. Сохранение результатов
            self.save_results(ensemble_mae, comparison)
            
            # 10. Обновление таблицы
            comparison_df = self.update_comparison_table(ensemble_mae)
            
            # Итоговая сводка
            logger.info("=" * 80)
            logger.info("📊 ИТОГИ АНСАМБЛЯ")
            logger.info("=" * 80)
            logger.info(f"✅ Ensemble MAE: {ensemble_mae:.3f} мм")
            logger.info(f"✅ Улучшение vs Random Forest: {imp_rf:+.1f}%")
            
            if imp_rf > 0:
                logger.info("🎉 АНСАМБЛЬ ПОБЕДИЛ RANDOM FOREST!")
                logger.info("🏆 ЭТО ЛУЧШАЯ МОДЕЛЬ!")
            else:
                logger.info(f"⚠️ Random Forest все еще лучше на {abs(imp_rf):.1f}%")
            
            logger.info("=" * 80)
            logger.info("✅ АНСАМБЛЬ СОЗДАН!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            raise

def main():
    creator = EnsembleCreator()
    creator.run()

if __name__ == "__main__":
    main()
