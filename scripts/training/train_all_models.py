#!/usr/bin/env python3
"""
ОБУЧЕНИЕ И СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ
Linear Regression, Random Forest, XGBoost, Ensemble
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import pickle
import json
import logging
from datetime import datetime
from src.utils.imputer import MedianImputerKeepAll

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.imputer = MedianImputerKeepAll(fill_value_if_all_nan=0.0)
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_names = []
        self.target_names = []
        self.production_dir = Path('models') / 'production'

    def load_data(self):
        """Загрузить данные"""
        logger.info("Загрузка данных...")
        
        train_df = pd.read_csv('data/processed/train.csv')
        val_df = pd.read_csv('data/processed/validation.csv')
        test_df = pd.read_csv('data/processed/test.csv')
        
        with open('data/processed/feature_names.json', 'r', encoding='utf-8') as f:
            raw_features = json.load(f)
        
        with open('data/processed/target_names.json', 'r', encoding='utf-8') as f:
            self.target_names = json.load(f)

        # Prevent leakage: when predicting delta_* targets, do not allow lateral coordinates as inputs.
        if self.target_names and all(t.startswith('delta_') for t in self.target_names):
            self.feature_names = [
                c for c in raw_features
                if (c not in self.target_names) and ('_lateral' not in c)
            ]
        else:
            self.feature_names = [c for c in raw_features if c not in self.target_names]

        # Ensure production dir exists
        self.production_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Features: {len(self.feature_names)}, Targets: {len(self.target_names)}")
        
        return train_df, val_df, test_df

    def prepare_data(self, train_df, val_df, test_df):
        """Подготовить данные"""
        logger.info("Подготовка данных...")
        
        # Разделение на X и y
        X_train = train_df[self.feature_names]
        y_train = train_df[self.target_names]
        X_val = val_df[self.feature_names]
        y_val = val_df[self.target_names]
        X_test = test_df[self.feature_names]
        y_test = test_df[self.target_names]
        
        # Impute missing values then scale (keep feature count stable)
        X_train_imputed = self.imputer.fit_transform(X_train.values)
        X_val_imputed = self.imputer.transform(X_val.values)
        X_test_imputed = self.imputer.transform(X_test.values)

        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_val_scaled = self.scaler.transform(X_val_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        logger.info(f"X_train: {X_train_scaled.shape}, y_train: {y_train.shape}")
        
        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test

    def evaluate_on_test(self, model_kind: str, model_obj, X_test, y_test):
        """Оценить модель на test set (MAE/RMSE/MaxError/P95/SuccessRate)."""
        y_pred = model_obj.predict(X_test)
        if isinstance(y_pred, (list, tuple)):
            y_pred = np.asarray(y_pred)

        # y_test is DataFrame
        y_true = y_test.values
        abs_err = np.abs(y_true - y_pred)
        per_sample_mae = abs_err.mean(axis=1)

        mae = float(per_sample_mae.mean())
        rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
        max_error = float(abs_err.max())
        p95 = float(np.percentile(per_sample_mae, 95))
        success_rate_10mm = float((per_sample_mae < 10.0).mean())

        return {
            'model': model_kind,
            'test_mae_mm': mae,
            'test_rmse_mm': rmse,
            'test_max_error_mm': max_error,
            'test_p95_per_sample_mae_mm': p95,
            'test_success_rate_lt_10mm': success_rate_10mm,
        }

    def train_linear_regression(self, X_train, y_train, X_val, y_val):
        """Обучить Linear Regression"""
        logger.info("Обучение Linear Regression...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Предсказания
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Метрики
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        # MAE по осям
        val_mae_per_axis = {}
        for i, target in enumerate(self.target_names):
            val_mae_per_axis[target] = mean_absolute_error(y_val[target], val_pred[:, i])
        
        results = {
            'model': model,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'val_mae_per_axis': val_mae_per_axis
        }
        
        logger.info(f"  Train MAE: {train_mae:.3f} мм")
        logger.info(f"  Val MAE: {val_mae:.3f} мм")
        for target, mae in val_mae_per_axis.items():
            logger.info(f"    {target}: {mae:.3f} мм")
        
        return results

    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Обучить Random Forest"""
        logger.info("Обучение Random Forest...")
        
        # Базовые параметры
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        model = MultiOutputRegressor(rf)
        model.fit(X_train, y_train)
        
        # Предсказания
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Метрики
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        # MAE по осям
        val_mae_per_axis = {}
        for i, target in enumerate(self.target_names):
            val_mae_per_axis[target] = mean_absolute_error(y_val[target], val_pred[:, i])
        
        # Feature importance
        feature_importance = None
        if hasattr(model.estimators_[0], 'feature_importances_'):
            # Усредняем важность признаков по всем целям
            importances = []
            for estimator in model.estimators_:
                importances.append(estimator.feature_importances_)
            feature_importance = np.mean(importances, axis=0)
        
        results = {
            'model': model,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'val_mae_per_axis': val_mae_per_axis,
            'feature_importance': feature_importance
        }
        
        logger.info(f"  Train MAE: {train_mae:.3f} мм")
        logger.info(f"  Val MAE: {val_mae:.3f} мм")
        for target, mae in val_mae_per_axis.items():
            logger.info(f"    {target}: {mae:.3f} мм")
        
        return results

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Обучить XGBoost"""
        logger.info("Обучение XGBoost...")
        
        models = {}
        val_mae_per_axis = {}
        train_preds = {}
        val_preds = {}
        
        # Обучаем отдельную модель для каждой цели
        for i, target in enumerate(self.target_names):
            logger.info(f"  Обучение XGBoost для {target}...")
            
            # Консервативные параметры для маленького датасета
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror'
            )
            
            model.fit(X_train, y_train[target])
            
            # Предсказания
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Метрики
            train_mae = mean_absolute_error(y_train[target], train_pred)
            val_mae = mean_absolute_error(y_val[target], val_pred)
            
            models[target] = model
            val_mae_per_axis[target] = val_mae
            train_preds[target] = train_pred
            val_preds[target] = val_pred
            
            logger.info(f"    Train MAE: {train_mae:.3f} мм")
            logger.info(f"    Val MAE: {val_mae:.3f} мм")
        
        # Собираем предсказания
        train_pred_matrix = np.column_stack([train_preds[target] for target in self.target_names])
        val_pred_matrix = np.column_stack([val_preds[target] for target in self.target_names])
        
        # Общие метрики
        train_mae = mean_absolute_error(y_train, train_pred_matrix)
        val_mae = mean_absolute_error(y_val, val_pred_matrix)
        
        results = {
            'models': models,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'val_mae_per_axis': val_mae_per_axis
        }
        
        logger.info(f"  Общий Train MAE: {train_mae:.3f} мм")
        logger.info(f"  Общий Val MAE: {val_mae:.3f} мм")
        
        return results

    def create_ensemble(self, rf_results, xgb_results, X_val, y_val):
        """Создать ансамбль RF + XGBoost"""
        logger.info("Создание ансамбля...")
        
        # Предсказания RF
        rf_val_pred = rf_results['model'].predict(X_val)
        
        # Предсказания XGBoost
        xgb_val_pred = np.column_stack([
            xgb_results['models'][target].predict(X_val) 
            for target in self.target_names
        ])
        
        # Простое усреднение
        ensemble_pred = (rf_val_pred + xgb_val_pred) / 2
        
        # Метрики
        val_mae = mean_absolute_error(y_val, ensemble_pred)
        
        val_mae_per_axis = {}
        for i, target in enumerate(self.target_names):
            val_mae_per_axis[target] = mean_absolute_error(y_val[target], ensemble_pred[:, i])
        
        # Сравнение с отдельными моделями
        rf_vs_ensemble = ((rf_results['val_mae'] - val_mae) / rf_results['val_mae']) * 100
        xgb_vs_ensemble = ((xgb_results['val_mae'] - val_mae) / xgb_results['val_mae']) * 100
        
        results = {
            'val_mae': val_mae,
            'val_mae_per_axis': val_mae_per_axis,
            'improvement_vs_rf': rf_vs_ensemble,
            'improvement_vs_xgb': xgb_vs_ensemble
        }
        
        logger.info(f"  Ensemble Val MAE: {val_mae:.3f} мм")
        logger.info(f"  Улучшение vs RF: {rf_vs_ensemble:+.1f}%")
        logger.info(f"  Улучшение vs XGBoost: {xgb_vs_ensemble:+.1f}%")
        
        return results

    def compare_models(self, lr_results, rf_results, xgb_results, ensemble_results):
        """Сравнить все модели"""
        logger.info("СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
        logger.info("="*50)
        
        # Target-agnostic comparison (works with delta_* targets)
        comparison_data = [
            {
                'Model': 'Linear Regression',
                'Train MAE (mm)': lr_results['train_mae'],
                'Val MAE (mm)': lr_results['val_mae'],
            },
            {
                'Model': 'Random Forest',
                'Train MAE (mm)': rf_results['train_mae'],
                'Val MAE (mm)': rf_results['val_mae'],
            },
            {
                'Model': 'XGBoost',
                'Train MAE (mm)': xgb_results['train_mae'],
                'Val MAE (mm)': xgb_results['val_mae'],
            },
            {
                'Model': 'Ensemble (RF + XGB)',
                'Train MAE (mm)': np.nan,
                'Val MAE (mm)': ensemble_results['val_mae'],
            },
        ]
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Вывод таблицы
        logger.info("Таблица сравнения:")
        for _, row in comparison_df.iterrows():
            logger.info(f"  {row['Model']}: Val MAE = {row['Val MAE (mm)']:.3f} мм")
        
        # Находим лучшую модель
        best_model = comparison_df.loc[comparison_df['Val MAE (mm)'].idxmin()]
        logger.info(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model['Model']}")
        logger.info(f"   Val MAE: {best_model['Val MAE (mm)']:.3f} мм")
        
        return comparison_df, best_model['Model']

    def save_results(self, lr_results, rf_results, xgb_results, ensemble_results, comparison_df):
        """Сохранить результаты"""
        logger.info("Сохранение результатов...")
        
        # Сохраняем imputer + scaler (production)
        with open(self.production_dir / 'imputer.pkl', 'wb') as f:
            pickle.dump(self.imputer, f)

        with open(self.production_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Сохраняем списки признаков/таргетов (production)
        with open(self.production_dir / 'feature_names.json', 'w', encoding='utf-8') as f:
            json.dump(self.feature_names, f, ensure_ascii=False, indent=2)

        with open(self.production_dir / 'target_names.json', 'w', encoding='utf-8') as f:
            json.dump(self.target_names, f, ensure_ascii=False, indent=2)
        
        # Сохраняем модели (production)
        with open(self.production_dir / 'linear_regression.pkl', 'wb') as f:
            pickle.dump(lr_results['model'], f)

        with open(self.production_dir / 'random_forest.pkl', 'wb') as f:
            pickle.dump(rf_results['model'], f)
        
        # XGBoost модели (production)
        for target, model in xgb_results['models'].items():
            with open(self.production_dir / f'xgboost_{target}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Feature importance для Random Forest (keep in models/)
        if rf_results['feature_importance'] is not None:
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_results['feature_importance']
            }).sort_values('importance', ascending=False)
            
            feature_importance_df.to_csv('models/feature_importance.csv', index=False)
        
        # Сохраняем таблицу сравнения
        comparison_df.to_csv('models/model_comparison.csv', index=False)
        
        # Сохраняем детальные результаты
        all_results = {
            'linear_regression': {
                'train_mae': lr_results['train_mae'],
                'val_mae': lr_results['val_mae'],
                'val_mae_per_axis': lr_results['val_mae_per_axis']
            },
            'random_forest': {
                'train_mae': rf_results['train_mae'],
                'val_mae': rf_results['val_mae'],
                'val_mae_per_axis': rf_results['val_mae_per_axis']
            },
            'xgboost': {
                'train_mae': xgb_results['train_mae'],
                'val_mae': xgb_results['val_mae'],
                'val_mae_per_axis': xgb_results['val_mae_per_axis']
            },
            'ensemble': {
                'val_mae': ensemble_results['val_mae'],
                'val_mae_per_axis': ensemble_results['val_mae_per_axis'],
                'improvement_vs_rf': ensemble_results['improvement_vs_rf'],
                'improvement_vs_xgb': ensemble_results['improvement_vs_xgb']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/all_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Production metadata
        prod_meta = {
            'timestamp': datetime.now().isoformat(),
            'feature_count': len(self.feature_names),
            'target_count': len(self.target_names),
            'val_metrics': {
                'linear_regression': {'val_mae': lr_results['val_mae']},
                'random_forest': {'val_mae': rf_results['val_mae']},
                'xgboost': {'val_mae': xgb_results['val_mae']},
                'ensemble': {'val_mae': ensemble_results['val_mae'], 'weight_rf': 0.5, 'weight_xgb': 0.5},
            },
        }

        with open(self.production_dir / 'model_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(prod_meta, f, indent=2, ensure_ascii=False)
         
        logger.info("✅ Все результаты сохранены")

    def run(self):
        """Запустить обучение всех моделей"""
        logger.info("НАЧАЛО ОБУЧЕНИЯ ВСЕХ МОДЕЛЕЙ")
        logger.info("="*50)
        
        try:
            # 1. Загрузка данных
            train_df, val_df, test_df = self.load_data()
            
            # 2. Подготовка данных
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(train_df, val_df, test_df)
            
            # 3. Обучение моделей
            lr_results = self.train_linear_regression(X_train, y_train, X_val, y_val)
            rf_results = self.train_random_forest(X_train, y_train, X_val, y_val)
            xgb_results = self.train_xgboost(X_train, y_train, X_val, y_val)
            
            # 4. Создание ансамбля
            ensemble_results = self.create_ensemble(rf_results, xgb_results, X_val, y_val)
            
            # 5. Сравнение моделей
            comparison_df, best_model = self.compare_models(lr_results, rf_results, xgb_results, ensemble_results)
            
            # 6. Сохранение результатов
            self.save_results(lr_results, rf_results, xgb_results, ensemble_results, comparison_df)
            
            # Итоговая сводка
            logger.info("="*50)
            logger.info("ИТОГИ ОБУЧЕНИЯ")
            logger.info("="*50)
            logger.info(f"🏆 Лучшая модель: {best_model}")
            logger.info(f"📊 Всего обучено моделей: 4")
            logger.info(f"📁 Результаты сохранены в models/")
            logger.info("="*50)
            logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            raise

def main():
    """Главная функция"""
    trainer = ModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main()
