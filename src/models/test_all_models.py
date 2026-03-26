#!/usr/bin/env python3
"""
КОМПЛЕКСНЫЙ ТЕСТ ВСЕХ МОДЕЛЕЙ с интеграцией множественных источников данных
Тестирование RandomForest и XGBoost с данными из:
- scripts/archive/dicoms_out.csv
- data/vybor_unified_features.csv  
- data/kits19_medical_grade_features.csv
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def check_data_sources(self):
        """Проверить наличие всех источников данных"""
        logger.info("=" * 60)
        logger.info("ПРОВЕРКА ИСТОЧНИКОВ ДАННЫХ")
        logger.info("=" * 60)
        
        sources = {
            'DICOMS': '../scripts/archive/dicoms_out.csv',
            'Vybor': '../data/vybor_unified_features.csv',
            'KiTS19': '../data/kits19_medical_grade_features.csv',
            'Main Train': '../data/processed/train.csv',
            'Main Val': '../data/processed/validation.csv'
        }
        
        available_sources = {}
        
        for name, path in sources.items():
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    available_sources[name] = {
                        'path': path,
                        'rows': len(df),
                        'cols': len(df.columns),
                        'size_mb': os.path.getsize(path) / (1024*1024)
                    }
                    logger.info(f"✅ {name}: {len(df)} строк, {len(df.columns)} колонок, {available_sources[name]['size_mb']:.2f} MB")
                except Exception as e:
                    logger.error(f"❌ {name}: ошибка чтения - {e}")
            else:
                logger.warning(f"⚠️  {name}: файл не найден - {path}")
        
        return available_sources
    
    def test_random_forest(self):
        """Тестирование RandomForest модели"""
        logger.info("=" * 60)
        logger.info("ТЕСТИРОВАНИЕ RANDOM FOREST")
        logger.info("=" * 60)
        
        try:
            # Импорт и запуск RandomForest
            from train_random_forest import RandomForestTrainer
            
            logger.info("🔄 Создание RandomForest тренера...")
            trainer = RandomForestTrainer(use_dicoms_data=True)
            
            logger.info("🔄 Загрузка данных...")
            start_time = time.time()
            train_df, val_df = trainer.load_data_and_scaler()
            load_time = time.time() - start_time
            
            logger.info("🔄 Подготовка данных...")
            start_time = time.time()
            X_train, y_train, X_val, y_val = trainer.prepare_data(train_df, val_df)
            prep_time = time.time() - start_time
            
            logger.info("🔄 Создание модели...")
            trainer.create_model()
            
            logger.info("🔄 Обучение модели...")
            start_time = time.time()
            trainer.train_model(X_train, y_train)
            train_time = time.time() - start_time
            
            logger.info("🔄 Предсказания...")
            start_time = time.time()
            train_pred, val_pred = trainer.make_predictions(X_train, X_val)
            pred_time = time.time() - start_time
            
            logger.info("🔄 Расчет метрик...")
            results = trainer.calculate_metrics(y_train, train_pred, y_val, val_pred)
            
            # Сохранение результатов
            rf_results = {
                'model': 'RandomForest',
                'success': True,
                'load_time': load_time,
                'prep_time': prep_time,
                'train_time': train_time,
                'pred_time': pred_time,
                'total_time': load_time + prep_time + train_time + pred_time,
                'train_mae': results['train_mae'],
                'val_mae': results['val_mae'],
                'n_features': results['n_features'],
                'n_targets': results['n_targets'],
                'train_samples': results['n_train_samples'],
                'val_samples': results['n_val_samples'],
                'use_dicoms_data': results.get('use_dicoms_data', False)
            }
            
            logger.info("✅ RandomForest тест завершен успешно!")
            logger.info(f"   Train MAE: {results['train_mae']:.3f} мм")
            logger.info(f"   Val MAE: {results['val_mae']:.3f} мм")
            logger.info(f"   Время обучения: {train_time:.2f} сек")
            logger.info(f"   Всего времени: {rf_results['total_time']:.2f} сек")
            
            return rf_results
            
        except Exception as e:
            logger.error(f"❌ RandomForest тест провален: {e}")
            return {
                'model': 'RandomForest',
                'success': False,
                'error': str(e)
            }
    
    def test_xgboost(self):
        """Тестирование XGBoost модели"""
        logger.info("=" * 60)
        logger.info("ТЕСТИРОВАНИЕ XGBOOST")
        logger.info("=" * 60)
        
        try:
            # Импорт и запуск XGBoost
            from train_xgboost import XGBoostTrainer
            
            logger.info("🔄 Создание XGBoost тренера...")
            trainer = XGBoostTrainer(use_dicoms_data=True)
            
            logger.info("🔄 Загрузка данных...")
            start_time = time.time()
            train_df, val_df = trainer.load_data_and_scaler()
            load_time = time.time() - start_time
            
            logger.info("🔄 Подготовка данных...")
            start_time = time.time()
            X_train, y_train, X_val, y_val = trainer.prepare_data(train_df, val_df)
            prep_time = time.time() - start_time
            
            logger.info("🔄 Обучение моделей...")
            start_time = time.time()
            trainer.train_models(X_train, y_train, X_val, y_val)
            train_time = time.time() - start_time
            
            logger.info("🔄 Предсказания...")
            start_time = time.time()
            train_pred, val_pred = trainer.make_predictions(X_train, X_val)
            pred_time = time.time() - start_time
            
            logger.info("🔄 Расчет метрик...")
            results = trainer.calculate_metrics(y_train, train_pred, y_val, val_pred)
            
            # Сохранение результатов
            xgb_results = {
                'model': 'XGBoost',
                'success': True,
                'load_time': load_time,
                'prep_time': prep_time,
                'train_time': train_time,
                'pred_time': pred_time,
                'total_time': load_time + prep_time + train_time + pred_time,
                'train_mae': results['train_mae'],
                'val_mae': results['val_mae'],
                'n_features': results['n_features'],
                'n_targets': results['n_targets'],
                'train_samples': results['n_train_samples'],
                'val_samples': results['n_val_samples'],
                'use_dicoms_data': results.get('use_dicoms_data', False)
            }
            
            logger.info("✅ XGBoost тест завершен успешно!")
            logger.info(f"   Train MAE: {results['train_mae']:.3f} мм")
            logger.info(f"   Val MAE: {results['val_mae']:.3f} мм")
            logger.info(f"   Время обучения: {train_time:.2f} сек")
            logger.info(f"   Всего времени: {xgb_results['total_time']:.2f} сек")
            
            return xgb_results
            
        except Exception as e:
            logger.error(f"❌ XGBoost тест провален: {e}")
            return {
                'model': 'XGBoost',
                'success': False,
                'error': str(e)
            }
    
    def test_data_integration(self):
        """Тестирование интеграции данных"""
        logger.info("=" * 60)
        logger.info("ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ ДАННЫХ")
        logger.info("=" * 60)
        
        try:
            from train_random_forest import RandomForestTrainer
            
            trainer = RandomForestTrainer(use_dicoms_data=True)
            train_df, val_df = trainer.load_data_and_scaler()
            
            # Анализ интеграции
            integration_results = {
                'success': True,
                'total_features': len(trainer.feature_names),
                'total_targets': len(trainer.target_names),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'feature_sources': {},
                'target_sources': {}
            }
            
            # Анализ источников признаков
            for feature in trainer.feature_names:
                if 'kits19_' in feature:
                    integration_results['feature_sources']['KiTS19'] = integration_results['feature_sources'].get('KiTS19', 0) + 1
                elif 'dicoms_' in feature:
                    integration_results['feature_sources']['DICOMS'] = integration_results['feature_sources'].get('DICOMS', 0) + 1
                else:
                    integration_results['feature_sources']['Vybor'] = integration_results['feature_sources'].get('Vybor', 0) + 1
            
            # Анализ источников целей
            for target in trainer.target_names:
                if 'kits19_' in target:
                    integration_results['target_sources']['KiTS19'] = integration_results['target_sources'].get('KiTS19', 0) + 1
                else:
                    integration_results['target_sources']['Vybor'] = integration_results['target_sources'].get('Vybor', 0) + 1
            
            logger.info("✅ Интеграция данных успешна!")
            logger.info(f"   Всего признаков: {integration_results['total_features']}")
            logger.info(f"   Всего целей: {integration_results['total_targets']}")
            logger.info(f"   Источники признаков: {integration_results['feature_sources']}")
            logger.info(f"   Источники целей: {integration_results['target_sources']}")
            
            return integration_results
            
        except Exception as e:
            logger.error(f"❌ Тест интеграции данных провален: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_report(self, sources, rf_results, xgb_results, integration_results):
        """Генерация финального отчета"""
        logger.info("=" * 60)
        logger.info("ФИНАЛЬНЫЙ ОТЧЕТ")
        logger.info("=" * 60)
        
        total_time = time.time() - self.start_time
        
        # Создание отчета
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'total_test_time': total_time,
            'data_sources': sources,
            'integration_test': integration_results,
            'random_forest': rf_results,
            'xgboost': xgb_results,
            'summary': {
                'models_tested': 0,
                'successful_tests': 0,
                'failed_tests': 0,
                'best_val_mae': None,
                'best_model': None
            }
        }
        
        # Подсчет статистики
        models = [rf_results, xgb_results]
        for model_result in models:
            if model_result['success']:
                report['summary']['successful_tests'] += 1
                report['summary']['models_tested'] += 1
                
                # Поиск лучшей модели
                if report['summary']['best_val_mae'] is None or model_result['val_mae'] < report['summary']['best_val_mae']:
                    report['summary']['best_val_mae'] = model_result['val_mae']
                    report['summary']['best_model'] = model_result['model']
            else:
                report['summary']['failed_tests'] += 1
                report['summary']['models_tested'] += 1
        
        # Вывод отчета
        logger.info(f"📊 ОБЩАЯ СТАТИСТИКА:")
        logger.info(f"   Время тестирования: {total_time:.2f} сек")
        logger.info(f"   Протестировано моделей: {report['summary']['models_tested']}")
        logger.info(f"   Успешных тестов: {report['summary']['successful_tests']}")
        logger.info(f"   Проваленных тестов: {report['summary']['failed_tests']}")
        
        if report['summary']['best_model']:
            logger.info(f"   Лучшая модель: {report['summary']['best_model']}")
            logger.info(f"   Лучший Val MAE: {report['summary']['best_val_mae']:.3f} мм")
        
        # Сравнение моделей
        if rf_results['success'] and xgb_results['success']:
            logger.info(f"\n🔄 СРАВНЕНИЕ МОДЕЛЕЙ:")
            logger.info(f"   RandomForest: Train={rf_results['train_mae']:.3f}, Val={rf_results['val_mae']:.3f} мм")
            logger.info(f"   XGBoost:      Train={xgb_results['train_mae']:.3f}, Val={xgb_results['val_mae']:.3f} мм")
            
            rf_improvement = ((rf_results['val_mae'] - xgb_results['val_mae']) / rf_results['val_mae']) * 100
            if rf_improvement > 0:
                logger.info(f"   XGBoost лучше RandomForest на {rf_improvement:.1f}%")
            else:
                logger.info(f"   RandomForest лучше XGBoost на {abs(rf_improvement):.1f}%")
        
        # Сохранение отчета
        report_file = f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📄 Отчет сохранен: {report_file}")
        
        return report
    
    def run_all_tests(self):
        """Запустить все тесты"""
        logger.info("🚀 ЗАПУСК КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ")
        logger.info("=" * 60)
        
        # 1. Проверка источников данных
        sources = self.check_data_sources()
        
        # 2. Тест интеграции данных
        integration_results = self.test_data_integration()
        
        # 3. Тест RandomForest
        rf_results = self.test_random_forest()
        
        # 4. Тест XGBoost
        xgb_results = self.test_xgboost()
        
        # 5. Генерация отчета
        report = self.generate_report(sources, rf_results, xgb_results, integration_results)
        
        return report

def main():
    """Главная функция"""
    tester = ModelTester()
    report = tester.run_all_tests()
    
    # Финальное сообщение
    if report['summary']['successful_tests'] == report['summary']['models_tested']:
        logger.info("\n🎉 ВСЕ ТЕСТЫ УСПЕШНО ЗАВЕРШЕНЫ!")
    else:
        logger.info(f"\n⚠️  {report['summary']['failed_tests']} из {report['summary']['models_tested']} тестов провалены")
    
    return report

if __name__ == "__main__":
    main()
