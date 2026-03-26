# 📁 Kidney Displacement Predictor - Текущая Структура Проекта

## 🎯 **Финальная организация файлов после чистки**

---

## 📂 **Структура папок**

```
d:\ml trainer\
├── 📁 data\                              # Данные проекта
├── 📁 models\phase1\                    # Модели Phase 1
├── 📁 enhanced_models\phase2\           # Модели Phase 2
├── 📁 results\phase1\                   # Результаты Phase 1
├── 📁 results\phase2\                   # Результаты Phase 2
├── 📁 results\phase3\                   # Результаты Phase 3
├── 📁 docs\archive\                     # Архивная документация
├── 📁 scripts\archive\                  # Архивные скрипты
├── 📁 tests\archive\                    # Архивные тесты
├── 📁 logs\                             # Логи API
├── 📁 .windsurf\                        # Конфигурация Windsurf
├── 📁 [другие системные папки]           # backend, src, notebooks, kits19
└── 📄 [основные файлы]                  # README, requirements, etc.
```

---

## 📁 **Папка `models\phase1\` (Phase 1: Production-ready)**

```
models\phase1\
├── 📄 kidney_displacement_predictor.py    # Основной предсказатель Phase 1
├── 📄 api_kidney_predictor.py             # REST API Phase 1
├── 📄 test_kidney_predictor.py            # Тесты Phase 1
├── 📄 ensemble_models.py                  # Базовые ансамбли
├── 📄 adaptive_ensemble.py                # Адаптивный ансамбль
├── 📄 target_specific_ensemble.py         # Специфичные ансамбли
├── 📄 error_correction_ensemble.py        # Ансамбль коррекции ошибок
├── 📄 compare_all_ensembles.py           # Сравнение ансамблей
├── 📄 train_lasso.py                      # Тренировка Lasso
└── 📄 train_ridge.py                      # Тренировка Ridge
```

---

## 📁 **Папка `enhanced_models\phase2\` (Phase 2: Enhanced Production)**

```
enhanced_models\phase2\
├── 📄 enhanced_kidney_displacement_predictor.py  # Улучшенный предсказатель
├── 📄 enhanced_api_kidney_predictor.py             # Улучшенный API
├── 📄 test_enhanced_predictor.py                   # Тесты Phase 2
├── 📄 dynamic_adaptive_ensemble.py                 # Динамический ансамбль
├── 📄 enhanced_feature_engineering.py              # Инженерия признаков
├── 📄 multivariate_displacement_predictor.py       # Многомерный предсказатель
└── 📄 compare_phase2_results.py                    # Сравнение Phase 2
```

---

## 📁 **Папка `results\phase3\` (Phase 3: Research Approaches)**

```
results\phase3\
├── 📄 neural_network_ensemble.py              # Нейронный ансамбль
├── 📄 multitask_learning_predictor.py         # Многозадачное обучение
├── 📄 uncertainty_quantification_predictor.py # Квантификация неопределенности
├── 📄 compare_phase3_results.py              # Сравнение Phase 3
├── 📄 run_phase3_research.py                 # Пайплайн Phase 3
├── 📄 neural_network_ensemble_results.csv    # Результаты нейронного ансамбля
├── 📄 multitask_learning_results.csv         # Результаты многозадачного обучения
├── 📄 phase3_research_summary.csv             # Сводка Phase 3
├── 📄 neural_network_ensemble.pth             # Модель нейронного ансамбля
└── 📄 multitask_learning_model.pth            # Модель многозадачного обучения
```

---

## 📁 **Папка `results\phase2\`**

```
results\phase2\
├── 📄 dynamic_adaptive_ensemble_results.csv  # Результаты динамического ансамбля
├── 📄 multivariate_displacement_results.csv  # Результаты многомерного предсказания
└── 📄 phase2_enhancement_summary.csv          # Сводка Phase 2
```

---

## 📁 **Папка `docs\archive\`**

```
docs\archive\
├── 📄 README_PRODUCTION.md                   # Документация Phase 1
├── 📄 README_ENHANCED_PRODUCTION.md          # Документация Phase 2
├── 📄 README_PHASE3_RESEARCH.md              # Документация Phase 3
├── 📄 PHASE3_FINAL_CONCLUSIONS.md            # Выводы Phase 3
├── 📄 feature_analysis_report.md             # Отчет по анализу признаков
├── 📄 README_FINAL.md                        # Финальный README
├── 📄 comment.md                             # Комментарии
└── 📄 ТЕХНИЧЕСКИЙ_ОТЧЁТ_ПО_ПРОЕКТУ.md        # Технический отчет
```

---

## 📁 **Папка `scripts\archive\`**

```
scripts\archive\
├── 📄 train_all_models_fixed.py              # Все модели (исправленный)
├── 📄 train_all_models_simple.py             # Все модели (простой)
├── 📄 train_elasticnet_optimized.py          # ElasticNet (оптимизированный)
├── 📄 train_gradient_boosting.py             # Gradient Boosting
├── 📄 train_gradient_boosting_optimized.py    # Gradient Boosting (оптимизированный)
├── 📄 train_kidney_displacement_model.py     # Основная модель
├── 📄 train_linear_regression.py             # Linear Regression
├── 📄 train_random_forest.py                 # Random Forest
├── 📄 train_random_forest_optimized.py       # Random Forest (оптимизированный)
├── 📄 check_dataset.py                       # Проверка датасета
├── 📄 compare_models.py                      # Сравнение моделей
├── 📄 debug_nan_issues.py                    # Отладка NaN
├── 📄 detailed_comparison.py                 # Детальное сравнение
├── 📄 extract_dicom_features.py              # Извлечение DICOM признаков
├── 📄 extract_medical_grade_features.py      # Извлечение медицинских признаков
├── 📄 final_check.py                         # Финальная проверка
├── 📄 outlier_analysis.py                    # Анализ выбросов
├── 📄 transform_vybor_data.py                # Трансформация данных Vybor
├── 📄 dicoms_out.csv                         # Выходные DICOM данные
└── 📄 Выборка - 50.csv                       # Выборка данных
```

---

## 📄 **Основные файлы в корне**

```
d:\ml trainer\
├── 📄 README.md                              # Основное описание проекта
├── 📄 PROJECT_PHASES_SUMMARY.md              # Сводка всех этапов
├── 📄 PROJECT_STRUCTURE_MANUAL.md           # Руководство по структуре
├── 📄 README_CURRENT_STRUCTURE.md           # Этот файл
├── 📄 requirements.txt                       # Зависимости Phase 1
├── 📄 requirements_enhanced.txt              # Зависимости Phase 2
├── 📄 requirements_phase3.txt                # Зависимости Phase 3
├── 📄 # Исполнительное резюме.ini             # Системный файл
└── 📁 [системные папки]                     # data, logs, .windsurf, etc.
```

---

## 🎯 **Что осталось в корне (используемые файлы):**

### 📄 **Основная документация:**
- `README.md` - Основное описание проекта
- `PROJECT_PHASES_SUMMARY.md` - Обзор всех этапов
- `PROJECT_STRUCTURE_MANUAL.md` - Руководство по восстановлению
- `README_CURRENT_STRUCTURE.md` - Текущая структура

### 📄 **Зависимости:**
- `requirements.txt` - Phase 1 зависимости
- `requirements_enhanced.txt` - Phase 2 зависимости  
- `requirements_phase3.txt` - Phase 3 зависимости

### 📁 **Системные папки:**
- `data\` - Датасеты (vybor_unified_features.csv, kits19_medical_grade_features.csv)
- `logs\` - Логи API серверов
- `.windsurf\` - Конфигурация Windsurf
- `backend\`, `src\`, `notebooks\`, `kits19\` - Другие компоненты проекта

---

## 🚀 **Как использовать структуру:**

### 🏥 **Для клинического использования (Phase 2):**
```bash
# Запустить улучшенный предсказатель
python enhanced_models\phase2\enhanced_kidney_displacement_predictor.py

# Запустить улучшенный API
python enhanced_models\phase2\enhanced_api_kidney_predictor.py

# Запустить тесты
python enhanced_models\phase2\test_enhanced_predictor.py
```

### 🔬 **Для исследований (Phase 3):**
```bash
# Запустить полный пайплайн Phase 3
python results\phase3\run_phase3_research.py

# Запустить отдельные компоненты
python results\phase3\neural_network_ensemble.py
python results\phase3\multitask_learning_predictor.py
python results\phase3\uncertainty_quantification_predictor.py
```

### 🏗️ **Для базовой системы (Phase 1):**
```bash
# Запустить базовый предсказатель
python models\phase1\kidney_displacement_predictor.py

# Запустить базовый API
python models\phase1\api_kidney_predictor.py

# Запустить тесты
python models\phase1\test_kidney_predictor.py
```

---

## 📊 **Преимущества новой структуры:**

### ✅ **Организация по фазам:**
- **Phase 1** в `models\phase1\` - базовая система
- **Phase 2** в `enhanced_models\phase2\` - улучшенная система  
- **Phase 3** в `results\phase3\` - исследования

### ✅ **Чистота корневой директории:**
- Только основные файлы и документация
- Все рабочие скрипты организованы по папкам
- Архивные материалы в `archive` папках

### ✅ **Легкость навигации:**
- Понятная структура по этапам разработки
- Логическое разделение кода и результатов
- Удобный доступ к нужным компонентам

---

## 🎯 **Итог:**

**Проект теперь имеет чистую, логичную структуру с разделением по фазам разработки. Все файлы организованы, архивированы или удалены согласно их назначению и использованию.**

**🏆 Рекомендуемое использование: Phase 2 (enhanced_models\phase2\) для клинического развертывания.**
