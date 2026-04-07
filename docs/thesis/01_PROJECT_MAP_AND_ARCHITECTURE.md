# 01. Карта проекта и архитектура (Project Map & Architecture)

## 1. Что делает проект (формулировка задачи)
Проект реализует ML-систему, которая по набору **антропометрических и анатомических признаков** пациента (в основном получаемых из DICOM/табличных источников) предсказывает **смещение почек** при изменении положения пациента.

В текущей production-ветке (Phase 1 интегрированного ансамбля) предсказываются 6 целевых переменных:
- `kidney_left_delta_x`, `kidney_left_delta_y`, `kidney_left_delta_z`
- `kidney_right_delta_x`, `kidney_right_delta_y`, `kidney_right_delta_z`

Единицы измерения: **миллиметры (мм)**.

Практическое назначение: поддержка предоперационного планирования и/или AR-навигации, где важно заранее оценить ожидаемое смещение органов.

## 2. Высокоуровневая архитектура
Архитектура проекта многоуровневая:

- **Данные**
  - `data/processed/*.csv` — подготовленные датасеты (train/validation/test)
  - `data/*` — дополнительные интегрированные/сырьевые наборы
- **Модели и обучение (табличные модели)**
  - `models/phase1/adaptive_ensemble.py` — интегрированный оптимизированный ансамбль + feature engineering
  - `scripts/training/train_all_models.py` — обучение и сравнение базовых моделей (LR/RF/XGB + простое усреднение)
  - `src/models/*` — набор тренировочных/диагностических скриптов и альтернативных моделей
- **Инференс / API**
  - `src/api/kidney_displacement_api.py` — FastAPI сервис, загружающий `models/adaptive_ensemble.pkl` и выполняющий предсказание
- **Документация и отчёты**
  - `docs/*` и `docs/archive/*` — техническая документация, отчёты фаз
  - `results/*` — численные результаты экспериментов

## 3. Основные входные точки (entry points)

### 3.1. Обучение ансамбля Phase 1 (интегрированная версия)
Файл:
- `models/phase1/adaptive_ensemble.py`

Роль:
- читает `data/processed/train.csv` + `data/processed/validation.csv`;
- строит расширенный набор признаков (base + engineered + cross);
- обучает базовые модели (RF/Lasso/Ridge/GradientBoosting);
- оптимизирует веса ансамбля по MAE (scipy minimize, L-BFGS-B);
- сохраняет модели/скейлер/список признаков в `models/adaptive_ensemble.pkl`.

### 3.2. Обучение “набора моделей” и сравнение (baseline pipeline)
Файл:
- `scripts/training/train_all_models.py`

Роль:
- грузит `data/processed/train.csv`, `validation.csv`, `test.csv`;
- использует `data/processed/feature_names.json` и `data/processed/target_names.json`;
- прогоняет LR / RF (MultiOutputRegressor) / XGBoost (по цели) / простой ансамбль (RF+XGB 0.5/0.5);
- сохраняет production-артефакты (imputer, scaler, feature_names, target_names, модели) в `models/production/`.

### 3.3. API инференса
Файл:
- `src/api/kidney_displacement_api.py`

Роль:
- загружает `models/adaptive_ensemble.pkl`;
- создаёт признаки точно так же, как в `AdaptiveEnsembleTrainer` (внутренние методы `_create_engineered_features`, `_create_cross_features`);
- применяет сохранённый `scaler`;
- выполняет предсказание по каждой цели;
- отдаёт JSON ответы.

## 4. Ключевые директории и их ответственность

### 4.1. `models/`
Содержит обученные артефакты и “референсные” файлы:
- `models/adaptive_ensemble.pkl` — production модель (joblib), используемая FastAPI.
- `models/phase1/*.py` — исходный код ансамбля Phase 1.
- `models/*_feature_names.json`, `*_target_names.json`, `*_feature_importance.*` — сериализованные метаданные.

### 4.2. `src/`
Содержит прикладную логику, которая “оборачивает” ML:
- `src/api/` — сервис предсказания.
- `src/features/` — альтернативные реализации feature engineering.
- `src/models/` — различные тренировки/фиксы/тесты моделей и данных.
- `src/data_validation.py`, `src/constraints.py`, `src/confidence_scoring.py` и др. — инфраструктурные компоненты (валидация, ограничения, оценка надежности).

### 4.3. `results/`
- `results/final_model_comparison_report.md` — итоговое сравнение моделей.
- `results/phase2/*.csv` — результаты Phase 2 экспериментов.
- `results/phase3/*` — исследовательские скрипты и их отчёты.

### 4.4. `docs/`
Содержит большую часть проектной документации.
В диссертации это важно использовать как “внутренние отчёты разработки”, но при необходимости переписать академическим стилем.

## 5. Концептуальный dataflow (train vs inference)

### 5.1. Train
1) чтение CSV
2) проверка наличия признаков
3) feature engineering
4) масштабирование (StandardScaler fit на train)
5) обучение базовых моделей
6) оптимизация весов ансамбля
7) сохранение артефактов

### 5.2. Inference
1) получение JSON
2) построение DataFrame из входных признаков
3) feature engineering (тот же набор формул)
4) масштабирование (transform сохранённым scaler)
5) предсказание каждой цели
6) упаковка ответа + метаданные

## 6. Выводы для диссертации
- Проект фактически имеет **две параллельные ML-линии**:
  - (A) интегрированный ансамбль (Phase 1) с малым количеством базовых признаков + engineered/cross.
  - (B) baseline “train_all_models” на feature_names/target_names (более общий механизм для множества таргетов).
- Для диссертации важно явно зафиксировать:
  - какая линия считается production в текущей поставке (`models/adaptive_ensemble.pkl` + FastAPI);
  - какие линии являются исследовательскими/архивными.
