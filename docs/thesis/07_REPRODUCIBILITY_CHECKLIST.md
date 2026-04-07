# 07. Чеклист воспроизводимости (Reproducibility Checklist)

## 1. Цель
Этот документ нужен для диссертации и защиты: чтобы любой проверяющий мог понять, как воспроизвести результаты и что именно считается «результатом».

## 2. Что считается воспроизводимым результатом в текущей поставке

### 2.1. Production inference через FastAPI
Воспроизводимый результат:
- запуск `src/api/kidney_displacement_api.py`
- успешный ответ `POST /predict` при наличии модели `models/adaptive_ensemble.pkl`

Критические зависимости:
- `models/adaptive_ensemble.pkl` должен соответствовать коду feature engineering.

### 2.2. Обучение Phase 1 integrated ensemble
Воспроизводимый результат:
- запуск `models/phase1/adaptive_ensemble.py`
- обучение на `data/processed/train.csv` + `data/processed/validation.csv`
- генерация/обновление артефакта `models/adaptive_ensemble.pkl`

### 2.3. Baseline training (train_all_models)
Воспроизводимый результат:
- запуск `scripts/training/train_all_models.py`
- наличие файлов:
  - `data/processed/train.csv`, `validation.csv`, `test.csv`
  - `data/processed/feature_names.json`, `data/processed/target_names.json`
- генерация артефактов в `models/production/`

## 3. Минимальный набор файлов

### 3.1. Для inference
- `models/adaptive_ensemble.pkl`
- `src/api/kidney_displacement_api.py`
- `models/phase1/adaptive_ensemble.py` (как зависимость для генерации фич)

### 3.2. Для обучения Phase 1
- `models/phase1/adaptive_ensemble.py`
- `data/processed/train.csv`
- `data/processed/validation.csv`

## 4. Контроль версий

### 4.1. Git
В репозитории 2 коммита, поэтому для диссертации важно фиксировать:
- хэш коммита;
- дату;
- соответствие артефактов моделей и кода.

### 4.2. Версионирование моделей
Рекомендуется для диссертации ввести соглашение (даже если не реализовано в коде):
- версия модели = (git commit hash) + (timestamp обучения) + (набор данных)

## 5. Детерминизм и случайность

### 5.1. Random seeds
В коде часто используется `random_state=42`.
Нужно фиксировать:
- random_state в split
- random_state в моделях

### 5.2. Параллелизм
Некоторые модели используют `n_jobs=-1`.
Для строгой воспроизводимости лучше:
- фиксировать число потоков;
- либо принимать минимальные расхождения из-за параллелизма.

## 6. Проверки качества данных
Перед обучением:
- проверить наличие всех required features
- проверить, что `body_width_mm > 0`, `body_depth_mm > 0`
- проверить, что таргеты (delta) в разумном диапазоне

## 7. Контроль утечек
В baseline pipeline есть защита от leakage:
- исключаются признаки `_lateral`, когда таргеты — `delta_*`

Это правило нужно явно описать в диссертации как обязательное.

## 8. Стандартизация единиц измерения
Необходимо:
- зафиксировать, какие поля в мм, какие в см³
- описать, почему деление/умножение в engineered признаках корректно по единицам

## 9. Воспроизводимость Phase 2 и Phase 3

### 9.1. Phase 2
- исходники отсутствуют
- доступны CSV и архивная документация

### 9.2. Phase 3
- исходники присутствуют (`results/phase3/*.py`)
- требуются данные (см. проверки в `run_phase3_research.py`)
- возможны ошибки окружения/зависимостей

## 10. Что приложить в диссертацию как приложения
- `docs/thesis/04_FEATURES_AND_PARAMETERS_CATALOG.md` (полный справочник)
- таблицу экспериментов (фаза → датасет → признаки → targets → метрика)
- пример JSON запроса/ответа API
- схему dataflow
