# Команды полного прогона Phase 1

Шпаргалка для запуска на **чужом компьютере**: DICOM → CSV → интеграция → обучение → валидация → API.

Рабочая директория — **корень репозитория** (`ml trainer/`).

---

## 0. Подготовка окружения

### Windows (PowerShell)

```powershell
cd "D:\path\to\ml trainer"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt

# Для DICOM-извлечения (обязательно):
py -3 -m pip install pydicom scikit-image nibabel

# Опционально: нейросегментация почек (медленно, ~4+ ГБ RAM)
# py -3 -m pip install totalsegmentator
```

### Linux / WSL

```bash
cd /path/to/ml-trainer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pydicom scikit-image nibabel
```

### Проверка

```powershell
py -3 scripts/run_phase1_pipeline.py info
py -3 -m pytest tests/test_feature_schema.py tests/test_ct_geometry.py -q
```

---

## 1. Структура DICOM на диске

Ожидается **корневая папка**, внутри — **подпапки по пациентам/исследованиям**:

```text
D:\DICOM СНИМКИ\
  Пациент_001\
    slice_001.dcm
    slice_002.dcm
    ...
  Пациент_002\
    ...
```

В каждой подпапке — CT-срезы (`.dcm` / без расширения с DICOM-заголовком).

---

## 2. Быстрый полный цикл (копипаст)

Замените `D:\DICOM СНИМКИ` на свой путь.

### Windows

```powershell
# A. Извлечь признаки из всех DICOM (рекомендуется batch-скрипт)
py -3 -u scripts/inference/run_dicom_batch.py "D:\DICOM СНИМКИ" `
  --output results/dicom_batch_extract.csv `
  --accuracy-mode fast

# B. Скопировать результат в канонический вход интегратора
# (вручную: оставить строки status=extracted, положить в data/dicom_medical_features.csv)
# Либо заменить файл целиком, если других DICOM-строк нет.

# C. KiTS reference (опционально, если есть kits19 CSV)
py -3 scripts/features/build_feature_reference.py

# D. Интеграция -> train/validation CSV
py -3 scripts/run_phase1_pipeline.py integrate

# E. Обучение
py -3 scripts/run_phase1_pipeline.py train

# F. Валидация (честный holdout на клинических данных)
py -3 scripts/run_phase1_pipeline.py validate `
  --run-id run_YYYYMMDD `
  --dataset data/processed/validation_clinical.csv `
  --holdout

# G. (Опционально) визуальные тесты
py -3 scripts/run_phase1_pipeline.py validate `
  --run-id run_YYYYMMDD_vis `
  --dataset data/processed/validation_clinical.csv `
  --holdout --visuals
```

### Linux / WSL

```bash
python3 -u scripts/inference/run_dicom_batch.py "/data/dicom_root" \
  --output results/dicom_batch_extract.csv \
  --accuracy-mode fast

python3 scripts/features/build_feature_reference.py
python3 scripts/run_phase1_pipeline.py integrate
python3 scripts/run_phase1_pipeline.py train
python3 scripts/run_phase1_pipeline.py validate \
  --run-id run_$(date +%Y%m%d) \
  --dataset data/processed/validation_clinical.csv \
  --holdout
```

---

## 3. DICOM / Inference — команды и режимы

### 3.1 Рекомендуемый batch: `run_dicom_batch.py`

| Аргумент | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `dicom_root` | путь (positional) | — | Корневая папка с подпапками пациентов |
| `--output` | путь | `results/dicom_batch_extract.csv` | Выходной CSV |
| `--accuracy-mode` | `high` \| `balanced` \| `fast` \| `minimal` | `fast` | Режим точности/скорости (см. таблицу ниже) |
| `--max-cases` | int | все | Ограничить число папок (для теста) |

**Примеры:**

```powershell
# Тест на 3 пациентах
py -3 -u scripts/inference/run_dicom_batch.py "D:\DICOM СНИМКИ" `
  --output results/dicom_test_3.csv --accuracy-mode minimal --max-cases 3

# Полный прогон, максимальное качество (долго)
py -3 -u scripts/inference/run_dicom_batch.py "D:\DICOM СНИМКИ" `
  --output results/dicom_full.csv --accuracy-mode high
```

### 3.2 Низкоуровневый: `enhanced_ct_extractor.py`

| Аргумент | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `dicom_root` | путь | — | Корень с подпапками (если не задан `--patient-folder`) |
| `--patient-folder` | путь | — | Одна папка пациента или корень с подпапками |
| `--output` | путь | `enhanced_ct_features.csv` | Выходной CSV |
| `--accuracy-mode` | см. ниже | `balanced` | Пресет срезов и downsample |
| `--downsample` | int | `2` | Шаг уменьшения матрицы (1=полное разрешение) |
| `--max-slices` | int | `300` | Макс. число срезов (переопределяется режимом) |
| `--debug` | flag | off | Подробные ошибки по срезам |
| `--disable-kidney-segmentation` | flag | off | Не искать почки (только демография/тело) |
| `--kidney-only` | flag | off | TotalSegmentator только ROI почек (если установлен) |

**Примеры:**

```powershell
# Все подпапки
py -3 -u scripts/inference/enhanced_ct_extractor.py "D:\DICOM СНИМКИ" `
  --output results/dicom_features.csv --accuracy-mode fast

# Один пациент
py -3 -u scripts/inference/enhanced_ct_extractor.py `
  --patient-folder "D:\DICOM СНИМКИ\Иванов" `
  --output results/ivanov.csv --accuracy-mode balanced

# Без почек (быстро, только метаданные + геометрия тела)
py -3 -u scripts/inference/enhanced_ct_extractor.py "D:\DICOM СНИМКИ" `
  --output results/dicom_no_kidney.csv --disable-kidney-segmentation
```

### 3.3 Таблица режимов `--accuracy-mode`

Используется в `enhanced_ct_extractor.py` и `run_dicom_batch.py`.

| Режим | max_slices | downsample | slice_strategy | Скорость | Когда использовать |
|-------|------------|------------|----------------|----------|-------------------|
| `minimal` | 25 | 6 | central | Самый быстрый | Smoke-тест, сотни пациентов |
| `fast` | 50 | 4 | uniform | Быстро | **Рекомендуется для большого batch** |
| `balanced` | 100 | 3 | uniform | Средне | Баланс качество/время |
| `high` | 200 | 2 | adaptive_kidney | Медленно | Максимум качества на малых когортах |

Явные `--max-slices` и `--downsample` переопределяют пресет **только если** вы меняете значения отличные от дефолта `300` / `2` в `enhanced_ct_extractor.py`.

### 3.4 Важно про displacement (таргеты)

Из **одной** CT-серии экстрактор **не** измеряет смещение почки при смене позы.

- Колонки `kidney_*_delta_*` в CSV = **`NaN`**
- Для обучения таргеты берутся из `data/vybor_unified_features.csv` и `data/train_displacement_dataset.csv`
- DICOM-CSV даёт **признаки** (геометрия, объёмы), не labels

---

## 4. Оркестратор: `run_phase1_pipeline.py`

```text
py -3 scripts/run_phase1_pipeline.py <command> [options]
```

| Команда | Назначение |
|---------|------------|
| `info` | Показать канонический поток и предупреждения о legacy-скриптах |
| `extract` | DICOM → CSV (обёртка над `enhanced_ct_extractor.py`) |
| `integrate` | Слияние источников → `data/processed/train.csv` |
| `train` | Обучение `models/adaptive_ensemble.pkl` |
| `validate` | Smoke + метрики (+ опционально визуализации) |

### 4.1 `extract`

| Аргумент | Обязателен | Описание |
|----------|------------|----------|
| `--dicom-root` | да | Путь к DICOM |
| `--output` | нет | CSV (default: `results/dicom_features.csv`) |
| `--max-slices` | нет | Пробрасывается в экстрактор |

```powershell
py -3 scripts/run_phase1_pipeline.py extract `
  --dicom-root "D:\DICOM СНИМКИ" `
  --output results/dicom_features.csv
```

### 4.2 `integrate`

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `--mode` | `labeled_only` | `labeled_only` — train только Vybor+Excel; `all` — legacy с KiTS deltas |
| `--no-kits-fill` | off | Не заполнять пропуски Excel медианами KiTS19 |

```powershell
# Рекомендуется (без fake KiTS displacement в train)
py -3 scripts/run_phase1_pipeline.py integrate

# Legacy-смешанный режим
py -3 scripts/run_phase1_pipeline.py integrate --mode all
```

Прямой вызов интегратора (те же флаги):

```powershell
py -3 src/models/data_integration_fix.py --mode labeled_only
py -3 src/models/data_integration_fix.py --mode labeled_only --excel-path data/train_displacement_dataset.csv
```

### 4.3 `train`

```powershell
py -3 scripts/run_phase1_pipeline.py train
# или
py -3 models/phase1/adaptive_ensemble.py
```

Артефакт: `models/adaptive_ensemble.pkl`

### 4.4 `validate`

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `--run-id` | **обязателен** | Имя папки в `results/validation_runs/` |
| `--dataset` | `data/processed/validation.csv` | CSV для оценки |
| `--model` | `models/adaptive_ensemble.pkl` | Путь к модели |
| `--out-dir` | `results/validation_runs` | Корень отчётов |
| `--seed` | `42` | Seed split (если без `--holdout`) |
| `--test-size` | `0.5` | Доля eval при re-split |
| `--holdout` | off | Оценить **весь** файл без повторного split |
| `--source` | — | Фильтр: `Vybor`, `Vybor,Excel` |
| `--visuals` | off | Дополнительно `run_visual_tests.py` |
| `--num-cases` | `8` | Число кейсов для визуализаций |

```powershell
# Честный аудит Vybor holdout
py -3 scripts/run_phase1_pipeline.py validate `
  --run-id vybor_holdout_20260610 `
  --dataset data/processed/validation_clinical.csv `
  --holdout

# Только подмножество Vybor
py -3 scripts/run_phase1_pipeline.py validate `
  --run-id vybor_only `
  --dataset data/processed/validation_vybor_only.csv `
  --holdout --source Vybor
```

---

## 5. Валидация — прямые скрипты

### 5.1 Smoke

```powershell
py -3 scripts/validation/smoke_check.py `
  --dataset data/processed/validation.csv `
  --model models/adaptive_ensemble.pkl
```

### 5.2 Метрики

| Аргумент | По умолчанию |
|----------|--------------|
| `--dataset` | `data/vybor_unified_features.csv` |
| `--model` | `models/adaptive_ensemble.pkl` |
| `--run-id` | обязателен |
| `--out-dir` | `results/validation_runs` |
| `--seed` | `42` |
| `--test-size` | `0.3` |
| `--top-n` | `10` |
| `--source` | фильтр по колонке `source` |
| `--holdout` | без re-split |

```powershell
py -3 scripts/validation/evaluate_metrics.py `
  --dataset data/processed/validation_clinical.csv `
  --model models/adaptive_ensemble.pkl `
  --run-id my_run `
  --holdout
```

Результаты: `results/validation_runs/<run-id>/metrics/`

### 5.3 Визуализации

```powershell
py -3 scripts/validation/run_visual_tests.py `
  --dataset data/processed/validation_clinical.csv `
  --model models/adaptive_ensemble.pkl `
  --run-id my_run_vis `
  --num-cases 6
```

### 5.4 WSL: всё одной командой

```bash
export DATASET_PATH="data/processed/validation_clinical.csv"
export MODEL_PATH="models/adaptive_ensemble.pkl"
bash scripts/validation/run_all.sh run_20260610 all
# MODE: all | metrics | visuals
```

---

## 6. KiTS reference (без displacement в train)

```powershell
py -3 scripts/features/build_feature_reference.py
py -3 scripts/features/build_feature_reference.py --kits-csv data/kits19_medical_grade_features.csv --out-dir data/processed
```

Выход: `data/processed/kits19_feature_reference.csv`, `kits19_feature_medians.json`

---

## 7. API inference (после обучения)

```powershell
py -3 -m uvicorn src.api.kidney_displacement_api:app --host 127.0.0.1 --port 8000
# или
py -3 src/api/kidney_displacement_api.py
```

Документация Swagger: http://127.0.0.1:8000/docs

---

## 8. Тесты

```powershell
py -3 -m pytest tests/ -q
py -3 -m pytest tests/test_feature_schema.py tests/test_feature_pipeline.py -q
py -3 -m pytest tests/test_excel_displacement_adapter.py -q
```

---

## 9. Выходные файлы (чеклист)

| Шаг | Файл |
|-----|------|
| DICOM extract | `results/dicom_batch_extract.csv` |
| Интеграция | `data/processed/train.csv`, `validation.csv`, `integration_manifest.json` |
| Обучение | `models/adaptive_ensemble.pkl` |
| Валидация | `results/validation_runs/<run-id>/metrics/*.csv` |
| Визуалы | `results/validation_runs/<run-id>/plots/*.png` |

---

## 10. Типичные проблемы

| Симптом | Решение |
|---------|---------|
| `No module named 'pydicom'` | `pip install pydicom scikit-image nibabel` |
| `UnicodeEncodeError` в консоли | Используйте `py -3 -u` и актуальную версию скриптов из репозитория |
| Очень долгий DICOM batch | `--accuracy-mode minimal` или `fast`; не ставить TotalSegmentator без нужды |
| `Too few training rows` | Проверьте `data/vybor_unified_features.csv` и полноту `kidney_*_delta_*` |
| KiTS «улучшает» метрики в legacy | Используйте `integrate --mode labeled_only` |

---

## 11. Legacy (не использовать для нового прогона)

- `scripts/inference/dicom_feature_extractor.py` → заменён на `enhanced_ct_extractor.py`
- `src/data/prepare_dataset.py`
- `models/phase1/train_lasso.py`, `train_ridge.py` без `run_phase1_pipeline.py`

Канонический вход: **`scripts/run_phase1_pipeline.py`**

---

См. также: [PHASE1_PIPELINE_RUNBOOK.md](PHASE1_PIPELINE_RUNBOOK.md) — архитектура и контекст.
