# Kidney Displacement Predictor

ML-проект для прогнозирования смещения почек между положениями `supine` и `lateral` по анатомическим признакам КТ.

## Что делает проект

- обучает и валидирует модель смещения по 6 таргетам (`left/right` × `x,y,z`);
- использует production-пайплайн с `Adaptive Ensemble` и честной групповой валидацией;
- включает клинические табличные признаки (пол, возраст, BMI, тип телосложения, предшествующие операции);
- поддерживает clinical/honest и proxy-режимы экспериментов;
- хранит отчёты в `results/validation_runs/` и текстовые сводки в `docs/`.

## Актуальная архитектура

- **Основная модель:** `Adaptive Ensemble` (RF + Lasso + Ridge + GBT).
- **Артефакт production:** `models/adaptive_ensemble_clinical_honest.pkl` (**121** признак).
- **Основной режим для клиники:** `na_trends` (когортные тренды из `na_spine` и `na_boku`) + клинические demographics.
- **Валидация:** `GroupKFold(5)` с OOF-оценкой по пациентам.
- **Ключевой скрипт обучения:** `scripts/data/train_clinical_honest.py`.
- **Ключевые отчёты:**
  - `docs/CLINICAL_VALIDATION_RUN_REPORT_20260630.md`
  - `docs/NA_TRENDS_PRODUCTION_REPORT.md`
  - `docs/SYSTEM_DATA_FLOW_SCHEME.md`
  - `docs/REPO_WORK_CHECKLIST.md`

## Метрики (актуальный срез)

GroupKFold-OOF на клинических парных метках из displacement XLSX.

### Клиническая валидация (production, na_trends)

| Метрика | Значение |
|---|---:|
| Avg MAE | **8.52 мм** |
| MAE Z | 11.63 мм |
| 95% CI (Avg MAE) | 7.82 – 9.26 мм |
| Признаков | **121** |
| Выборка | **n=87** |

## Быстрый старт (локально)

```bash
cd /path/to/ml-trainer
pip install -r requirements.txt
python -m uvicorn src.api.ct_workbench_api:app --host 0.0.0.0 --port 8010
```

Открыть: http://127.0.0.1:8010/

### Обучение honest-модели

```bash
python scripts/data/train_clinical_honest.py --z-head ensemble
```

Артефакт: `models/adaptive_ensemble_clinical_honest.pkl`.

## Развёртывание в Docker

Ниже — рабочий путь для **CT Workbench** (UI + API на порту **8010**).  
Модель в git обычно не лежит (`models/*.pkl` в `.gitignore`): файл должен быть на хосте и монтируется в контейнер.

### Что нужно на машине

1. [Docker Engine](https://docs.docker.com/engine/install/) + Docker Compose v2 (`docker compose version`).
2. Файл модели: `models/adaptive_ensemble_clinical_honest.pkl`.
3. Для GPU-профиля: NVIDIA GPU, драйвер и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Структура Docker-файлов

| Файл | Назначение |
|------|------------|
| `Dockerfile` | CPU-образ: API, UI, predict, PDF |
| `Dockerfile.gpu` | GPU-образ: + CUDA PyTorch + TotalSegmentator |
| `docker-compose.yml` | сервис `workbench` (CPU) и `workbench-gpu` (profile `gpu`) |
| `requirements-docker.txt` | зависимости CPU-образа |
| `.dockerignore` | исключает `dicexe/`, кейсы, venv, большие архивы |

Переменные окружения:

| Переменная | По умолчанию | Смысл |
|------------|--------------|--------|
| `MODEL_PATH` | `/app/models/adaptive_ensemble_clinical_honest.pkl` | путь к `.pkl` внутри контейнера |
| `CASES_ROOT` | `/data/cases` | хранилище кейсов DICOM/артефактов |

### Запуск CPU (рекомендуется для проверки)

```bash
# из корня репозитория
# убедитесь, что модель на месте:
ls -lh models/adaptive_ensemble_clinical_honest.pkl

docker compose up -d --build
```

Проверка:

```bash
curl -s http://127.0.0.1:8010/health
# ожидается: "status":"ok", "model_loaded":true, "feature_count":121
```

UI: http://127.0.0.1:8010/

Остановка:

```bash
docker compose down
```

Кейсы сохраняются в Docker volume `workbench_cases` (не пропадают при пересборке образа).

### Запуск GPU (полный analyze с TotalSegmentator)

CPU-контейнер умеет API и прогноз; тяжёлая сегментация DICOM рассчитана на GPU-профиль:

```bash
# остановите CPU-сервис, если занимает порт 8010
docker compose down

docker compose --profile gpu up -d --build workbench-gpu
curl -s http://127.0.0.1:8010/health
```

Первый запуск TotalSegmentator может скачать веса в volume `totalseg_cache`.

### Типичные проблемы

| Симптом | Что проверить |
|---------|----------------|
| `model_loaded: false` | есть ли `models/adaptive_ensemble_clinical_honest.pkl` на хосте и смонтирован ли volume |
| порт занят | `docker compose down` или другой процесс на 8010 |
| GPU не виден | `nvidia-smi` на хосте; установлен ли NVIDIA Container Toolkit; профиль `gpu` |
| огромный контекст сборки | не убирайте `.dockerignore`; не кладите `dicexe/` и zip в корень без игнора |
| analyze падает в CPU-образе | для сегментации нужен `workbench-gpu` |
| UI не открывается с хоста в чистом WSL-Docker | проверьте `docker compose ps` и проброс портов; надёжнее Docker Desktop; health внутри контейнера: `docker compose exec workbench curl -s localhost:8010/health` |

### Инженерные замечания по образу

- В образ **не** копируются веса модели и `data/cases` — только код; модель и кейсы монтируются.
- Один worker uvicorn: сегментация и joblib-модель не рассчитаны на многопроцессный sharing без доработки.
- Healthcheck бьёт в `/health`.
- Legacy API (`kidney_displacement_api`, порт 8000) этим compose **не** поднимается — канонический вход: CT Workbench `:8010`.

## Структура репозитория (основное)

```text
models/                     # обученные модели (в т.ч. clinical_honest.pkl)
scripts/data/               # обучение и подготовка датасетов
scripts/validation/         # запуск валидации и сравнений
src/features/               # feature engineering (в т.ч. na_trends)
src/api/                    # FastAPI (legacy + CT Workbench)
frontend/public/            # UI CT Workbench
tests/                      # unit и интеграционные тесты
Dockerfile                  # CPU-образ Workbench
Dockerfile.gpu              # GPU-образ Workbench
docker-compose.yml
docs/                       # отчёты и материалы
```

## CT Workbench UI

Браузерный интерфейс для загрузки supine-МСКТ, QA признаков и ML-прогноза смещения почек.

- Спецификация: [`frontend/docs/PRD.md`](frontend/docs/PRD.md)
- Локально: `python -m uvicorn src.api.ct_workbench_api:app --port 8010`
- Docker: см. раздел выше

## Важные замечания

- **Proxy ≠ production:** clinical production — honest-путь (`scripts/data/train_clinical_honest.py` → `models/adaptive_ensemble_clinical_honest.pkl`).
- **KiTS опционален** для honest-обучения.
- Операционный чеклист: [`docs/REPO_WORK_CHECKLIST.md`](docs/REPO_WORK_CHECKLIST.md).
- Система исследовательская / вспомогательная; не заменяет клиническое решение врача.
