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
- **Рабочий alias (не победитель):** `models/adaptive_ensemble_clinical_honest.pkl` — currently the **111-feature** working-tree file. Git HEAD archive is the **121-feature** candidate. Neither is promoted.
- **Основной режим для клиники:** `na_trends` (когортные тренды из `na_spine` и `na_boku`) + клинические demographics.
- **Валидация:** требуется исправленная nested `GroupKFold` OOF на n=87; исторический не-nested GKF-OOF не является production-доказательством.
- **Ключевой скрипт обучения:** `scripts/data/train_clinical_honest.py`.
- **Источник метрик и гейтов:** [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), [`docs/MODEL_RELEASE_GATES.md`](docs/MODEL_RELEASE_GATES.md), JSON-карточки в `models/archive/`.

## Метрики (заморозка, не production)

**Ни 8.52 мм, ни 8.40 / 8.49 мм не доказаны для текущего alias.** Оба кандидата в архиве, `production_winner: false`. Исторический MAE **2.14 мм** — leaky/in-sample, не production. Исправленная nested OOF на n=87 для f111 vs f121 ещё не прогонялась в этом блоке.

| Кандидат | Признаки | SHA prefix | Статус |
|---|---:|---|---|
| working-tree snapshot | **111** | `3E03B8FA` | archived, research-only |
| Git HEAD archive | **121** | `5F317838` | archived, research-only |

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
# ожидается: "status":"ok", "model_loaded":true;
# feature_count = число признаков загруженного файла (111 или 121), не флаг победителя.
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

- **Proxy ≠ clinical labels:** honest-путь (`scripts/data/train_clinical_honest.py` → alias `.pkl`). The alias is **not** a promoted winner.
- **KiTS опционален** для honest-обучения.
- Карточка модели / гейты: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), [`docs/MODEL_RELEASE_GATES.md`](docs/MODEL_RELEASE_GATES.md).
- Операционный чеклист: [`docs/REPO_WORK_CHECKLIST.md`](docs/REPO_WORK_CHECKLIST.md).
- Система исследовательская / вспомогательная; не заменяет клиническое решение врача.
