# CT Workbench UI

Браузерный интерфейс для загрузки supine-МСКТ, извлечения анатомических признаков, ручной проверки (QA) и ML-прогноза смещения почек.

## Документация

| Документ | Назначение |
|----------|------------|
| [docs/PRD.md](docs/PRD.md) | Product Requirements Document |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Архитектура, API, ML-интеграция |
| [docs/FEATURES_AND_SCENARIOS.md](docs/FEATURES_AND_SCENARIOS.md) | Фичи F-xxx и сценарии US-xx |
| [docs/UI_IMPROVEMENTS.md](docs/UI_IMPROVEMENTS.md) | Требуемые улучшения UI (ревью MVP) |

## Запуск (MVP)

### 1. Обучить production-модель (если нет `.pkl`)

```bash
python scripts/data/train_clinical_honest.py --z-head ensemble
```

### 2. Запустить API + UI

```bash
python -m uvicorn src.api.ct_workbench_api:app --host 127.0.0.1 --port 8010 --reload
```

Откройте в браузере: http://127.0.0.1:8010/

### 3. Поток работы

1. **Создать кейс** → загрузить DICOM zip
2. **Запустить extraction** (async, может занять несколько минут)
3. **Проверить/править** 17 ключевых признаков в QA-форме
4. **Прогноз** → 6 значений ΔX/Y/Z
5. **Скачать** `report.json`

## Структура

```text
frontend/
  public/          # MVP UI (HTML/CSS/JS)
  docs/            # PRD, architecture, features
src/api/
  ct_workbench_api.py
  cases/           # Cases REST API
data/cases/        # хранилище кейсов (gitignored)
```

## API

Базовый URL: `http://127.0.0.1:8010/api/v1/cases`

| Метод | Endpoint | Описание |
|-------|----------|----------|
| POST | `/cases` | Создать кейс |
| POST | `/cases/{id}/upload` | Загрузить DICOM zip |
| POST | `/cases/{id}/analyze` | Запустить extraction |
| GET | `/cases/{id}/status` | Статус job |
| GET | `/cases/{id}/features` | Признаки |
| PATCH | `/cases/{id}/features/manual` | Ручная правка |
| POST | `/cases/{id}/predict` | ML-прогноз |
| GET | `/cases/{id}/report.json` | Отчёт |

## Связь с backend ML

- Feature pipeline: `src/features/pipeline.py`
- DICOM extraction: `scripts/inference/extract_from_dicom.py`
- Production-модель: `models/adaptive_ensemble_clinical_honest.pkl`
