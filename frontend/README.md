# CT Workbench UI

Браузерный интерфейс для загрузки supine-МСКТ, извлечения анатомических признаков, ручной проверки (QA) и ML-прогноза смещения почек.

На текущем этапе в этой папке находится **только спецификация** (документация). Исходный код UI будет добавлен позже; выбор frontend-стека отложен.

## Документация

| Документ | Назначение |
|----------|------------|
| [docs/PRD.md](docs/PRD.md) | Product Requirements Document — цели, scope, требования |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Архитектура системы, API, интеграция с ML-пайплайном |
| [docs/FEATURES_AND_SCENARIOS.md](docs/FEATURES_AND_SCENARIOS.md) | Каталог фич (F-xxx) и пользовательские сценарии (US-xx) |

## Связь с backend

ML-логика и inference остаются в корне репозитория:

- API (текущий): `src/api/kidney_displacement_api.py`
- Feature pipeline: `src/features/pipeline.py`
- DICOM extraction: `scripts/inference/extract_from_dicom.py`
- Production-модель: `models/adaptive_ensemble_clinical_honest.pkl`

Новые REST-endpoint'ы для кейсов планируются в `src/api/cases/` (см. ARCHITECTURE.md).
