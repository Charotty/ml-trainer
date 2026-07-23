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

## Быстрый старт

```bash
cd /path/to/ml-trainer
pip install -r requirements.txt
```

### Обучение honest-модели

```bash
python scripts/data/train_clinical_honest.py --z-head ensemble
```

Артефакт: `models/adaptive_ensemble_clinical_honest.pkl`.

### Сравнение honest vs proxy

```bash
python scripts/validation/compare_proxy_vs_honest.py
```

## Структура репозитория (основное)

```text
models/                     # обученные модели (в т.ч. clinical_honest.pkl)
scripts/data/               # обучение и подготовка датасетов
scripts/validation/         # запуск валидации и сравнений
src/features/               # feature engineering (в т.ч. na_trends)
src/api/                    # FastAPI (legacy + CT Workbench)
frontend/public/            # UI CT Workbench
tests/                      # unit и интеграционные тесты
results/validation_runs/    # артефакты прогонов
docs/                       # отчёты и материалы для диссертации
```

## CT Workbench UI

Браузерный интерфейс для загрузки supine-МСКТ, QA признаков и ML-прогноза смещения почек.

- Спецификация: [`frontend/docs/PRD.md`](frontend/docs/PRD.md)
- Запуск: `python -m uvicorn src.api.ct_workbench_api:app --port 8010` → http://127.0.0.1:8010/

## Важные замечания

- **Proxy ≠ production:** proxy-эксперименты полезны для исследования, но clinical production — только honest-путь (`scripts/data/train_clinical_honest.py` → `models/adaptive_ensemble_clinical_honest.pkl`).
- **KiTS опционален:** для honest-обучения KiTS не обязателен (тренды без `--with-kits`).
- Операционный чеклист (staging → build → train → validate → API): [`docs/REPO_WORK_CHECKLIST.md`](docs/REPO_WORK_CHECKLIST.md).
- Метрика по оси `Z` остаётся самым сложным местом (ошибка выше `X/Y`).
- Система исследовательская / вспомогательная; не заменяет клиническое решение врача.
- Для публикаций и диссертационных разделов используйте отчёты из `docs/thesis/` и `docs/*.md`.
