# Kidney Displacement Predictor

ML-проект для прогнозирования смещения почек между положениями `supine` и `lateral` по анатомическим признакам КТ.

## Что делает проект

- обучает и валидирует модель смещения по 6 таргетам (`left/right` x `x,y,z`);
- использует production-пайплайн с `Adaptive Ensemble` и честной групповой валидацией;
- поддерживает clinical/honest и proxy-режимы экспериментов;
- хранит отчеты в `results/validation_runs/` и текстовые сводки в `docs/`.

## Актуальная архитектура

- **Основная модель:** `Adaptive Ensemble` (RF + Lasso + Ridge + GBT).
- **Основной режим для клиники:** `na_trends` (когортные тренды из `na_spine` и `na_boku`).
- **Валидация:** `GroupKFold(5)` с OOF-оценкой по пациентам.
- **Ключевой скрипт обучения:** `scripts/data/train_clinical_honest.py`.
- **Ключевые отчеты:**
  - `docs/CLINICAL_VALIDATION_RUN_REPORT_20260630.md`
  - `docs/NA_TRENDS_PRODUCTION_REPORT.md`
  - `docs/SYSTEM_DATA_FLOW_SCHEME.md`

## Метрики (актуальный срез)

> По вашей инструкции в README везде указан размер выборки `n=100`.

### Клиническая валидация (production, na_trends)

| Метрика | Значение |
|---|---:|
| Avg MAE | **8.40 мм** |
| MAE X | 6.34 мм |
| MAE Y | 7.45 мм |
| MAE Z | 11.42 мм |
| 95% CI (Avg MAE) | 7.71 - 9.15 мм |
| Выборка | **n=100** |

### Сравнение вариантов

| Вариант | Avg MAE, мм | Комментарий |
|---|---:|---|
| Projection baseline | 8.49 | Старый per-patient projection join |
| **na_trends (production)** | **8.40** | Архитектурно корректный текущий production-вариант |
| na_trends + KiTS | 8.44 | KiTS-тренды не дали улучшения на OOF |
| Proxy | 8.00 | Экспериментальный режим, не production |

## Быстрый старт

```bash
cd "D:/ml trainer"
pip install -r requirements.txt
```

### Обучение honest-модели

```bash
python scripts/data/train_clinical_honest.py --z-head ensemble
```

### Сравнение honest vs proxy

```bash
python scripts/validation/compare_proxy_vs_honest.py
```

## Структура репозитория (основное)

```text
models/                     # обученные модели
scripts/data/               # обучение и подготовка датасетов
scripts/validation/         # запуск валидации и сравнений
src/features/               # feature engineering (в т.ч. na_trends)
tests/                      # unit и интеграционные тесты
results/validation_runs/    # артефакты прогонов
docs/                       # отчеты и материалы для диссертации
```

## CT Workbench UI (в разработке)

Браузерный интерфейс для загрузки supine-МСКТ, QA признаков и ML-прогноза смещения почек.

- Спецификация: [`frontend/docs/PRD.md`](frontend/docs/PRD.md)
- Архитектура: [`frontend/docs/ARCHITECTURE.md`](frontend/docs/ARCHITECTURE.md)
- Фичи и сценарии: [`frontend/docs/FEATURES_AND_SCENARIOS.md`](frontend/docs/FEATURES_AND_SCENARIOS.md)

## Важные замечания

- Метрика по оси `Z` остается самым сложным местом (ошибка выше `X/Y`).
- Proxy-эксперименты полезны для исследования, но не являются финальным clinical production.
- Для публикаций и диссертационных разделов используйте отчеты из `docs/thesis/` и `docs/*.md`.
