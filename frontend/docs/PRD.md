# CT Workbench — Product Requirements Document (PRD)

**Версия:** 0.1 (спецификация)  
**Дата:** 2026-07-09  
**Статус:** Draft — этап документации, без реализации кода UI

---

## 1. Резюме продукта

**CT Workbench** — локальное браузерное приложение для загрузки supine-МСКТ, автоматического извлечения анатомических признаков, ручной проверки (QA) и ML-прогноза смещения почек при переходе в lateral (6 значений ΔX, ΔY, ΔZ для левой и правой почки).

**Целевая аудитория:** врачи-урологи/хирурги, исследователи (диссертация), инженеры ML.

**Детальная документация:**

| Документ | Содержание |
|----------|------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Архитектура, API, ML-интеграция, NFR, roadmap компонентов |
| [FEATURES_AND_SCENARIOS.md](FEATURES_AND_SCENARIOS.md) | Каталог фич F-001…F-018, сценарии US-01…US-08 |

---

## 2. Проблема и ценность

### 2.1. Проблема

При лапароскопической операции на почке положение органа на боку **существенно отличается** от положения на спине (медиальное смещение до ~12 мм, вентральное до ~15 мм, каудальное левой почки ~8.6 мм — см. [docs/дисер.md](../../docs/дисер.md)). Единая поправка для всех пациентов невозможна (CV ~80% по оси Y).

### 2.2. Решение

Предоперационный прогноз смещения по **одному supine-КТ** с помощью production-модели `Adaptive Ensemble` + когортные тренды `na_trends` (см. [docs/NA_TRENDS_PRODUCTION_REPORT.md](../../docs/NA_TRENDS_PRODUCTION_REPORT.md)).

### 2.3. Ценность для пользователя

| Пользователь | Ценность |
|--------------|----------|
| Врач | Прогноз смещения без lateral-КТ; подготовка к AR-наложению |
| Исследователь | Воспроизводимый пайплайн, отчёты, batch для валидации |
| Инженер | Единый UI поверх существующего `ml-trainer` репозитория |

Сравнение с диссертацией: [docs/thesis/10_CLINICAL_VALIDATION_AND_COMPARISON.md](../../docs/thesis/10_CLINICAL_VALIDATION_AND_COMPARISON.md).

---

## 3. Цели и не-цели

### 3.1. Goals (цели)

| ID | Цель |
|----|------|
| G-01 | Загрузить DICOM supine и получить ML-прогноз за один рабочий сеанс |
| G-02 | Позволить ручную правку критичных анатомических точек |
| G-03 | Использовать production-модель `adaptive_ensemble_clinical_honest.pkl` (na_trends) |
| G-04 | Обеспечить паритет признаков train/infer через `src/features/pipeline.py` |
| G-05 | Экспортировать результат в JSON (MVP) и PDF (v2) |
| G-06 | Хранить кейсы локально с audit trail ручных правок |

### 3.2. Non-goals (не-цели)

| ID | Не делаем |
|----|-----------|
| NG-01 | Полноценный PACS / интеграция с больничной ИС |
| NG-02 | Облачный SaaS с хранением PHI |
| NG-03 | Интраоперационная навигация внутри браузера |
| NG-04 | Обучение модели из UI (только inference) |
| NG-05 | Замена клинического протокола — инструмент исследовательский |

---

## 4. Scope по фазам

### 4.1. MVP (must-have)

| Фича | Описание | Детали |
|------|----------|--------|
| F-001 | Upload DICOM | [FEATURES § F-001](FEATURES_AND_SCENARIOS.md#f-001--загрузка-dicom) |
| F-002 | Автовыбор серии | [FEATURES § F-002](FEATURES_AND_SCENARIOS.md#f-002--автовыбор-серии-abdomen-supine) |
| F-003 | Async extraction | [FEATURES § F-003](FEATURES_AND_SCENARIOS.md#f-003--async-extraction-job--прогресс) |
| F-004 | Авто BASE_FEATURES | [FEATURES § F-004](FEATURES_AND_SCENARIOS.md#f-004--автозаполнение-23-base_features) |
| F-005 | QA ручная правка | [FEATURES § F-005](FEATURES_AND_SCENARIOS.md#f-005--ручная-правка-координат-почек-qa) |
| F-006 | Predict 6 таргетов | [FEATURES § F-008](FEATURES_AND_SCENARIOS.md#f-008--прогноз-6-смещений-мм) |
| F-007 | JSON-отчёт | [FEATURES § F-017](FEATURES_AND_SCENARIOS.md#f-017--json-отчёт-mvp-export) |

**MVP-сценарии:** US-01, US-02, US-03 (без diff на MVP — только повторный predict).

### 4.2. v1.1 (should-have)

| Фича | Описание |
|------|----------|
| F-008 | DICOM viewer + оверлеи (F-016) |
| F-009 | Coverage % (F-007) |
| F-010 | Diff до/после правки (F-010) |
| F-011 | Dashboard кейсов (F-011) |
| F-012 | Расширенные поля lordosis/abd wall (F-006) |
| F-013 | Подсветка оси Z (F-009) |

**Сценарии:** US-07, частично US-05.

### 4.3. v2 (nice-to-have)

| Фича | Описание |
|------|----------|
| F-014 | PDF-отчёт (F-015) |
| F-015 | AR JSON export (F-018) |
| F-016 | Batch когорта (F-013) |
| F-017 | История версий (F-012) |
| F-018 | Режим proxy vs honest (F-014) |

**Сценарии:** US-04, US-05, US-06, US-08.

---

## 5. Функциональные требования (FR)

| ID | Требование | Фича | Сценарий |
|----|------------|------|----------|
| FR-001 | Система SHALL принимать DICOM zip и создавать кейс | F-001 | US-01 |
| FR-002 | Система SHALL запускать extraction асинхронно с отображением прогресса | F-003 | US-01 |
| FR-003 | Система SHALL заполнять 23 BASE_FEATURES после extraction | F-004 | US-01 |
| FR-004 | Пользователь SHALL иметь возможность редактировать координаты почек и опорные точки | F-005 | US-02 |
| FR-005 | Система SHALL возвращать 6 предсказаний смещения в мм | F-008 | US-01, US-03 |
| FR-006 | Система SHALL использовать `adaptive_ensemble_clinical_honest.pkl` и `na_trends` | F-008 | US-01 |
| FR-007 | Система SHALL предоставлять скачивание `report.json` | F-017 | US-03, US-06 |
| FR-008 | Система SHALL логировать все ручные правки с timestamp | F-005 | US-02 |
| FR-009 | Система SHALL показывать coverage признаков (v1.1) | F-007 | US-07 |
| FR-010 | Система SHALL показывать diff прогноза до/после правки (v1.1) | F-010 | US-03 |
| FR-011 | Система SHALL генерировать PDF-отчёт (v2) | F-015 | US-06 |
| FR-012 | Система SHALL экспортировать AR JSON по контракту (v2) | F-018 | US-04 |

Полные acceptance criteria — в [FEATURES_AND_SCENARIOS.md](FEATURES_AND_SCENARIOS.md).

---

## 6. Нефункциональные требования (NFR)

См. [ARCHITECTURE.md § 8](ARCHITECTURE.md#8-нефункциональные-требования-nfr).

| ID | Требование | Целевое значение |
|----|------------|------------------|
| NFR-001 | Latency predict | < 2 с |
| NFR-002 | Extraction | Async, 2–15 мин/кейс |
| NFR-003 | CORS | Enabled для UI |
| NFR-004 | Data residency | Локально |
| NFR-005 | Audit trail | 100% manual overrides |
| NFR-006 | Feature parity | Единый `pipeline.py` |
| NFR-007 | Disclaimer | В каждом отчёте |
| NFR-008 | Coverage warning | < 90% — warning |

---

## 7. Метрики успеха

### 7.1. Технические

| Метрика | Цель | Источник |
|---------|------|----------|
| Feature coverage (BASE) | ≥ 90% кейсов с coverage ≥ 80% | F-007 |
| Время до прогноза (после extract) | < 30 с пользовательских действий | US-01 |
| Успешность extraction | ≥ 85% загруженных zip | логи worker |
| Predict latency | < 2 с p95 | NFR-001 |

### 7.2. Клинические / научные

| Метрика | Референс | Источник |
|---------|----------|----------|
| Avg MAE (production) | ~8.40 мм GKF-OOF | [README.md](../../README.md) |
| MAE Z | ~11.42 мм (ожидаемо выше) | [thesis/10](../../docs/thesis/10_CLINICAL_VALIDATION_AND_COMPARISON.md) |
| Выборка в документации | n=100 | README (публичная формулировка) |

UI **не обязан** улучшать MAE на MVP — цель: воспроизвести production inference с QA.

---

## 8. Зависимости от репозитория ml-trainer

| Компонент | Путь | Готовность |
|-----------|------|------------|
| DICOM extraction | `scripts/inference/extract_from_dicom.py` | Есть (CLI) |
| Feature schema | `src/features/phase1_schema.py` | Есть |
| Inference pipeline | `src/features/pipeline.py` | Есть |
| NaTrendStore | `src/features/na_trend_features.py` | Есть |
| Production model | `models/adaptive_ensemble_clinical_honest.pkl` | Локально (не в git) |
| Legacy API | `src/api/kidney_displacement_api.py` | Есть, требует обновления |
| Cases API | `src/api/cases/` | **Не реализован** |
| Frontend UI | `frontend/src/` | **Не реализован** |
| Worker / queue | — | **Не реализован** |

---

## 9. Риски и допущения

### 9.1. Риски

См. [ARCHITECTURE.md § 10](ARCHITECTURE.md#10-риски-и-митигации).

### 9.2. Допущения

| ID | Допущение |
|----|-----------|
| A-01 | Пользователь имеет локально обученную `clinical_honest.pkl` |
| A-02 | GPU доступен для TotalSegmentator (или fallback CPU с предупреждением) |
| A-03 | Когортные CSV `na_spine` / `na_boku` присутствуют в `data/harmonized/` |
| A-04 | Исследования — supine abdominal CT |
| A-05 | Frontend stack будет выбран на этапе 3 (после Cases API) |

---

## 10. План поставки (delivery)

| Этап | Содержание | Статус |
|------|------------|--------|
| **1** | Документация + `frontend/` layout | **Текущий** (ветка `feature/ct-workbench-ui-spec`) |
| **2** | Cases API + worker + `data/cases/` storage | Запланирован |
| **3** | UI MVP: upload, form, predict, JSON report | Запланирован |
| **4** | Viewer, PDF, batch, AR export | Запланирован |

Архитектурные детали этапов: [ARCHITECTURE.md § 11](ARCHITECTURE.md#11-план-реализации-по-этапам).

---

## 11. Открытые вопросы (TBD)

| ID | Вопрос | Влияние |
|----|--------|---------|
| TBD-01 | Frontend stack (React/Vue/другое) | Этап 3 |
| TBD-02 | Job queue: RQ vs Celery vs subprocess | Этап 2 |
| TBD-03 | Хранение DICOM: локальный диск vs сетевой share | Deployment |
| TBD-04 | Аутентификация пользователей | v1.1+ |
| TBD-05 | Максимальный размер upload zip | Infra |

---

## 12. Трассируемость FR → F → US → Component

| FR | Feature | Scenario | Architecture component |
|----|---------|----------|------------------------|
| FR-001 | F-001 | US-01 | Cases API upload |
| FR-002 | F-003 | US-01 | Worker + Jobs API |
| FR-003 | F-004 | US-01 | extract_from_dicom + schema |
| FR-004 | F-005 | US-02 | PATCH features/manual |
| FR-005 | F-008 | US-01, US-03 | pipeline.predict_targets |
| FR-006 | F-008 | US-01 | clinical_honest.pkl + NaTrendStore |
| FR-007 | F-017 | US-03, US-06 | Report endpoint |
| FR-008 | F-005 | US-02 | manual_overrides.json |
| FR-009 | F-007 | US-07 | validate_base_features |
| FR-010 | F-010 | US-03 | Predict panel |
| FR-011 | F-015 | US-06 | Report PDF service |
| FR-012 | F-018 | US-04 | AR export |

---

## 13. Утверждение и ссылки

| Роль | Документ для review |
|------|---------------------|
| Product / исследователь | Этот PRD + [FEATURES_AND_SCENARIOS.md](FEATURES_AND_SCENARIOS.md) |
| Архитектор / инженер | [ARCHITECTURE.md](ARCHITECTURE.md) |
| ML | [docs/SYSTEM_DATA_FLOW_SCHEME.md](../../docs/SYSTEM_DATA_FLOW_SCHEME.md) |
| Клиника | [docs/thesis/10_CLINICAL_VALIDATION_AND_COMPARISON.md](../../docs/thesis/10_CLINICAL_VALIDATION_AND_COMPARISON.md) |

**Точка входа в UI-пакет:** [frontend/README.md](../README.md)
