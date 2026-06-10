# Code Audit Fixes Log (1.1 — 1.13, 2.1)

Журнал изменений по критическим пунктам из `docs/CODE_AUDIT_REPORT.md`.
Каждая запись содержит: статус, затронутые файлы, суть правки, выявленные противоречия (если есть), верификацию.

Легенда статусов: `pending`, `in_progress`, `done`, `blocked` (требует решения).

---

## Сводная таблица

| ID | Проблема | Статус | Файлы |
|----|----------|--------|-------|
| 1.1 | Несогласованность feature pipeline train↔inference | done | `models/phase1/adaptive_ensemble.py`, `src/api/kidney_displacement_api.py` |
| 1.2 | `KidneyDisplacementPredictor.predict` использует неинициализированный ансамбль | done | `models/phase1/kidney_displacement_predictor.py` |
| 1.3 | Shared sklearn-estimators между таргетами | done | `models/phase1/adaptive_ensemble.py`, `models/phase1/target_specific_ensemble.py` |
| 1.4 | `train_lasso.py` / `train_ridge.py` падают при первом запуске | done | `models/phase1/train_lasso.py`, `models/phase1/train_ridge.py` |
| 1.5 | Синтетические признаки/таргеты в инференс-пайплайнах | done | `scripts/inference/enhanced_ct_extractor.py`, `scripts/inference/dicom_feature_extractor.py`, `src/models/data_integration_fix.py` |
| 1.6 | Train/val leakage | done | `models/phase1/adaptive_ensemble.py`, `scripts/inference/convert_single_file.py` |
| 1.7 | API возвращает HTTP 500 на клиентские ошибки | done | `src/api/kidney_displacement_api.py` |
| 1.8 | Несовместимые API контракты для одной задачи | done | `src/api/kidney_displacement_api.py`, `src/api/api_server.py`, `models/phase1/api_kidney_predictor.py` |
| 1.9 | Тихая работа AR-системы без модели | done | `src/ar_system/kidney_ar_system.py` |
| 1.10 | `FallbackHandler` путает позицию и смещение | done | `src/reliability/confidence_constraints.py` |
| 1.11 | `DataValidator.validate_processed_data` сломанная проверка диапазона | done | `src/data_validation.py` |
| 1.12 | `tests/test_simple.py` и `tests/test_setup.py` — не тесты | done | `tests/test_simple.py`, `tests/test_setup.py` |
| 1.13 | `kidney_displacement_prediction/` — пустая обёртка | done | `kidney_displacement_prediction/setup.py`, `README.md`, `__init__.py` |
| 2.1 | Неверные/неполные feature-схемы | done | `config/phase1_feature_schema.yaml`, `src/features/phase1_schema.py`, `models/phase1/adaptive_ensemble.py`, API, extractors |

---

## Записи по проблемам

### 1.4 — `train_lasso.py` / `train_ridge.py` падают на старте

Статус: done.

**`models/phase1/train_lasso.py`:**

- Добавлен импорт `from sklearn.impute import SimpleImputer` (отсутствовал при использовании в `__init__`).
- `analyze_feature_selection` теперь принимает `selected_feature_names` и выравнивает имена с длиной `model.coef_` (раньше индексировал полный `self.feature_names` против ужатой подматрицы → IndexError/неверные имена).
- В ветке `use_tuning=False` исправлен `for target_name in self.target_names` → `for i, target_name in enumerate(self.target_names)` (раньше `y_train_target = y_train[:, i]` падало `NameError`).
- `cross_val_score` теперь использует `X_train_selected` (соответствует обученной модели), а не полный `X_train`.
- В `train_and_evaluate` пробрасываем `selected_feature_names=selected_features` в `analyze_feature_selection`.

**`models/phase1/train_ridge.py`:**

- В ветке `use_tuning=False`: `for target_name in self.target_names` → `for i, target_name in enumerate(self.target_names)`.
- `cross_val_score` переключён на `X_train_selected`.
- Перенесён блок вычисления `nonzero_coef`/`top_features` **до** записи в `results[target_name]` и до `print` — раньше использовались переменные, которые ещё не были определены (`UnboundLocalError`).
- Убраны не-ASCII символы `±`/`²` из `print`-блока, заменены на ASCII (`+/-`, `R2`) — снижает риск падений на Windows-консолях.

**Верификация:**

- `py -3 -m py_compile` для обоих файлов — OK.
- Linter — без ошибок.

**Замечания (не фиксили в рамках 1.4):**

- `load_and_prepare_data` обоих скриптов читает `data/vybor_unified_features.csv` и `data/kits19_medical_grade_features.csv` — последнего файла нет в текущей копии репозитория, runtime-запуск этих скриптов даст `FileNotFoundError`. Это отдельная инфраструктурная проблема, не входит в скоуп 1.1–1.8.
- Оба скрипта по-прежнему делают `train_test_split` после `imputer.fit_transform` поверх объединённых датасетов (leakage). Это входит в 1.6, исправим позже.

---

### 1.7 — FastAPI возвращает 500 на клиентские ошибки

Статус: done.

**`src/api/kidney_displacement_api.py`:**

- `predict_displacement` теперь корректно разделяет коды:
  - `503` — если модель не загружена (`model_data is None or feature_names is None`).
  - `400` — если входные данные не валидны или отсутствуют признаки.
  - `500` — только для непредвиденных серверных ошибок.
- Раньше `HTTPException(400, ...)` ловился собственным `except Exception` и переоборачивался в 500 → теперь явный `except HTTPException: raise` стоит до общего обработчика.
- `/predict` упрощён: внутренний try ловит `HTTPException` и пробрасывает; остальное → 500.
- `/predict_batch`:
  - 400 если список пустой;
  - 503 если модели нет;
  - `success` в теле ответа теперь честный: `True` только если ВСЕ пациенты обработаны;
  - per-item ошибка сохраняет `status_code` (`400`/`500`/...);
  - убран ошибочный `success: True` при полном падении батча.
- `metadata: Dict[str, any]` → `Dict[str, Any]` (раньше Pydantic v2 не валидировал `any` как тип, ломал OpenAPI).
- `patient_data.dict()` → `patient_data.model_dump()` (с fallback на старый `.dict()`) — снимает DeprecationWarning Pydantic v2.

**Верификация:** `py -3 -m py_compile`, линтер — OK.

**Не входит в скоуп 1.7 (но рядом):**

- Конфликт схемы `BatchPredictRequest.patients: List[Dict[str, PatientData]]` и фактической обработки — мы оставили обработчик максимально терпимым к обоим форматам, но финальное приведение схемы попадёт в 1.8.
- Mock-up performance numbers в `/model_info` и сам факт того, что startup-event делает `load_model` — это High-2.10, не критично.

---

### 1.3 — Shared sklearn-estimators между таргетами

Статус: done.

**Контекст:** В `adaptive_ensemble.py` и `target_specific_ensemble.py` в цикле по 6 таргетам каждый раз создавались `VotingRegressor` со ссылками на одни и те же `base_models[name]` экземпляры. Параллельно, `create_best_single_model` возвращал `models[best_model_name]` напрямую, и `.fit()` затем мутировал тот же объект. Формально `VotingRegressor.fit` делает `clone()` внутри, но (а) логически грязно и (б) `create_best_single_model` мутировал реальный экземпляр между таргетами.

**`models/phase1/adaptive_ensemble.py`:**

- Импорт `from sklearn.base import clone`.
- `create_optimized_voting_ensemble`, `create_adaptive_voting_ensemble`, `create_standard_voting_ensemble` теперь оборачивают `models[name]` в `clone(models[name])` при формировании списка estimators.
- `evaluate_model_cv` теперь подаёт `clone(model)` в `cross_val_score` — даже если sklearn в будущем перестанет клонировать сам, наша вызывающая сторона защищена.
- `_copy_model` переписан на `sklearn.base.clone` (с fallback на старую логику для нестандартных моделей).
- В `print` MAE символ `±` заменён на `+/-` — снижаем риск кодировки на Windows-консолях.

**`models/phase1/target_specific_ensemble.py`:**

- Импорт `from sklearn.base import clone`.
- `create_target_specific_ensemble` — estimators строятся из `clone(models[name])`.
- `create_best_single_model` — теперь возвращает `clone(models[best_model_name])`. Раньше при `.fit()` мутировался shared экземпляр, и при формировании ensemble для следующего таргета использовался уже-обученный (с другой целью) объект.
- `evaluate_model_cv` — `cross_val_score(clone(model), ...)`.
- `±` → `+/-` в print MAE.

**Верификация:** `py_compile` обоих файлов — OK. Линтер — без ошибок.

**Не входит в скоуп 1.3 (но рядом):**

- `_optimized_weights` всё ещё не сохраняется в pickle через `save_model` — это пункт 2.12, отдельная задача.
- Целевой инвариант для будущих PR: `id(model)` всех вложенных estimators не должен совпадать с `id(base_models[name])`. Это можно добавить как assertion в unit-тест.

---

### 1.5 — Синтетические признаки/таргеты в инференс-пайплайнах

Статус: done.

**`scripts/inference/enhanced_ct_extractor.py`:**

- В `_extract_kidney_coordinates_lightweight` удалены захардкоженные значения `kidney_left_delta_x = 12.5`, `kidney_left_delta_y = 4.2`, `kidney_left_delta_z = 8.1`, `kidney_right_delta_x = -8.3`, `kidney_right_delta_y = 3.8`, `kidney_right_delta_z = 7.9`.
- Вместо них возвращается `float('nan')` с пояснением: смещения почек — это **целевые переменные**, их нельзя «измерить» из одной CT-серии; они должны приходить из парных сканов или из разметки. Это устраняет основной источник отравления датасета.
- Эта же замена выполнена в дублирующем dead-code блоке после первого `return result` (raw `replace_all` = 2 occurrences). Сам dead-code в этой задаче я не удалял — это отдельная чистка.

**`scripts/inference/dicom_feature_extractor.py` и `scripts/inference/enhanced_ct_extractor.py`:**

- В обоих экстракторах все 6 случаев `idx = np.random.randint(len(<coords>[0]))` заменены на детерминированный `idx = len(<coords>[0]) // 2` (медианный индекс в наборе кандидатов на конкретном Z-срезе).
- Этим устранена основная причина невоспроизводимости признаков run-to-run (раньше каждое извлечение давало разные точки upper/middle/lower на одних и тех же DICOM).

**`src/models/data_integration_fix.py`:**

- В `DataIntegrationFix.__init__` добавлен seeded RNG (`np.random.default_rng(seed)`), seed по умолчанию `42`, опционально переопределяется через аргумент конструктора `synthetic_volume_seed`.
- `np.random.normal(0, 10, len(df))` заменён на `self._rng.normal(0, 10, n)` — синтетические объёмы почек теперь воспроизводимы.
- Добавлен явный комментарий, что этот блок — fallback и не должен использоваться для боевой модели.

**Верификация:** `py_compile` всех трёх файлов — OK. Линтер — без ошибок.

**Замечания (не входит в скоуп 1.5):**

- В `enhanced_ct_extractor.py` остался ~80-строчный dead-code после `return result` в `_extract_kidney_coordinates_lightweight` (двойная реализация). Это Medium-удаление, нужно делать отдельно с проверкой, что нигде не вызывается.
- Mapping `sex` (M=1/F=2 в одном экстракторе, M=1/F=0 в других) и body_type — отдельная проблема и фиксить нужно в рамках 1.8 (унификация контрактов).

---

### 1.6 — Train/val leakage

Статус: done.

**`models/phase1/adaptive_ensemble.py`:**

- Выделен `_build_feature_matrix(df)` — общая логика feature engineering (используется и в legacy, и в leakage-safe пути).
- Добавлен новый метод `prepare_training_data_split(train_df, val_df)`:
  - feature engineering применяется к train и val **раздельно**;
  - `StandardScaler.fit` теперь происходит **только на train**, затем `transform` к val;
  - есть защита от расходящегося набора признаков между train и val (берутся общие колонки + warning);
  - есть warning при различиях в наборе target-колонок.
- Старый `prepare_training_data(df)` оставлен, но помечен `DeprecationWarning` и предупреждает: при склейке train+val он создаёт утечку статистики через `train_test_split`.
- `main()` переключён на `prepare_training_data_split(train_df, val_df)`. Раньше: `combined_df = concat(train_df, val_df) → train_test_split → scaler.fit`. Теперь: `train_df → fit_transform`, `val_df → transform`. Это устраняет основной leakage в обучающем пайплайне Phase 1.

**`scripts/inference/convert_single_file.py`:**

- `clean_final_data` больше не делает медианную импутацию признаков по всему датасету.
- Добавлен новый метод `impute_features_after_split(train_df, val_df, test_df, feature_cols)` — обучает медиану ТОЛЬКО на `train_df` и применяет её к val/test.
- В `convert()` добавлен шаг 8a: импутация выполняется ПОСЛЕ `split_data`, что устраняет утечку статистики val/test в train на этапе подготовки CSV.

**Верификация:** `py_compile` обоих файлов — OK. Линтер — без ошибок.

**Замечания (не входит в скоуп 1.6):**

- `imputer` в `AdaptiveEnsembleTrainer.__init__` всё ещё не вызывается в новом пути (`prepare_training_data_split`). Это часть 1.1 — добавим `imputer.fit_transform` на train и `imputer.transform` на val в пункте 1.1, чтобы дополнить scaler.
- `train_lasso.py`/`train_ridge.py` всё ещё делают `imputer.fit_transform` на объединённом датасете — этим скриптам нужен такой же refactor, но они вне production-пути phase1, поэтому отложил.
- `data/processed/train.csv` и `data/processed/validation.csv` нужно сгенерировать через починенный `convert_single_file.py` или эквивалент. Без этого `load_integrated_data()` упадёт `FileNotFoundError` — это инфраструктурная задача.

---

### 1.1 — Несогласованность feature pipeline между train и inference

Статус: done.

**Симптом до фикса:** в трейн-времени `imputer` инициализировался в `__init__` и сохранялся в pkl, но в `prepare_training_data` фактически НЕ вызывался — преобразование было `scaler.fit_transform(X)` поверх данных, в которых NaN могли быть после feature engineering (`/`, `divide_by_zero`, отсутствие части базовых полей у нового пациента). В то же время `src/api/kidney_displacement_api.py` загружал из pkl только `scaler`, ничего не зная про `imputer`, и для отсутствующих признаков валился с `KeyError`. Это означает, что train-time и inference-time pipelines объективно расходились, и любой `NaN` после `_create_cross_features` (например `kidney_left_to_spine_ratio = left_to_spine / body_width` при `body_width=0`) шёл напрямую в `scaler.transform → NaN → model.predict(NaN)`.

**`models/phase1/adaptive_ensemble.py`:**

- В `_build_feature_matrix` X и y приведены к `astype(float)` — гарантия, что pandas object/Int64-колонки не пройдут в scaler.
- Добавлен метод `build_inference_matrix(df)` для инференса: применяет `_create_engineered_features` + `_create_cross_features`, добавляет недостающие train-time колонки как `NaN`, возвращает ndarray в строгом порядке `self.feature_names`. Это та же самая логика инжиниринга, что используется в обучении — гарантия, что новые точки попадают в модель в том же признаковом пространстве.
- В `prepare_training_data` (legacy): добавлен `imputer.fit_transform(X_train) → imputer.transform(X_test) → scaler.fit_transform → scaler.transform`. Раньше `imputer` создавался, но никогда не вызывался.
- В `prepare_training_data_split`: тот же фикс — `imputer.fit` ТОЛЬКО на train, `imputer.transform` на val. Это и leakage-safe, и согласовано с inference path.

**`src/api/kidney_displacement_api.py`:**

- `predict_displacement` теперь точно повторяет pipeline:
  1. `create_features(patient_data)` → DataFrame с engineered + cross features.
  2. Reindex по `feature_names` (отсутствующие колонки добавляются как `NaN`).
  3. `imputer.transform(X)` — если в pkl есть `imputer` (новые модели). Если его нет (legacy pkl) — логируем единичный warning и идём дальше.
  4. `scaler.transform(X)` — как и раньше.
  5. После шага 4 явно проверяем, что в `X_scaled` нет `NaN`; при наличии — возвращаем 400 с понятным сообщением (раньше уходило в 500).
  6. `model.predict(X_scaled)` по каждому таргету.
- Документация HTTP-кодов в docstring обновлена.

**Верификация:** `py_compile` обоих файлов — OK. Линтер — без ошибок.

**Совместимость:**

- Старые pkl, сохранённые ДО этой правки (с `imputer` ключом, но без вызова `imputer.fit_transform` в тренировке), будут содержать «нетренированный» imputer. `imputer.transform` на нетренированном объекте даст `NotFittedError`. Поэтому при первом прогоне любого старого pkl нужно либо ре-обучить модель, либо использовать обходной путь через `scripts/validation/common.py` fallback (тренировка на лету), который мы добавили ранее. Это явно описано в логе и в коде.
- В новых тренировках `imputer` фитится правильно, и API будет использовать его прозрачно.

**Не входит в скоуп 1.1:**

- `train_lasso.py` / `train_ridge.py` всё ещё имеют свою отдельную pipeline (где imputer применяется на объединённом датасете до сплита). Их рефакторинг под общий контракт — задача более широкая, попадёт в продолжение 1.6/1.1.
- Точная инверсия StandardScaler для confidence interval (то есть калибровка confidence) не входит в этот фикс.

---

### 1.2 — `KidneyDisplacementPredictor.predict` использует неинициализированный ансамбль

Статус: done.

**Проблема:** В `predict()` для каждого таргета вызывалось `self.ensemble_trainer.create_adaptive_voting_ensemble(load_base_models(), target_name)` — это создавало **новый, необученный** `VotingRegressor`. Затем сразу `adaptive_ensemble.predict(X_scaled)` падал бы `NotFittedError`. Параллельно `train()` вызывал `self.ensemble_trainer.load_and_prepare_data()` — метода, которого в `AdaptiveEnsembleTrainer` не существует (NameError на первом обращении).

**`models/phase1/kidney_displacement_predictor.py`:**

- `predict()` полностью переписан:
  - Использует уже обученные модели из `ensemble_trainer.trained_models[target_name]` (они кладутся туда в `train_and_evaluate_adaptive_ensembles`). Если словарь пустой — выбрасывает читаемую `ValueError`, а не уходит в `NotFittedError`.
  - Перед моделью применяет **тот же** train-time pipeline: `build_inference_matrix(df)` (alignment по `feature_names`) → `imputer.transform` (если imputer фитан) → `scaler.transform`. Это закрывает 1.1 и для этого высокоуровневого API одновременно.
  - Валидация входа теперь требует только базовые `required_features` тренера (engineered/cross считаются внутри), что снимает ложноположительные `ValueError`.
- `train()` теперь использует `load_integrated_data()` + `prepare_training_data_split(train_df, val_df)` — устраняет вызов несуществующего `load_and_prepare_data()` и одновременно поднимает leakage-safe путь из 1.6.
- `load_model()` теперь проверяет, что у загруженного `ensemble_trainer` есть непустой `trained_models`. Если нет — `is_trained = False` и `predict()` не упадёт неожиданно, а сразу сообщит понятную причину.

**Верификация:** `py_compile` файла — OK. Линтер — без ошибок.

**Не входит в скоуп 1.2:**

- `validate_input` всё ещё считает NaN ошибкой. После 1.1 NaN должен быть допустимым (imputer обработает), но я оставил строгую валидацию как защитный механизм — её ослабление сделаю отдельно, если будет нужно.
- `save_model` всё ещё pickle-ит `self.ensemble_trainer` объектом (а не `joblib.dump(model_data dict)` как делает `AdaptiveEnsembleTrainer.save_model`) — это рассинхрон двух форматов, попадёт в follow-up.

---

### 1.8 — Несовместимые API контракты для одной задачи

Статус: done.

**Противоречие (зафиксировано, не сливали в один сервис):**

В проекте было три HTTP-слоя с разными контрактами:

| Файл | Стек | Назначение |
|------|------|------------|
| `src/api/kidney_displacement_api.py` | FastAPI | Предсказание смещения почек (phase1 ensemble) |
| `models/phase1/api_kidney_predictor.py` | Flask | То же по смыслу, но другой набор полей (30 признаков с `_norm`) |
| `src/api/api_server.py` | FastAPI | AR-навигация + сенсоры (другой домен: `age`, `bmi`, AR-матрицы) |

**Решение (по индустриальной практике ML-сервинга 2026):**

- **Canonical predict-API:** `src/api/kidney_displacement_api.py` (FastAPI + Pydantic + Uvicorn).
- **Flask legacy:** `models/phase1/api_kidney_predictor.py` помечен `[DEPRECATED]` + `DeprecationWarning` при импорте; новые интеграции не должны его использовать.
- **AR API:** `src/api/api_server.py` **не трогали** — это отдельный сервис (AR/сенсоры), не дублирует displacement predict.

**Изменения в canonical FastAPI:**

- Добавлены Pydantic-модели `BatchPatientEntry` и исправлен `BatchPredictRequest.patients: List[BatchPatientEntry]` (раньше `List[Dict[str, PatientData]]` не валидировался Pydantic v2 корректно).
- Обработчик `/predict_batch` переписан под typed-модели (`entry.patient_data`, `entry.patient_id`) вместо dict-хаков.
- Корневой `/` теперь явно указывает `service_role`, canonical endpoints и ссылки на deprecated/related сервисы.

**Верификация:** `py_compile` FastAPI + Flask legacy — OK. Линтер — без ошибок.

**Follow-up (не блокирует 1.8):**

- Полное удаление Flask-сервера — только после миграции всех клиентов.
- Унификация `sex_encoded` / `patient_position_encoded` между extractors — отдельная задача (упомянута в 1.5).

---

### 1.9 — Тихая работа AR-системы без модели

Статус: done.

**Наблюдение из аудита:** `src/ar_system/kidney_ar_system.py` при отсутствии модели возвращал синтетический вектор смещения `[5.0, -3.0, 2.0, 5.0, -3.0, 2.0]` и `confidence=0.7`, что визуально выглядело как «успешное» AR-предсказание.

**Проверка через @Browser (внешний референс):**

- Для медицинского ПО применяется fail-safe/fail-closed подход: при недоступности критичного компонента (модель) система должна переходить в безопасное состояние и явно сигнализировать отказ, а не выдавать правдоподобные synthetic значения.
- По результатам просмотра (`DuckDuckGo` выдача по IEC 62304/risk-control/safe-state) выбрано поведение «явный отказ», а не «тихий fallback».

**Внесённые изменения в `src/ar_system/kidney_ar_system.py`:**

- Добавлен флаг готовности `self.model_ready`, который выставляется только при успешной загрузке обученных моделей.
- В `predict_kidney_displacement` добавлена ранняя fail-safe проверка:
  - если модель не загружена, сразу возвращается `_create_error_response(...)` с `success=False`.
- В `_predict_displacement` удалён synthetic fallback; теперь при пустом `self.models` выбрасывается `RuntimeError`.
- В `_estimate_confidence` default confidence изменён с `0.7` на `0.0`, если `confidence_estimator` недоступен.
- В `_apply_constraints_and_fallback` удалён synthetic fallback при некорректной размерности предсказания; теперь выбрасывается `ValueError`.

**Итог поведения после фикса:**

- Без модели AR-предикт больше не «притворяется валидным».
- API слоя AR получает явный failure-сигнал и может корректно остановить отображение навигации.

---

### 1.10 — `FallbackHandler` путает позицию и смещение

Статус: done.

**Проблема на уровне кода:**

`KidneyARSystem._apply_constraints_and_fallback` передаёт в `FallbackHandler.handle_prediction` вектор `ml_prediction` как **дельту смещения** (мм). Но внутри handler вызывалось:

```python
self.constraints.apply_constraints(original_position, ml_prediction)
```

А `AnatomicalConstraints.apply_constraints` ожидает второй аргумент как **абсолютную позицию** и сам вычисляет `displacement = predicted_pos - original_pos`. В итоге при дельте `+5 мм` получалось `displacement = delta - original_pos` (сотни мм) — геометрически неверно.

Дополнительно `handle_prediction` возвращал **абсолютную позицию**, а вызывающий код трактовал результат как дельту для `apply_displacement`.

**Решение (`src/reliability/confidence_constraints.py`):**

- Добавлен метод `AnatomicalConstraints.apply_constraints_from_displacement(original_pos, displacement)` — принимает дельту, ограничивает её, проверяет spine/body на целевой позиции `original + delta`, возвращает **скорректированную дельту**.
- `FallbackHandler.handle_prediction` переведён на новый метод; docstring явно фиксирует семантику «ml_prediction = displacement».
- Старый `apply_constraints(original, predicted_pos)` сохранён для сценариев с абсолютными координатами.

**Верификация:** `tests/test_simple.py::test_fallback_handler_treats_ml_output_as_displacement`.

---

### 1.11 — Сломанная проверка диапазона в `validate_processed_data`

Статус: done.

**Проблема на уровне кода:**

В `src/data_validation.py` (строки ~328–329) вместо булевой маски использовалось:

```python
out_of_range = np.abs(target_values[mask_high]) + np.abs(target_values[mask_low])
```

Это **сумма абсолютных значений** выбросов, а не список/количество out-of-range точек. Условие `if len(out_of_range) > 0` срабатывало некорректно, предупреждения о клинически невалидных смещениях фактически не работали.

**Решение:**

```python
out_of_range_mask = (target_values < low) | (target_values > high)
out_of_range_count = int(np.sum(out_of_range_mask))
```

В `ValidationResult.value` теперь попадают реальные out-of-range значения (`target_values[out_of_range_mask].tolist()`).

**Верификация:** `tests/test_simple.py::test_validate_processed_data_flags_out_of_range_deltas`.

---

### 1.12 — Псевдо-тесты без регрессий

Статус: done.

**Проблема:**

- `tests/test_simple.py` — исполняемый скрипт с `print` и emoji, без `assert`; всегда «проходил» визуально.
- `tests/test_setup.py` — только импорты и комментарий «rest unchanged», без проверок.

**Решение:**

- `tests/test_setup.py` переписан как pytest-модуль с параметризованным `test_critical_module_imports`.
- `tests/test_simple.py` переписан как pytest-модуль с регрессиями:
  - fail-closed AR без модели (1.9),
  - корректная семантика displacement в FallbackHandler (1.10),
  - out-of-range delta validation (1.11).

**Верификация:** `pytest tests/test_setup.py tests/test_simple.py`.

---

### 1.13 — Пустая обёртка `kidney_displacement_prediction/`

Статус: done.

**Проблема:**

- `setup.py` делал `open("README.md")`, но файла не было → `pip install` падал с `FileNotFoundError`.
- Пакет содержал только пустые `__init__.py` без полезного API.

**Решение:**

- Добавлен `kidney_displacement_prediction/README.md` с описанием назначения wrapper-пакета.
- `setup.py` переписан: `Path(__file__).parent / "README.md"`, `include_package_data=True`, `package_data` для `config/*.yaml`, `MANIFEST.in`.
- В `kidney_displacement_prediction/__init__.py` добавлена `get_config_dir()` для доступа к YAML-конфигам после установки.

**Ограничение (не блокирует 1.13):** основной код по-прежнему в корневом `src/` репозитория; полная консолидация в один pip-пакет — отдельная архитектурная задача.

---

### 2.1 — Неверные/неполные feature-схемы

Статус: done.

**Проблема на уровне кода:**

В проекте одновременно существовало **5 разных «истин»** по признакам:

| Источник | Что ожидал | Проблема |
|----------|------------|----------|
| `AdaptiveEnsembleTrainer` | 23 base + 13 engineered + 15 cross | Эталон обучения |
| `KidneyDisplacementPredictor` / Flask API | 30 полей с `*_center_*_norm` | Norm-координаты **не входят** в train-пайплайн |
| `feature_config.yaml` | rel + norm + `patient_position_encoded` в required | Дублирование и путаница с engineered |
| `scripts/validation/common.py` | Урезанный список без `length_mm` | Fallback RF не совпадал с production |
| DICOM extractors | `body_com_x_mm`, `kidney_left_vs_spine_x`, `*_upper_x` | Имена не совпадали с canonical |

Итог: scaler/imputer обучались на одном наборе колонок, а inference/API/extractors подавали другой → silent mismatch перед переобучением.

**Решение:**

1. **`config/phase1_feature_schema.yaml`** — декларативный контракт (base / engineered / cross / targets).
2. **`src/features/phase1_schema.py`** — единый Python-модуль:
   - константы `BASE_FEATURES`, `ENGINEERED_FEATURES`, `CROSS_FEATURES`, `TARGET_NAMES`;
   - `normalize_dataframe()` / `normalize_record()` — алиасы (`body_com_x_mm`→`body_com_x`, `kidney_left_vs_spine_x`→`kidney_left_center_x_rel`, …) и вычисление недостающих distance/length;
   - `encode_patient_position()` для `scan_position` / DICOM `PatientPosition`.
3. **Интеграция:** `adaptive_ensemble.py` импортирует схему и вызывает `normalize_dataframe` в `_build_feature_matrix` / `build_inference_matrix`; predictor, FastAPI, Flask legacy, validation, data integration, `enhanced_ct_extractor` переведены на canonical base.
4. **Убраны `*_norm` из required** в predictor / `feature_config.yaml` — они остаются опциональными колонками в Vybor/KiTS19 CSV, но не являются входом модели.

**Верификация:** `pytest tests/test_feature_schema.py` (4 теста) + существующие `test_setup` / `test_simple` — 12 passed.
