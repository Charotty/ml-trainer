# Code Audit Report — ML Trainer (Kidney Displacement Predictor)

Дата: 27 мая 2026.
Аудит проведён без модификации исходников: только чтение, статический анализ и трасировка зависимостей.
Скоуп: `src/`, `models/`, `scripts/`, `tests/`, корневые `test_*.py`, конфиги, `kidney_displacement_prediction/`.

> Цель отчёта: дать честную картину состояния проекта по логике, чистоте, стилю, функциональной корректности и потенциальным утечкам/ошибкам. Никаких правок не вносится.

---

## 0. Краткое резюме

**Общая оценка качества кода: ниже среднего для production-системы.**

- Архитектурно проект разбит правильно (data → features → model → API → AR), но в репозитории **сосуществуют параллельные несовместимые реализации** одних и тех же абстракций (валидаторы, метрики, координаты, ансамбли, API).
- В коде есть **критические функциональные баги**, при которых система формально "работает", но клинические числа могут быть неверны (несогласованность train/inference, несовместимые feature contracts между API и моделью, неинициализированные ансамбли, hardcoded синтетические дельты).
- Стилево — много **технического долга**: emojis в production, RU/EN миксы, `sys.path.append`-хаки, отсутствие линтера/форматтера в CI, отсутствие `pyproject.toml`.
- Безопасность: повсеместный `joblib.load`/`pickle.load` без верификации, CORS `*` + credentials, утечка stacktrace в JSON ответах.
- Тесты: значительная часть `test_*.py` — это **скрипты с `print`**, а не реальные регрессии (нет `assert`, нет фикстур, нет mock'ов).

| Слой | Critical | High | Medium | Low |
|------|---------:|-----:|-------:|----:|
| API | 4 | 7 | 4 | 7 |
| models/phase1 + src/models | 7 | 9 | 11 | 9 |
| scripts (training+inference) | 2 | 6 | 7 | 4 |
| scripts/validation (новые) | 0 | 4 | 6 | 4 |
| src/* utility | 5 | 8 | 9 | 6 |
| tests/ | 2 | 4 | 3 | 3 |
| kidney_displacement_prediction/ | 2 | 2 | 2 | 2 |
| Стиль/конфиги | 0 | 2 | 3 | 4 |
| **Итого** | **22** | **42** | **45** | **39** |

---

## 1. Критические проблемы (Critical)

Проблемы, при которых поведение системы может быть клинически или операционно неверным.

### 1.1. Несогласованность feature-pipeline между обучением и инференсом

- В `src/api/kidney_displacement_api.py` инференс **создаёт engineered/cross features через свежий `AdaptiveEnsembleTrainer()`**, но не использует сохранённый `imputer` (его и в обучении нет в `adaptive_ensemble.prepare_training_data`).
- При этом scaler был обучен на матрице без явного импьютинга → если на входе появится `NaN` или другой порядок колонок, scaler даст некорректные значения, а модель — некорректный прогноз.

```83:106:src/api/kidney_displacement_api.py
model_data = joblib.load(model_path)
...
# Создаем инженерные и cross-features
test_df_with_features = trainer._create_engineered_features(test_df)
test_df_with_features = trainer._create_cross_features(test_df_with_features)
...
X_test = test_df_with_features[feature_names].values
X_test_scaled = model_data['scaler'].transform(X_test)
```

### 1.2. `KidneyDisplacementPredictor.predict` использует **неинициализированный ансамбль**

В `models/phase1/kidney_displacement_predictor.py` (строки ~194–206) на каждом вызове создаётся `VotingRegressor` и сразу вызывается `predict` без `fit` → `NotFittedError` или непредсказуемое поведение.

### 1.3. Поделённые между таргетами sklearn-эстимторы

В `models/phase1/adaptive_ensemble.py`:

- `evaluate_model_cv` фитит `base_models[model_name]` **in place** и переиспользует те же экземпляры для других таргетов в `VotingRegressor` → каждая следующая ось координаты переобучается на предыдущую.
- Тот же паттерн в `models/phase1/target_specific_ensemble.py` (~L290–L310).

### 1.4. `train_lasso.py` / `train_ridge.py` падают при первом запуске

- `train_lasso.py` использует `SimpleImputer`, но **не импортирует** его (`NameError`).
- `train_ridge.py` в `train_and_evaluate` записывает `'NonZero_Coefficients': nonzero_coef` до того, как переменная определена → `UnboundLocalError`.

### 1.5. Синтетические признаки и таргеты в инференс-пайплайнах

- `scripts/inference/enhanced_ct_extractor.py` (`_extract_kidney_coordinates_lightweight`, ~L628–636): **hardcoded** значения `kidney_*_delta_x/y/z = 12.5, 4.2, 8.1, ...` подмешиваются к признакам, извлекаемым из DICOM. Если этот извлекатель когда-либо используется для обучения или для построения "новых" датасетов — данные физически некорректны.
- `src/models/data_integration_fix.py`: `np.random.normal` создаёт **синтетические** kidney_volume для DICOMS-источника без контроля seed.
- `dicom_feature_extractor.py` / `enhanced_ct_extractor.py`: выбор точек почки через `np.random.randint(...)` **без seed** → невоспроизводимость от запуска к запуску.

### 1.6. Train/val leakage

- `models/phase1/adaptive_ensemble.py` (`load_integrated_data`): склеивает train+validation в один dataframe (`combined_df`), затем `prepare_training_data` делает `train_test_split` поверх объединения → валидационные строки попадают в "новый train" и в обучение scaler.
- `scripts/inference/convert_single_file.py` (`clean_final_data` → `split_data`): медианная импутация выполняется **до** разбиения — статистика теста и валидации утекает в train.

### 1.7. API возвращает HTTP 500 на клиентские ошибки

В `src/api/kidney_displacement_api.py` `HTTPException(400, ...)` ловится широким `except Exception` и переоборачивается в `HTTPException(500, ...)` → клиент не отличает свою ошибку от падения сервиса.

### 1.8. Несовместимые API контракты для одной задачи

Три параллельных API:

| API | Файл | Порт | Признаков | Формат запроса |
|-----|------|-----:|---------:|----------------|
| FastAPI displacement | `src/api/kidney_displacement_api.py` | 8000 | ~23 | `{"patient_data": {...}}` |
| FastAPI AR-стек | `src/api/api_server.py` | разн. | через `DataValidator` | flat fields |
| Flask phase1 | `models/phase1/api_kidney_predictor.py` | 5000 | ~30 + `*_norm` | flat fields |

Тесты из `test_api.py` и `test_api_endpoints.py` подтверждают это — они шлют **взаимоисключающие** payload'ы (см. 5.2).

### 1.9. Тихая работа AR-системы без модели

`src/ar_system/kidney_ar_system.py` (L244–246, L264): если `joblib`-моделей нет — возвращает захардкоженный вектор `[5.0, -3.0, 2.0, 5.0, -3.0, 2.0]` с `success: True` и `confidence: 0.7`. Это особенно опасно в **AR навигации**.

### 1.10. `FallbackHandler` путает позицию и смещение

`src/reliability/confidence_constraints.py::apply_constraints` ожидает `predicted_pos` как абсолютную позицию, а `KidneyARSystem` передаёт ему **дельту**. Внутри делается `displacement = predicted_pos - original_pos` → результат геометрически некорректен.

### 1.11. `DataValidator.validate_processed_data` сломанная проверка диапазона

`src/data_validation.py` (L328–329):

```
out_of_range = np.abs(target_values[...]) + np.abs(target_values[...])
```

Это **сумма абсолютов**, а не булева маска — предупреждения о выходе за диапазон по сути не работают.

### 1.12. `tests/test_simple.py` и `tests/test_setup.py` — не тесты

- `test_simple.py`: скрипт с `print` и emoji, без `assert` — всегда "проходит".
- `test_setup.py`: только импорты и комментарий "rest unchanged".

Реальные регрессии не покрыты.

### 1.13. `kidney_displacement_prediction/` — пустая обёртка

`setup.py` читает несуществующий `README.md` → `pip install` упадёт.
`src/{models,api,utils}/` содержат только пустые `__init__.py`. То есть **производственный пакет фактически нечем установить**.

---

## 2. Серьёзные проблемы (High)

### 2.1. Неверные/неполные feature-схемы

- `models/phase1/kidney_displacement_predictor.py` декларирует 30 базовых/норм признаков, а `AdaptiveEnsembleTrainer.prepare_training_data` обучает на 51+ engineered/cross признаках → scaler-размерности не совпадут.
- `models/phase1/api_kidney_predictor.py` требует `*_norm` признаки, которых нет в FastAPI-схеме.
- `scripts/inference/extract_from_dicom.py` создаёт `X_upper_right`, `Y_middle_left` и т.п. — это **не** имена признаков, которые ожидает обучение.

### 2.2. `train()` зовёт несуществующий API

`KidneyDisplacementPredictor.train` (L115) вызывает `self.ensemble_trainer.load_and_prepare_data()`, тогда как `AdaptiveEnsembleTrainer` имеет только `load_integrated_data` → `AttributeError`.

### 2.3. Несоответствия в request-схемах FastAPI

- `BatchPredictRequest.patients: List[Dict[str, PatientData]]` в Pydantic v2 не валидируется так, как используется в коде (`patient_data["patient_data"]`, `patient_data.get("patient_id")`).
- `metadata: Dict[str, any]` — `any` (нижний регистр) **невалидный** Pydantic-тип; требуется `Any`.

### 2.4. CORS `*` + `allow_credentials=True`

`src/api/api_server.py`: небезопасная комбинация для браузерных клиентов.

### 2.5. Утечка трейсбэка в response body

`models/phase1/api_kidney_predictor.py` (L327–335): при ошибке отдаёт `details: str(e)` клиенту даже при `debug=False`.

### 2.6. Visualization fallback скрывает реальное поведение модели

`scripts/validation/common.py::build_or_load_predictor`: при любой ошибке `joblib.load` (включая mismatch версии sklearn) **молча** обучает on-the-fly RandomForest и помечает режим `fallback_random_forest`. Метрики и визуализации потом отражают не production-модель.

### 2.7. Несогласованность визуализаций между тремя режимами

`scripts/validation/run_visual_tests.py`:

| Режим | Позвоночник | Sagittal | Подписи осей | Легенда |
|-------|:-----------:|:--------:|:-------------:|:-------:|
| single_case_3d | yes | no | full | yes |
| multi_panel_2d3d | yes | no (3D) | сокращ. | no |
| overlay_supine_vs_predicted | **no** | yes | full | yes |

Из-за этого визуальный quality-gate не симметричен.

### 2.8. `np.sign(0) == 0` ломает quality-check

`scripts/validation/run_visual_tests.py::quality_checks`: проверка `left_right_x_have_opposite_sign` ложно срабатывает, когда любая ось ровно 0.

### 2.9. Pickle и joblib без верификации повсеместно

Все API/предикторы делают `joblib.load` / `pickle.load` без проверки хэшей и схемы. При компрометации файла модели — RCE.

### 2.10. Утечка ресурсов

- `enhanced_ct_extractor.py` создаёт `tempfile.mkdtemp()`, не чистит.
- `dicom_feature_extractor.py` пишет `segmentation_output/` внутрь пациентских DICOM-папок.
- `models/phase1/api_kidney_predictor.py`: `logging.FileHandler('api.log')` относительный путь, нет ротации.
- FastAPI: `@app.on_event("startup")` устаревший, `shutdown` отсутствует, глобалы `model_data`, `trainer` не освобождаются.

### 2.11. Дублирующиеся пакеты и классы

| Сущность | Файлы |
|----------|-------|
| `coordinate_system` | `src/coordinate_system.py` + `src/coordinate_system/` (package) — **конфликт имён** |
| `DataValidator` | `src/data_validation.py`, `src/validation/data_validator.py` |
| `ClinicalMetrics` | `src/metrics/clinical_metrics.py`, `src/validation/data_validator.py` |
| `SystemLogger` | `src/system_logging/system_logger.py`, `src/validation/data_validator.py` |
| `AnatomicalConstraints` | `src/constraints.py`, `src/reliability/confidence_constraints.py` |
| `ConfidenceEstimator` | `src/confidence_scoring.py`, `src/reliability/confidence_constraints.py` |
| KiTS19 loader | `src/data/kits19_loader.py`, `KiTS19Loader` в `unified_loader.py` |
| Unpaired data | `src/unpaired_data.py`, `src/unpaired/unpaired_trainer.py` |
| Relative coords | `src/relative_coordinates.py`, `src/features/advanced_features.py`, `patient_coords.transform_features_to_patient` |
| requirements | `requirements.txt` (корень) vs `kidney_displacement_prediction/requirements.txt` |

### 2.12. Mock-up production-метрик в API

`/model_info` возвращает hardcoded `average_mae_mm: 2.140`, `r2_avg: 0.139` и т.п., не сверяясь с реальными значениями модели — это вводит клиента в заблуждение.

---

## 3. Средние проблемы (Medium)

- **Division без guard**: `models/phase1/adaptive_ensemble.py` и `src/features/advanced_features.py` делят на `body_*`, `kidney_*_length_mm` без проверки на 0 → `inf`/`NaN`.
- **Scoring на разном feature-space**: `train_lasso.py`/`train_ridge.py` гонят CV на полном scaled X, а модель тренируется на `X_train_selected`.
- **`use_tuning=False`** ветка обоих скриптов: `for target_name ... y_train_target = y_train[:, i]` — `i` не определён → `NameError`.
- **`patient_position_encoded` всегда 1** (`adaptive_ensemble.py` L295–299).
- **`kidney_separation_angle` считается в 2D**, хотя точки 3D.
- **`save_results` инверсия знака** в `adaptive_ensemble.py` (L912 vs L867).
- **`processing_time` в AR всегда 0.0**.
- **Пути по CWD**: `compare_all_ensembles.py`, `train_all_models.py`, `convert_single_file.py`, `test_all_models.py` ломаются вне корня репо.
- **Метрики `within_5mm`/`within_10mm`** в `evaluate_metrics.py` считаются поэлементно по 6 целям, а не по евклидовой норме вектора смещения.
- **`smoke_check.py`** требует `fastapi` (для offline ML).
- **`smoke_check.py`** возвращает 0 при отсутствующей модели → не блокирует pipeline.
- **`evaluate_metrics.py::worst_cases.csv`** пишется с `index=True` → дублирующая колонка.
- **`run_visual_tests.py`** выбирает кейсы как `eval_df.index.tolist()[:num_cases]` — детерминировано по порядку строк, не сэмплируется.
- **`run_visual_tests.py`** рисует только predicted lateral (без ground truth lateral).
- **Слабые тесты**: `test_temporal_smoothing` проверяет только `len`, `test_prediction_with_invalid_data` тестирует внутренний валидатор, а не end-to-end.
- **`test_prediction_module._check_consistency`**: ожидает низкий CV ответов при **разных** входах → инвертированная логика.
- **`test_model_loading.py`** при отсутствии ключей всё равно возвращает `True`.
- **Логирование основано на CWD**: `api.log`, `logs/training.log` — пишутся туда, откуда запущен процесс.
- **Per-split `LabelEncoder`** в `final_data_fix.py` → несогласованные коды категорий между train/val.
- **`fix_nan_issue.py` / `final_data_fix.py` перезаписывают** `train.csv`/`validation.csv` без бэкапа.

---

## 4. Низкие/стилевые проблемы (Low)

### 4.1. Стиль и читабельность

- **Emojis** в production-логах и `print` повсеместно (`✅`, `❌`, `🎉`, `🔧`, `📋`...). Делает grep по логам неудобным, ломает рендер в системах без UTF-8.
- **Микс RU/EN**: docstrings, message strings, имена переменных.
- **Сырые `print` вместо `logging`** во всех тренировочных и тестовых скриптах.
- **`warnings.filterwarnings('ignore')`** глобально в большинстве модулей — скрывает convergence-warning'и sklearn, deprecation-warning'и Pydantic.
- **`sys.path.append` / `sys.path.insert`** в production-коде:
  - `src/api/kidney_displacement_api.py`
  - `src/ar_system/kidney_ar_system.py`
  - `models/phase1/adaptive_ensemble.py`
  - `tests/test_system_integration.py`
  - root `test_*.py`
- **Магические числа** (HU thresholds, kidney volume bounds, 50.0/0.95/0.5 — `confidence`) без констант.
- **Неиспользуемые импорты**: `itertools` в `adaptive_ensemble.py`, `glob` в `compare_all_ensembles.py`, `cross_val_score` в `train_all_models.py`, повторный `import numpy as np` внутри функции в `api_kidney_predictor.py`.
- **Pydantic v1 `.dict()`** вместо v2 `model_dump()`.
- **Pylint/mypy hints**: отсутствуют почти во всём, кроме нового `scripts/validation/*`.
- **Опечатка**: `SmoothingMethod.SAVITZKY_GOLAY = "savitgky_golay"` в `src/smoothing.py` (L20).
- **`reload=True`, `host="0.0.0.0"`** прописаны в `__main__` `api_server.py` — dev-конфиг в production-файле.

### 4.2. Тесты

- **Два стиля одновременно**: `unittest`-классы (`tests/test_system_integration.py`, `tests/test_predictor.py`) и executable-скрипты (`tests/test_simple.py`, `tests/quick_test.py`).
- **Жёстко прибитые абсолютные пути**: `tests/quick_test.py` — `cwd="d:/ml trainer"` (Windows-only).
- **Дублирующиеся test-data-dict**'ы** в `diagnose_consistency.py`, `test_prediction_module.py`, `test_api.py`.
- **Опечатки**: "Prediciton" в `test_api_endpoints.py`.
- **Inconsistent host**: `127.0.0.1` vs `localhost` в разных тестах API.

### 4.3. Конфиги и упаковка

- **Нет `pyproject.toml` / `setup.cfg` / `.flake8` / `.editorconfig`**, хотя `black`, `flake8`, `mypy` лежат в `requirements.txt`.
- **Два конфига обучения** с разной парадигмой:
  - `config/unified_config.yaml` — мульти-таск (сегментация + координаты),
  - `config/training_config.yaml` — табличный ансамбль.
  - Нет указания, какой из них канонический.
- **`.gitignore` не покрывает**:
  - `api.log`,
  - `results/validation_runs/`,
  - `__pycache__` уже есть, но `*.json` в `models/` — нет.

### 4.4. Документация

- **Несколько README** (`README.md`, `docs/archive/README_FINAL.md`, `README_PRODUCTION.md`, `README_ENHANCED_PRODUCTION.md`, `README_PHASE3_RESEARCH.md`) — нет одного source-of-truth.
- В `models/` лежат старые `test_report_*.json` — мусор репозитория.
- Цифры производительности в README и `model_info` API не подтверждены вычислимыми артефактами в актуальной копии репозитория (`models/adaptive_ensemble.pkl` существует, но загружается только при совместимой версии sklearn).

---

## 5. Подробности по слоям

### 5.1. API (`src/api/*`, `models/phase1/api_kidney_predictor.py`)

- Три параллельных API (см. 1.8) с разными контрактами — основная архитектурная проблема.
- FastAPI стек использует устаревший `@app.on_event` без shutdown; глобальное состояние модели не освобождается.
- В Flask стеке нет thread-safety (`global predictor`) и нет rate-limit для batch.
- Confidence в FastAPI имитированный (`1.0 - abs(pred)/50.0`), не калиброван.
- Pydantic v2 несовместимости (`any`, `.dict()`).

### 5.2. Модели (`models/phase1/*`, `src/models/*`)

- Production-инференс `KidneyDisplacementPredictor` де-факто **не работает** (см. 1.2).
- `adaptive_ensemble.py` — самый длинный (~1000+ LOC), несёт большинство багов: shared estimators, train-val leakage, scaler без impute, скрытый feature drop.
- `train_lasso.py` / `train_ridge.py` падают на старте.
- Два разных формата сохраняемых артефактов:
  - `models/adaptive_ensemble.pkl` (joblib, dict из `models`, `scaler`, `feature_names`, ...)
  - `models/production/*.pkl` (по моделям отдельно, метаданные в `*.json`).
- Логика `data_integration_fix.py` создаёт `data/processed/train.csv` из частично синтетических признаков (`np.random.normal`).

### 5.3. Скрипты обучения и инференса

- Тренировочный пайплайн `train_all_models.py` корректно работает на train/val, **но не вызывает `evaluate_on_test`** — test split грузится и масштабируется, но не используется.
- Извлекатели признаков из DICOM имеют **рассогласованные** mapping'и `sex` / `body_type`, разные дефолты, разные имена выходных колонок.
- `enhanced_ct_extractor.py` содержит ~80 строк дублирующего dead-кода после `return result` и unreachable `except`.

### 5.4. Validation layer (`scripts/validation/*`)

Это самая молодая, наиболее аккуратная по стилю часть. Основные замечания:

- Fallback на on-the-fly RandomForest скрывает проблемы с production-моделью.
- Визуализации не унифицированы между тремя режимами.
- Метрики `within_5mm`/`within_10mm` считаются неправильно (поэлементно, а не по вектору).
- Отсутствует ground-truth overlay.
- `evaluate_metrics.py` записывает `worst_cases.csv` с лишним индексным столбцом.

### 5.5. Утилитарные модули (`src/`)

- Геометрические/координатные модули содержат смысловые ошибки (`get_capsule_points` дважды добавляет Z; `transform_features_to_patient` пишет только X).
- Дублирование пакетов (см. 2.11) — главный source of confusion.
- AR-пайплайн строит "успешный" результат даже при отсутствии модели.
- `Smoothing`-Kalman реализован с element-wise `/` на матрицах (численно сомнительно).
- Multiple bare `except:` блоки в `analyze_data_files.py`, `prepare_dataset.py`, `validation/data_validator.py`.

### 5.6. Тесты

- `unittest` тесты хороши там, где написаны (`tests/test_system_integration.py`, `tests/test_predictor.py`), но имеют hard-dep на артефакты, которых нет в чистом checkout.
- Корневые `test_*.py` — это **сценарные демо**, не CI-тесты.
- Покрытие критических модулей (`coordinate_system.py`, `relative_coordinates.py`, `confidence_scoring.py`, `smoothing.py`, `utils/imputer.py`) — отсутствует.

### 5.7. Дистрибуционный пакет (`kidney_displacement_prediction/`)

- Имеет правильную форму (`setup.py`, `config/*.yaml`, `requirements.txt`), но **не содержит кода**. Реальная имплементация всё ещё в `src/`.
- `setup.py` упадёт на `open("README.md")`.

---

## 6. Безопасность и эксплуатация

| Категория | Файл / место | Риск |
|-----------|--------------|------|
| Pickle RCE | Все `joblib.load` / `pickle.load` без подписи | Высокий, если артефакты приходят извне |
| CORS | `src/api/api_server.py` | Средний |
| Stack-trace leak | `models/phase1/api_kidney_predictor.py` | Средний |
| Race condition | `global predictor` без lock | Низкий (зависит от deploy) |
| Изменение source-данных | `dicom_feature_extractor.py` пишет в DICOM-папки | Средний |
| Логи без ротации | `api.log`, `logs/training.log` | Низкий |
| Hardcoded host `0.0.0.0` + `reload=True` | `api_server.py __main__` | Низкий |

---

## 7. Воспроизводимость

| Аспект | Состояние |
|--------|-----------|
| `random_state` в моделях | Есть в `RandomForest`, `XGBoost`, `KFold` |
| Seed в визуализации | Установлен в `scripts/validation/run_visual_tests.py` |
| Seed в DICOM-извлекателях | Отсутствует (`np.random.randint`) |
| Версии зависимостей | `requirements.txt` с `>=`, без верхней границы — артефакты не воспроизводятся между sklearn 1.5 ↔ 1.8 (уже подтверждено) |
| Lock-файл | Нет (`pip-compile`, `poetry.lock`, `uv.lock` отсутствуют) |
| Linter/formatter конфиг | Нет, хотя инструменты в `requirements.txt` |
| CI | Нет workflow в репозитории |

---

## 8. Топ-15 действий по приоритету (рекомендации, **не выполняются**)

1. Зафиксировать **один** canonical API + один Pydantic-контракт; удалить две другие реализации.
2. Привести `AdaptiveEnsembleTrainer.prepare_training_data` к контракту: `imputer.fit_transform` → `scaler.fit_transform`, и точно так же в инференсе.
3. В `evaluate_model_cv` использовать `clone(base_models[name])`, а не in-place fit.
4. Удалить hardcoded `kidney_*_delta_*` из `enhanced_ct_extractor.py`.
5. Удалить `data_integration_fix.py` synthetic generation или заменить детерминированной формулой с seed.
6. Прибить `FallbackHandler` к семантике "displacement, not position".
7. Зафиксировать версии: `scikit-learn==1.5.x`, `pydantic==2.x` (с явным портом API-кода под v2).
8. Заменить корневые `test_*.py` на pytest со скипами `@pytest.mark.skipif(not artifact_present)`.
9. Включить `pyproject.toml` с `[tool.black]`, `[tool.ruff]`, `[tool.mypy]`, GH Actions / Cursor Bugbot.
10. Поднять reproducibility: lock-файл + pin sklearn в `requirements.txt` (минимально для прод-модели).
11. Заменить fallback `RandomForest` в validation на **hard fail** при невозможности загрузить production-pkl, либо ясно прометить отчёт.
12. Объединить дублирующиеся классы (`DataValidator`, `ConfidenceEstimator`, `AnatomicalConstraints`, `ClinicalMetrics`, `SystemLogger`) в одну реализацию.
13. Перевести логи на `logging` модуль с `RotatingFileHandler`, убрать emoji-print.
14. Сделать `evaluate_metrics.py::within_5mm` per-patient по евклидовой норме.
15. Сделать визуализации унифицированными (одна функция-конструктор сцены, разные projections).

---

## 9. Контроль качества аудита

- Проверены: `src/` (полностью), `models/` (полностью), `scripts/` (полностью), `tests/` (полностью), корневые `test_*.py`, `diagnose_consistency.py`, конфиги `config/*.yaml`, `requirements.txt`, `.gitignore`, `kidney_displacement_prediction/`.
- Не открывались: notebooks (`notebooks/*.ipynb`), архивные `docs/archive/`, `kits19/` сырые данные, `venv/`.
- Все цитаты строк указаны как ориентир и могут смещаться на ±2 строки при правках, не затрагивающих логику.
- Аудит не модифицировал ни одного файла.

---

_Конец отчёта._
