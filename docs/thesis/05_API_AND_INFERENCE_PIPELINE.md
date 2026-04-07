# 05. API и inference pipeline (FastAPI)

## 1. Назначение API уровня
API нужен, чтобы:
- предоставить единый интерфейс для внешних систем (клиника, исследовательские сервисы, интеграция в AR/планирование);
- стандартизировать входные данные;
- обеспечить повторяемость feature engineering и нормализации;
- отдавать предсказания + метаданные.

## 2. Основной API модуль
Файл: `src/api/kidney_displacement_api.py`

### 2.1. Инициализация
- создаётся объект `FastAPI(...)` с метаданными (title/description/version)
- глобальные переменные:
  - `model_data` — загруженный joblib артефакт
  - `trainer` — объект `AdaptiveEnsembleTrainer()`
  - `feature_names` — список признаков из артефакта модели

### 2.2. Загрузка модели (`load_model()`)
Логика:
1) `model_path = models/adaptive_ensemble.pkl`
2) `model_data = joblib.load(model_path)`
3) `trainer = AdaptiveEnsembleTrainer()` — нужен не для обучения, а для генерации признаков
4) `feature_names = model_data['feature_names']`

Критический момент:
- **тренер в API используется как “feature generator”**, поэтому любые изменения feature engineering в trainer должны быть согласованы с обучением.

## 3. Контракт входных данных (Pydantic)

### 3.1. `PatientData`
Содержит 23 базовых поля (все `float`), например:
- `kidney_left_center_x_rel`, `kidney_left_center_y_rel`, `kidney_left_center_z_rel`
- `kidney_left_length_mm`, `kidney_left_volume_cm3`
- `body_width_mm`, `body_depth_mm`, `body_area_mm2`
- расстояния до позвоночника и центра тела
- координаты центров `spine_center_*`, `body_com_*` (часть из них имеет default=0)

Для диссертации важно:
- объяснить смысл каждого поля;
- объяснить единицы измерения;
- объяснить почему часть координат имеет default.

### 3.2. `PredictRequest`
- поле `patient_data: PatientData`

### 3.3. `BatchPredictRequest`
- поле `patients: List[Dict[str, PatientData]]`

Замечание:
- структура batch-запроса в текущем коде выглядит нетипично (dict внутри list). Для диссертации важно зафиксировать реальный контракт, а при необходимости предложить нормализованный вариант.

## 4. Feature engineering на inference

### 4.1. `create_features(patient_data)`
1) входной объект `PatientData` превращается в dict
2) dict превращается в DataFrame из одной строки
3) вызываются:
   - `trainer._create_engineered_features(df)`
   - `trainer._create_cross_features(df)`

То есть API повторяет feature engineering тренера.

### 4.2. Проверка признаков
Перед инференсом:
- вычисляется `missing_features = [f for f in feature_names if f not in df.columns]`
- если отсутствуют, возвращается HTTP 400

Этот механизм важен как “охранный контур”, который защищает от:
- несовместимого входного JSON;
- рассинхронизации feature engineering.

## 5. Нормализация и предсказание

### 5.1. Масштабирование
- `X = df[feature_names].values`
- `X_scaled = model_data['scaler'].transform(X)`

### 5.2. Предсказание
Артефакт модели содержит словарь моделей по целям:
- `for target_name, model in model_data['models'].items(): pred = model.predict(X_scaled)[0]`

Выход:
- `predictions: Dict[str, float]`

## 6. Эндпоинты

- `GET /health`
  - проверка статуса и числа признаков/целей
- `GET /model_info`
  - информация о модели + список признаков
- `POST /predict`
  - одиночное предсказание
- `POST /predict_batch`
  - пакетное предсказание
- `GET /`
  - базовая информация

## 7. Метаданные и “confidence”

### 7.1. Confidence в текущем API
В `POST /predict` confidence вычисляется упрощённо:
- `confidence = min(0.95, max(0.5, 1.0 - abs(pred)/50.0))`

Это **не статистическая** неопределённость, а эвристика.

В диссертации важно честно обозначить:
- это “псевдо-confidence” для интерфейса;
- для клинической интерпретации требуется строгий метод (например, ансамблевая дисперсия, байесовские подходы, conformal prediction).

## 8. Ошибки и устойчивость
- ошибки загрузки модели логируются и приводят к `RuntimeError` на startup
- ошибки инференса приводят к HTTP 500

Для диссертации:
- описать типы ошибок (входные данные, отсутствие признаков, несовместимость артефакта)
- описать, как обеспечивать наблюдаемость (логирование).

## 9. Что включить в диссертацию дополнительно
- диаграмму последовательности: “клиент → API → feature engineering → scaler → модели → ответ”
- таблицу “endpoint → вход → выход → ошибки”
- раздел о безопасности: валидация диапазонов, запрет NaN, контроль единиц
