# Руководство по интеграции AR Laparoscopy API

## Обзор

API предоставляет две основные функции:
1. **ML-предсказание** смещения почек при изменении положения пациента
2. **Планирование троакаров** на основе анатомии почек

## Схема интеграции

```
DICOM → Feature Extraction → ML Prediction → Trocar Planning → AR Visualization
```

### Шаг 1: Извлечение признаков из DICOM

Используйте скрипты в `scripts/inference/`:

```python
# Для одного файла
from scripts.inference.convert_single_file import extract_features
features = extract_features("path/to/dicom")

# Или пакетная обработка
from scripts.inference.extract_from_dicom import batch_extract
features_df = batch_extract("path/to/dicom_folder")
```

**Важно:** Убедитесь, что извлечённые фичи содержат все 37 полей из `models/production/feature_names.json`.

### Шаг 2: Вызов API для предсказания

```python
import requests

API_BASE = "http://127.0.0.1:8000"

def predict_displacement(features: dict) -> dict:
    response = requests.post(
        f"{API_BASE}/predict_displacement",
        json={"features": features}
    )
    response.raise_for_status()
    return response.json()["predictions"]
```

### Шаг 3: Планирование троакаров

```python
def plan_trocars(kidney_points: dict) -> list:
    response = requests.post(
        f"{API_BASE}/plan_trocars",
        json=kidney_points
    )
    response.raise_for_status()
    return response.json()["trocars"]
```

### Шаг 4: Визуализация в AR

Передайте координаты троакаров в AR-систему:

```python
def render_trocars_in_ar(trocar_list):
    for trocar in trocar_list:
        pos = trocar["position_mm"]
        # Отправить в AR SDK
        ar_sdk.place_trocar(
            name=trocar["name"],
            position=(pos["x"], pos["y"], pos["z"]),
            depth_mm=trocar["depth_mm"],
            angle_deg=trocar["entry_angle_deg"]
        )
```

## Полный пример интеграции

```python
import requests
from pathlib import Path

API_BASE = "http://127.0.0.1:8000"

class LaparoscopyIntegration:
    def __init__(self, api_base: str = API_BASE):
        self.api_base = api_base
    
    def extract_features_from_dicom(self, dicom_path: str) -> dict:
        """Извлечь фичи из DICOM (заглушка, замените на реальный вызов)"""
        # TODO: интегрировать с scripts/inference/
        pass
    
    def predict_displacement(self, features: dict) -> dict:
        """Предсказать смещение почек"""
        resp = requests.post(
            f"{self.api_base}/predict_displacement",
            json={"features": features}
        )
        resp.raise_for_status()
        return resp.json()["predictions"]
    
    def plan_trocars(self, kidney_points: dict) -> list:
        """Запланировать троакары"""
        resp = requests.post(
            f"{self.api_base}/plan_trocars",
            json=kidney_points
        )
        resp.raise_for_status()
        return resp.json()["trocars"]
    
    def full_pipeline(self, features: dict, kidney_points: dict) -> dict:
        """Полный цикл: предсказание + планирование"""
        resp = requests.post(
            f"{self.api_base}/full_pipeline",
            json={
                "features": features,
                "kidney_points_mm": kidney_points
            }
        )
        resp.raise_for_status()
        return resp.json()

# Использование
integration = LaparoscopyIntegration()

# Шаг 1: извлечь фичи из DICOM
features = integration.extract_features_from_dicom("patient_001.dcm")

# Шаг 2: получить точки почек (из сегментации или вручную)
kidney_points = {
    "upper": {"x": 120.5, "y": 85.2, "z": 45.7},
    "middle": {"x": 118.0, "y": 84.0, "z": 44.5},
    "lower": {"x": 115.5, "y": 82.8, "z": 43.3}
}

# Шаг 3: полный цикл
result = integration.full_pipeline(features, kidney_points)

# Шаг 4: визуализация
displacement = result["displacement"]["predictions"]
trocars = result["trocars"]["trocars"]

print(f"Predicted displacement: {displacement}")
print(f"Trocar positions: {trocars}")
```

## Обработка ошибок

```python
def safe_api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return None
```

## Валидация входных данных

Перед вызовом API проверяйте:

```python
def validate_features(features: dict) -> bool:
    required = set(json.load(open("models/production/feature_names.json")))
    return required.issubset(set(features.keys()))

def validate_kidney_points(points: dict) -> bool:
    required_keys = {"x", "y", "z"}
    return all(
        all(k in pt for k in required_keys)
        for pt in points.values()
    )
```

## Производительность

- **Latency**: предсказание ~50-100ms, планирование ~10ms
- **Throughput**: uvicorn default ~1000 req/s (single worker)
- **Memory**: ML-модели ~50MB RAM

Для продакшена используйте несколько worker'ов:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Логирование и мониторинг

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("laparoscopy_integration")

def predict_with_logging(features):
    logger.info("Starting displacement prediction")
    result = predict_displacement(features)
    logger.info(f"Prediction completed: {len(result)} deltas")
    return result
```

## Безопасность

- Для внутреннего использования — нет аутентификации
- Для внешнего — добавьте API key или OAuth
- Валидируйте все входные данные на стороне клиента
- Используйте HTTPS в продакшене

## Тестирование

```python
def test_integration():
    # Mock данные
    features = {"age": 65, "sex_M": 1, "sex_F": 0, ...}
    kidney_points = {
        "upper": {"x": 120.5, "y": 85.2, "z": 45.7},
        "middle": {"x": 118.0, "y": 84.0, "z": 44.5},
        "lower": {"x": 115.5, "y": 82.8, "z": 43.3}
    }
    
    integration = LaparoscopyIntegration()
    result = integration.full_pipeline(features, kidney_points)
    
    assert "displacement" in result
    assert "trocars" in result
    assert len(result["displacement"]["predictions"]) == 9
    assert len(result["trocars"]["trocars"]) == 3
```

## Следующие шаги

1. **Интегрировать** извлечение признаков из DICOM в основной pipeline
2. **Добавить** валидацию предсказаний (проверка на анатомически невозможные значения)
3. **Оптимизировать** trocar planning под конкретные хирургические протоколы
4. **Реализовать** кэширование предсказаний для повторных запросов

## Поддержка

При проблемах:
- Проверьте логи API: `uvicorn` выводит ошибки в консоль
- Сверьте фичи с `models/production/feature_names.json`
- Убедитесь, что сервер запущен и доступен по адресу
