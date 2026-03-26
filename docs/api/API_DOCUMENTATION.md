# AR Laparoscopy API Documentation

## Base URL

```
http://127.0.0.1:8000
```

## Общие принципы

- Все запросы и ответы в формате JSON
- Ошибки возвращаются с HTTP 4xx и полем `detail`
- Для ML-инференса фичи должны соответствовать `models/production/feature_names.json`

## Эндпоинты

### GET /health

Проверка работоспособности сервиса.

**Ответ:**
```json
{
  "status": "ok"
}
```

---

### GET /model_info

Метаданные загруженных ML-моделей.

**Ответ:**
```json
{
  "timestamp": "2026-03-20T11:52:54.514156",
  "feature_count": 37,
  "target_count": 9,
  "val_metrics": {
    "linear_regression": {"val_mae": 16.54},
    "random_forest": {"val_mae": 9.17},
    "xgboost": {"val_mae": 10.72},
    "ensemble": {"val_mae": 9.90, "weight_rf": 0.5, "weight_xgb": 0.5}
  }
}
```

---

### POST /predict_displacement

Предсказание смещения почек (supine → lateral).

**Request Body:**
```json
{
  "features": {
    "age": 65,
    "sex_M": 1,
    "sex_F": 0,
    "X_upper_supine": 120.5,
    "Y_upper_supine": 85.2,
    "Z_upper_supine": 45.7,
    "...": "..."
  }
}
```

**Ответ:**
```json
{
  "predictions": {
    "delta_X_lower": -7.3,
    "delta_X_middle": -7.2,
    "delta_X_upper": -3.8,
    "delta_Y_lower": -2.2,
    "delta_Y_middle": 0.1,
    "delta_Y_upper": 0.3,
    "delta_Z_lower": -1.9,
    "delta_Z_middle": -1.5,
    "delta_Z_upper": -1.2
  }
}
```

**Ошибки:**
- `400` — отсутствуют или некорректны фичи

---

### POST /plan_trocars

Планирование позиций троакаров.

**Request Body:**
```json
{
  "upper": {"x": 120.5, "y": 85.2, "z": 45.7},
  "middle": {"x": 118.0, "y": 84.0, "z": 44.5},
  "lower": {"x": 115.5, "y": 82.8, "z": 43.3}
}
```

**Ответ:**
```json
{
  "trocars": [
    {
      "name": "camera",
      "position_mm": {"x": 10.2, "y": -20.1, "z": 5.3},
      "depth_mm": 120.0,
      "entry_angle_deg": 35.4,
      "safety_score": 0.8
    },
    {
      "name": "working_1",
      "position_mm": {"x": 8.5, "y": -18.7, "z": 4.9},
      "depth_mm": 110.0,
      "entry_angle_deg": 32.1,
      "safety_score": 0.8
    },
    {
      "name": "working_2",
      "position_mm": {"x": 12.1, "y": -21.5, "z": 5.8},
      "depth_mm": 110.0,
      "entry_angle_deg": 38.7,
      "safety_score": 0.8
    }
  ]
}
```

**Поля:**
- `name` — тип троакара (camera/working_1/working_2)
- `position_mm` — координаты входа в мм
- `depth_mm` — глубина введения до почки
- `entry_angle_deg` — угол входа относительно антериорного направления
- `safety_score` — оценка безопасности (0–1)

**Ошибки:**
- `400` — отсутствуют или некорректны координаты

---

### POST /full_pipeline

Полный цикл: предсказание смещения + планирование троакаров.

**Request Body:**
```json
{
  "features": {
    "age": 65,
    "sex_M": 1,
    "sex_F": 0,
    "X_upper_supine": 120.5,
    "Y_upper_supine": 85.2,
    "Z_upper_supine": 45.7,
    "...": "..."
  },
  "kidney_points_mm": {
    "upper": {"x": 120.5, "y": 85.2, "z": 45.7},
    "middle": {"x": 118.0, "y": 84.0, "z": 44.5},
    "lower": {"x": 115.5, "y": 82.8, "z": 43.3}
  }
}
```

**Ответ:**
```json
{
  "displacement": {
    "predictions": {
      "delta_X_lower": -7.3,
      "...": "..."
    }
  },
  "trocars": {
    "trocars": [
      {
        "name": "camera",
        "...": "..."
      },
      "...": "..."
    ]
  }
}
```

**Ошибки:**
- `400` — ошибки в фичах или координатах

## Примеры использования

### curl

```bash
# Health
curl http://127.0.0.1:8000/health

# Predict displacement
curl -X POST http://127.0.0.1:8000/predict_displacement \
  -H "Content-Type: application/json" \
  -d '{"features":{"age":65,"sex_M":1,"sex_F":0,"X_upper_supine":120.5,"Y_upper_supine":85.2,"Z_upper_supine":45.7}}'

# Plan trocars
curl -X POST http://127.0.0.1:8000/plan_trocars \
  -H "Content-Type: application/json" \
  -d '{"upper":{"x":120.5,"y":85.2,"z":45.7},"middle":{"x":118.0,"y":84.0,"z":44.5},"lower":{"x":115.5,"y":82.8,"z":43.3}}'
```

### Python

```python
import requests

base = "http://127.0.0.1:8000"

# Predict displacement
features = {"age": 65, "sex_M": 1, "sex_F": 0, "X_upper_supine": 120.5, ...}
resp = requests.post(f"{base}/predict_displacement", json={"features": features})
print(resp.json())

# Plan trocars
kidney = {"upper": {"x": 120.5, "y": 85.2, "z": 45.7}, ...}
resp = requests.post(f"{base}/plan_trocars", json=kidney)
print(resp.json())
```

## Ограничения

- Сервер работает в режиме single-thread (uvicorn default)
- Нет аутентификации (для внутреннего использования)
- Trocar planning использует упрощённую геометрическую модель
- ML-модели обучены на ограниченном датасете (см. model_info)

## Продакшен-развертывание

Рекомендуется:
- Использовать WSGI-сервер (gunicorn/uvicorn) с несколькими worker'ами
- Добавить HTTPS и аутентификацию
- Настроить health checks и логирование
- Валидировать входные данные на стороне клиента
