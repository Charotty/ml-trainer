# 🏥 API для предсказания смещения почек

## Обзор

API предсказания смещения почек на основе оптимизированной адаптивной модели ансамбля. Модель использует 51 признак (23 базовых + 13 инженерных + 15 cross-features) для предсказания смещения по 6 осям.

---

## 🚀 Быстрый старт

### Установка и запуск

```bash
# Активация виртуального окружения
cd "d:\ml trainer"
venv\Scripts\activate

# Запуск API сервера
python src/api/api_server.py
```

### Базовый URL
```
http://127.0.0.1:8000
```

---

## 📋 Эндпоинты

### 1. GET /health

Проверка работоспособности сервиса.

**Ответ:**
```json
{
  "status": "ok",
  "model_version": "optimized_adaptive_ensemble_v1.0",
  "features_count": 51,
  "targets_count": 6
}
```

---

### 2. GET /model_info

Детальная информация о модели.

**Ответ:**
```json
{
  "model_info": {
    "name": "Optimized Adaptive Ensemble",
    "version": "1.0",
    "features_count": 51,
    "targets_count": 6,
    "data_sources": "DICOMS+Vybor+KiTS19",
    "performance": {
      "status": "unavailable",
      "average_mae_mm": null,
      "average_r2": null,
      "accuracy_5mm": null,
      "accuracy_10mm": null,
      "detail": "training_meta present but no performance metrics"
    },
    "training_meta": {
      "clinical_only": true
    },
    "feature_types": {
      "base_features": 23,
      "engineered_features": 13,
      "cross_features": 15
    },
    "optimized_weights": {
      "kidney_left_delta_x": {
        "RandomForest": 0.614,
        "Lasso": 0.317,
        "Ridge": 0.069,
        "GradientBoosting": 0.0
      }
    }
  },
  "feature_names": [
    "kidney_left_center_x_rel",
    "kidney_left_center_y_rel",
    "kidney_left_center_z_rel",
    "kidney_right_center_x_rel",
    "kidney_right_center_y_rel",
    "kidney_right_center_z_rel",
    "kidney_left_length_mm",
    "kidney_left_volume_cm3",
    "kidney_right_length_mm",
    "kidney_right_volume_cm3",
    "body_width_mm",
    "body_depth_mm",
    "body_area_mm2",
    "kidney_left_to_spine_distance",
    "kidney_right_to_spine_distance",
    "kidney_left_to_body_center_distance",
    "kidney_right_to_body_center_distance",
    "spine_center_x",
    "spine_center_y",
    "spine_center_z",
    "body_com_x",
    "body_com_y",
    "body_com_z",
    "body_ratio",
    "kidney_distance_lr",
    "kidney_left_volume_norm",
    "kidney_right_volume_norm",
    "kidney_left_length_norm",
    "kidney_right_length_norm",
    "volume_asymmetry",
    "length_asymmetry",
    "spine_distance_asymmetry",
    "body_center_asymmetry",
    "kidney_left_to_spine_ratio",
    "kidney_right_to_spine_ratio",
    "patient_position_encoded",
    "body_volume_estimated",
    "kidney_left_density_ratio",
    "kidney_right_density_ratio",
    "spine_to_body_ratio_x",
    "spine_to_body_ratio_y",
    "body_com_to_spine_distance",
    "kidney_left_spine_interaction",
    "kidney_right_spine_interaction",
    "body_size_index",
    "kidney_position_index_left",
    "kidney_position_index_right",
    "volume_to_area_ratio_left",
    "volume_to_area_ratio_right",
    "relative_volume_sum",
    "kidney_separation_angle"
  ]
}
```

---

### 3. POST /predict

Основной эндпоинт для предсказания смещения почек.

**Request Body:**
```json
{
  "patient_data": {
    "kidney_left_center_x_rel": 85.5,
    "kidney_left_center_y_rel": 142.3,
    "kidney_left_center_z_rel": -745.2,
    "kidney_right_center_x_rel": 95.8,
    "kidney_right_center_y_rel": 148.7,
    "kidney_right_center_z_rel": -752.1,
    "kidney_left_length_mm": 98.5,
    "kidney_left_volume_cm3": 125.3,
    "kidney_right_length_mm": 102.1,
    "kidney_right_volume_cm3": 132.7,
    "body_width_mm": 385.2,
    "body_depth_mm": 285.6,
    "body_area_mm2": 110000.0,
    "kidney_left_to_spine_distance": 45.3,
    "kidney_right_to_spine_distance": 48.7,
    "kidney_left_to_body_center_distance": 92.1,
    "kidney_right_to_body_center_distance": 96.4,
    "spine_center_x": 0.0,
    "spine_center_y": 0.0,
    "spine_center_z": 0.0,
    "body_com_x": 0.0,
    "body_com_y": 0.0,
    "body_com_z": 0.0
  }
}
```

**Ответ:**
```json
{
  "success": true,
  "predictions": {
    "kidney_left_delta_x": 15.317,
    "kidney_left_delta_y": 5.527,
    "kidney_left_delta_z": 8.894,
    "kidney_right_delta_x": -5.872,
    "kidney_right_delta_y": 4.943,
    "kidney_right_delta_z": 8.851
  },
  "metadata": {
    "model_version": "optimized_adaptive_ensemble_v1.0",
    "features_used": 51,
    "prediction_confidence": {
      "kidney_left_delta_x": 0.85,
      "kidney_left_delta_y": 0.92,
      "kidney_left_delta_z": 0.88,
      "kidney_right_delta_x": 0.87,
      "kidney_right_delta_y": 0.91,
      "kidney_right_delta_z": 0.89
    }
  }
}
```

**Ошибки:**
- `400 Bad Request` - отсутствуют или некорректные входные данные
- `500 Internal Server Error` - ошибка модели или сервера

---

### 4. POST /predict_batch

Пакетное предсказание для нескольких пациентов.

**Request Body:**
```json
{
  "patients": [
    {
      "patient_id": "patient_001",
      "patient_data": {
        "kidney_left_center_x_rel": 85.5,
        "kidney_left_center_y_rel": 142.3,
        "kidney_left_center_z_rel": -745.2,
        "kidney_right_center_x_rel": 95.8,
        "kidney_right_center_y_rel": 148.7,
        "kidney_right_center_z_rel": -752.1,
        "kidney_left_length_mm": 98.5,
        "kidney_left_volume_cm3": 125.3,
        "kidney_right_length_mm": 102.1,
        "kidney_right_volume_cm3": 132.7,
        "body_width_mm": 385.2,
        "body_depth_mm": 285.6,
        "body_area_mm2": 110000.0,
        "kidney_left_to_spine_distance": 45.3,
        "kidney_right_to_spine_distance": 48.7,
        "kidney_left_to_body_center_distance": 92.1,
        "kidney_right_to_body_center_distance": 96.4,
        "spine_center_x": 0.0,
        "spine_center_y": 0.0,
        "spine_center_z": 0.0,
        "body_com_x": 0.0,
        "body_com_y": 0.0,
        "body_com_z": 0.0
      }
    },
    {
      "patient_id": "patient_002",
      "patient_data": {
        "kidney_left_center_x_rel": 90.2,
        "kidney_left_center_y_rel": 138.7,
        "kidney_left_center_z_rel": -750.1,
        "kidney_right_center_x_rel": 98.5,
        "kidney_right_center_y_rel": 145.2,
        "kidney_right_center_z_rel": -758.3,
        "kidney_left_length_mm": 101.3,
        "kidney_left_volume_cm3": 130.7,
        "kidney_right_length_mm": 105.8,
        "kidney_right_volume_cm3": 138.2,
        "body_width_mm": 395.1,
        "body_depth_mm": 295.3,
        "body_area_mm2": 116500.0,
        "kidney_left_to_spine_distance": 47.2,
        "kidney_right_to_spine_distance": 50.1,
        "kidney_left_to_body_center_distance": 94.7,
        "kidney_right_to_body_center_distance": 98.9,
        "spine_center_x": 0.0,
        "spine_center_y": 0.0,
        "spine_center_z": 0.0,
        "body_com_x": 0.0,
        "body_com_y": 0.0,
        "body_com_z": 0.0
      }
    }
  ]
}
```

**Ответ:**
```json
{
  "success": true,
  "results": [
    {
      "patient_id": "patient_001",
      "predictions": {
        "kidney_left_delta_x": 15.317,
        "kidney_left_delta_y": 5.527,
        "kidney_left_delta_z": 8.894,
        "kidney_right_delta_x": -5.872,
        "kidney_right_delta_y": 4.943,
        "kidney_right_delta_z": 8.851
      }
    },
    {
      "patient_id": "patient_002",
      "predictions": {
        "kidney_left_delta_x": 14.892,
        "kidney_left_delta_y": 5.218,
        "kidney_left_delta_z": 8.657,
        "kidney_right_delta_x": -6.125,
        "kidney_right_delta_y": 4.782,
        "kidney_right_delta_z": 8.913
      }
    }
  ],
  "metadata": {
    "total_patients": 2,
    "successful_predictions": 2,
    "model_version": "optimized_adaptive_ensemble_v1.0"
  }
}
```

---

## 📖 Подробное описание признаков

### Базовые признаки (23)

| Признак | Тип | Описание | Единицы |
|---------|-----|----------|---------|
| `kidney_left_center_x_rel` | float | Относительная X-координата центра левой почки | мм |
| `kidney_left_center_y_rel` | float | Относительная Y-координата центра левой почки | мм |
| `kidney_left_center_z_rel` | float | Относительная Z-координата центра левой почки | мм |
| `kidney_right_center_x_rel` | float | Относительная X-координата центра правой почки | мм |
| `kidney_right_center_y_rel` | float | Относительная Y-координата центра правой почки | мм |
| `kidney_right_center_z_rel` | float | Относительная Z-координата центра правой почки | мм |
| `kidney_left_length_mm` | float | Длина левой почки | мм |
| `kidney_left_volume_cm3` | float | Объем левой почки | см³ |
| `kidney_right_length_mm` | float | Длина правой почки | мм |
| `kidney_right_volume_cm3` | float | Объем правой почки | см³ |
| `body_width_mm` | float | Ширина тела пациента | мм |
| `body_depth_mm` | float | Глубина тела пациента | мм |
| `body_area_mm2` | float | Площадь поперечного сечения | мм² |
| `kidney_left_to_spine_distance` | float | Расстояние от левой почки до позвоночника | мм |
| `kidney_right_to_spine_distance` | float | Расстояние от правой почки до позвоночника | мм |
| `kidney_left_to_body_center_distance` | float | Расстояние от левой почки до центра масс тела | мм |
| `kidney_right_to_body_center_distance` | float | Расстояние от правой почки до центра масс тела | мм |
| `spine_center_x` | float | X-координата центра позвоночника | мм |
| `spine_center_y` | float | Y-координата центра позвоночника | мм |
| `spine_center_z` | float | Z-координата центра позвоночника | мм |
| `body_com_x` | float | X-координата центра масс тела | мм |
| `body_com_y` | float | Y-координата центра масс тела | мм |
| `body_com_z` | float | Z-координата центра масс тела | мм |

### Инженерные признаки (13)

Эти признаки создаются автоматически из базовых:

| Признак | Описание |
|---------|----------|
| `body_ratio` | Отношение ширины тела к глубине |
| `kidney_distance_lr` | Расстояние между левой и правой почками |
| `kidney_left_volume_norm` | Объем левой почки, нормализованный на ширину тела |
| `kidney_right_volume_norm` | Объем правой почки, нормализованный на ширину тела |
| `kidney_left_length_norm` | Длина левой почки, нормализованная на ширину тела |
| `kidney_right_length_norm` | Длина правой почки, нормализованная на ширину тела |
| `volume_asymmetry` | Разница объемов левой и правой почек |
| `length_asymmetry` | Разница длин левой и правой почек |
| `spine_distance_asymmetry` | Асимметрия расстояний до позвоночника |
| `body_center_asymmetry` | Асимметрия расстояний до центра масс |
| `kidney_left_to_spine_ratio` | Отношение расстояния до позвоночника к ширине тела |
| `kidney_right_to_spine_ratio` | Отношение расстояния до позвоночника к ширине тела |
| `patient_position_encoded` | Кодированное положение пациента (1 = supine) |

### Cross-features (15)

Эти признаки также создаются автоматически:

| Признак | Описание |
|---------|----------|
| `body_volume_estimated` | Оценочный объем тела |
| `kidney_left_density_ratio` | Плотность левой почки (объем/длина) |
| `kidney_right_density_ratio` | Плотность правой почки (объем/длина) |
| `spine_to_body_ratio_x` | Отношение координаты позвоночника к ширине тела |
| `spine_to_body_ratio_y` | Отношение координаты позвоночника к глубине тела |
| `body_com_to_spine_distance` | Расстояние между центром масс и позвоночником |
| `kidney_left_spine_interaction` | Взаимодействие левой почки и позвоночника |
| `kidney_right_spine_interaction` | Взаимодействие правой почки и позвоночника |
| `body_size_index` | Индекс размера тела |
| `kidney_position_index_left` | Позиционный индекс левой почки |
| `kidney_position_index_right` | Позиционный индекс правой почки |
| `volume_to_area_ratio_left` | Отношение объема левой почки к площади тела |
| `volume_to_area_ratio_right` | Отношение объема правой почки к площади тела |
| `relative_volume_sum` | Сумма относительных объемов почек |
| `kidney_separation_angle` | Угол между почками относительно позвоночника |

---

## 🎯 Целевые переменные

| Переменная | Описание | Единицы | Типичный диапазон |
|------------|----------|---------|------------------|
| `kidney_left_delta_x` | Смещение левой почки по оси X | мм | -20...+20 |
| `kidney_left_delta_y` | Смещение левой почки по оси Y | мм | -10...+10 |
| `kidney_left_delta_z` | Смещение левой почки по оси Z | мм | -15...+15 |
| `kidney_right_delta_x` | Смещение правой почки по оси X | мм | -20...+20 |
| `kidney_right_delta_y` | Смещение правой почки по оси Y | мм | -10...+10 |
| `kidney_right_delta_z` | Смещение правой почки по оси Z | мм | -15...+15 |

---

## 💻 Примеры использования

### Python

```python
import requests
import json

# Базовый URL API
base_url = "http://127.0.0.1:8000"

# 1. Проверка здоровья сервера
response = requests.get(f"{base_url}/health")
print(response.json())

# 2. Получение информации о модели
response = requests.get(f"{base_url}/model_info")
model_info = response.json()
print(f"Модель использует {model_info['model_info']['features_count']} признаков")

# 3. Предсказание смещения
patient_data = {
    "patient_data": {
        "kidney_left_center_x_rel": 85.5,
        "kidney_left_center_y_rel": 142.3,
        "kidney_left_center_z_rel": -745.2,
        "kidney_right_center_x_rel": 95.8,
        "kidney_right_center_y_rel": 148.7,
        "kidney_right_center_z_rel": -752.1,
        "kidney_left_length_mm": 98.5,
        "kidney_left_volume_cm3": 125.3,
        "kidney_right_length_mm": 102.1,
        "kidney_right_volume_cm3": 132.7,
        "body_width_mm": 385.2,
        "body_depth_mm": 285.6,
        "body_area_mm2": 110000.0,
        "kidney_left_to_spine_distance": 45.3,
        "kidney_right_to_spine_distance": 48.7,
        "kidney_left_to_body_center_distance": 92.1,
        "kidney_right_to_body_center_distance": 96.4,
        "spine_center_x": 0.0,
        "spine_center_y": 0.0,
        "spine_center_z": 0.0,
        "body_com_x": 0.0,
        "body_com_y": 0.0,
        "body_com_z": 0.0
    }
}

response = requests.post(f"{base_url}/predict", json=patient_data)
if response.status_code == 200:
    result = response.json()
    print("Предсказания:")
    for target, value in result['predictions'].items():
        print(f"  {target}: {value:.3f} мм")
else:
    print(f"Ошибка: {response.status_code} - {response.text}")
```

### JavaScript

```javascript
// Базовый URL API
const baseUrl = 'http://127.0.0.1:8000';

// Функция для предсказания смещения
async function predictDisplacement(patientData) {
    try {
        const response = await fetch(`${baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                patient_data: patientData
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Пример использования
const patientData = {
    kidney_left_center_x_rel: 85.5,
    kidney_left_center_y_rel: 142.3,
    kidney_left_center_z_rel: -745.2,
    kidney_right_center_x_rel: 95.8,
    kidney_right_center_y_rel: 148.7,
    kidney_right_center_z_rel: -752.1,
    kidney_left_length_mm: 98.5,
    kidney_left_volume_cm3: 125.3,
    kidney_right_length_mm: 102.1,
    kidney_right_volume_cm3: 132.7,
    body_width_mm: 385.2,
    body_depth_mm: 285.6,
    body_area_mm2: 110000.0,
    kidney_left_to_spine_distance: 45.3,
    kidney_right_to_spine_distance: 48.7,
    kidney_left_to_body_center_distance: 92.1,
    kidney_right_to_body_center_distance: 96.4,
    spine_center_x: 0.0,
    spine_center_y: 0.0,
    spine_center_z: 0.0,
    body_com_x: 0.0,
    body_com_y: 0.0,
    body_com_z: 0.0
};

predictDisplacement(patientData)
    .then(result => {
        console.log('Предсказания:');
        const predictions = result.predictions;
        Object.entries(predictions).forEach(([target, value]) => {
            console.log(`${target}: ${value.toFixed(3)} мм`);
        });
    })
    .catch(error => {
        console.error('Ошибка предсказания:', error);
    });
```

### cURL

```bash
# Проверка здоровья
curl -X GET http://127.0.0.1:8000/health

# Получение информации о модели
curl -X GET http://127.0.0.1:8000/model_info

# Предсказание смещения
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "kidney_left_center_x_rel": 85.5,
      "kidney_left_center_y_rel": 142.3,
      "kidney_left_center_z_rel": -745.2,
      "kidney_right_center_x_rel": 95.8,
      "kidney_right_center_y_rel": 148.7,
      "kidney_right_center_z_rel": -752.1,
      "kidney_left_length_mm": 98.5,
      "kidney_left_volume_cm3": 125.3,
      "kidney_right_length_mm": 102.1,
      "kidney_right_volume_cm3": 132.7,
      "body_width_mm": 385.2,
      "body_depth_mm": 285.6,
      "body_area_mm2": 110000.0,
      "kidney_left_to_spine_distance": 45.3,
      "kidney_right_to_spine_distance": 48.7,
      "kidney_left_to_body_center_distance": 92.1,
      "kidney_right_to_body_center_distance": 96.4,
      "spine_center_x": 0.0,
      "spine_center_y": 0.0,
      "spine_center_z": 0.0,
      "body_com_x": 0.0,
      "body_com_y": 0.0,
      "body_com_z": 0.0
    }
  }'
```

---

## ⚠️ Ограничения и рекомендации

### Ограничения модели
- **Положение пациента**: Модель обучена на данных пациентов в положении supine (на спине)
- **Диапазон значений**: Предсказания надежны в обученном диапазоне признаков
- **Валидация**: Рекомендуется проверять предсказания на адекватность (>30 мм считается аномальным)

### Рекомендации по использованию
1. **Валидация входных данных**: Проверяйте диапазоны значений перед отправкой
2. **Мониторинг**: Установите пороги для аномальных предсказаний
3. **Логирование**: Сохраняйте запросы и ответы для анализа
4. **Тестирование**: Проверяйте работу API на тестовых данных перед интеграцией

### Производительность
- **Время ответа**: ~100-500 мс на одно предсказание
- **Память**: ~500MB для загрузки модели
- **CPU**: Модель использует CPU, не требует GPU

---

## 🔧 Интеграция и развертывание

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "src/api/api_server.py"]
```

### Продакшен-настройки

```bash
# Запуск с несколькими worker'ами
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.api_server:app --host 0.0.0.0 --port 8000

# С HTTPS
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.api_server:app --host 0.0.0.0 --port 8000 --keyfile key.pem --certfile cert.pem
```

### Мониторинг

```python
# Пример middleware для логирования
import logging
import time
from fastapi import Request

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logging.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
    return response
```

---

## 📞 Поддержка

При возникновении проблем:
1. Проверьте журнал ошибок сервера
2. Убедитесь в корректности формата входных данных
3. Проверьте доступность модели по эндпоинту `/health`
4. Обратитесь к документации модели в `docs/`

**Версия API**: 1.0  
**Версия модели**: optimized_adaptive_ensemble_v1.0  
**Дата обновления**: 26 марта 2026
