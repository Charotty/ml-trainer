# 🚀 Руководство по развертыванию API предсказания смещения почек

## 📋 Обзор

Это руководство описывает процесс развертывания API для предсказания смещения почек на основе оптимизированной адаптивной модели ансамбля.

---

## 🔧 Требования

### Системные требования
- **Python**: 3.11+
- **RAM**: минимум 2GB (рекомендуется 4GB)
- **CPU**: multi-core для production
- **Диск**: минимум 5GB свободного пространства

### Зависимости
```bash
pip install fastapi uvicorn joblib pandas numpy scikit-learn pydantic requests
```

---

## 🚀 Быстрый старт

### 1. Подготовка окружения

```bash
# Переход в директорию проекта
cd "d:\ml trainer"

# Активация виртуального окружения
venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Запуск API сервера

```bash
# Запуск в режиме разработки
python src/api/kidney_displacement_api.py

# Или с использованием uvicorn
uvicorn src.api.kidney_displacement_api:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Проверка работоспособности

```bash
# Тестирование API
python test_api.py

# Или проверка здоровья
curl http://127.0.0.1:8000/health
```

---

## 🏭 Продакшен-развертывание

### 1. Использование Gunicorn

```bash
# Установка Gunicorn
pip install gunicorn

# Запуск с несколькими worker'ами
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.kidney_displacement_api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
```

### 2. Docker-развертывание

Создайте `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY . .

# Создание необходимых директорий
RUN mkdir -p logs

# Экспорт порта
EXPOSE 8000

# Запуск приложения
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.api.kidney_displacement_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Создайте `docker-compose.yml`:

```yaml
version: '3.8'

services:
  kidney-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Запуск:

```bash
# Сборка и запуск
docker-compose up -d

# Проверка статуса
docker-compose ps

# Просмотр логов
docker-compose logs -f
```

### 3. Nginx конфигурация

```nginx
upstream kidney_api {
    server 127.0.0.1:8000;
    # Для нескольких worker'ов
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://kidney_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Таймауты
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Для Swagger документации
    location /docs {
        proxy_pass http://kidney_api/docs;
    }
}
```

---

## 🔒 Безопасность

### 1. HTTPS с SSL

```bash
# Запуск с HTTPS
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.kidney_displacement_api:app \
    --host 0.0.0.0 \
    --port 8443 \
    --keyfile /path/to/key.pem \
    --certfile /path/to/cert.pem
```

### 2. Аутентификация API

Добавьте middleware для аутентификации:

```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Ваша логика проверки токена
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=403, detail="Invalid token")
    return credentials

@app.post("/predict")
async def predict_protected(request: PredictRequest, token: str = Depends(verify_token)):
    # Ваш код
    pass
```

### 3. Rate limiting

```bash
# Установка rate limiting
pip install slowapi

# В коде API
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: PredictRequest):
    # Ваш код
    pass
```

---

## 📊 Мониторинг и логирование

### 1. Структурированное логирование

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        return json.dumps(log_entry)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.handlers[0].setFormatter(JSONFormatter())
```

### 2. Метрики производительности

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} executed in {end_time - start_time:.3f}s")
        return result
    return wrapper

@app.post("/predict")
@timing_decorator
async def predict(request: PredictRequest):
    # Ваш код
    pass
```

### 3. Health checks

```python
@app.get("/health/detailed")
async def detailed_health():
    """Детальная проверка здоровья"""
    try:
        # Проверка модели
        model_status = "ok" if model_data else "not_loaded"
        
        # Проверка памяти
        import psutil
        memory_usage = psutil.virtual_memory().percent
        
        # Проверка диска
        disk_usage = psutil.disk_usage('/').percent
        
        return {
            "status": "ok",
            "model_status": model_status,
            "memory_usage_percent": memory_usage,
            "disk_usage_percent": disk_usage,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")
```

---

## ⚡ Оптимизация производительности

### 1. Кэширование предсказаний

```python
from functools import lru_cache
import hashlib

def get_cache_key(patient_data: dict) -> str:
    """Создание ключа кэша на основе данных пациента"""
    data_str = json.dumps(patient_data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

@lru_cache(maxsize=1000)
def cached_predict(cache_key: str, patient_data: dict) -> dict:
    """Кэшированное предсказание"""
    return predict_displacement(patient_data)

@app.post("/predict")
async def predict(request: PredictRequest):
    cache_key = get_cache_key(request.patient_data.dict())
    predictions = cached_predict(cache_key, request.patient_data.dict())
    return predictions
```

### 2. Batch processing оптимизация

```python
@app.post("/predict_batch")
async def predict_batch_optimized(request: BatchPredictRequest):
    """Оптимизированное пакетное предсказание"""
    # Создание DataFrame для всех пациентов сразу
    all_data = []
    for patient in request.patients:
        patient_dict = patient["patient_data"].dict()
        all_data.append(patient_dict)
    
    # Обработка всех данных сразу
    df = pd.DataFrame(all_data)
    df = create_features_batch(df)
    
    # Пакетное предсказание
    predictions = predict_batch(df)
    
    return format_batch_response(predictions, request.patients)
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions пример

```yaml
name: Deploy API

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python test_api.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to server
      run: |
        # Ваш скрипт развертывания
        scp -r . user@server:/path/to/app
        ssh user@server 'cd /path/to/app && docker-compose up -d'
```

---

## 🚨 Траблшутинг

### Частые проблемы

1. **Модель не загружается**
   ```bash
   # Проверка пути к модели
   ls -la models/adaptive_ensemble.pkl
   
   # Проверка прав доступа
   chmod 644 models/adaptive_ensemble.pkl
   ```

2. **Ошибка импорта модулей**
   ```bash
   # Проверка PYTHONPATH
   export PYTHONPATH=/path/to/project
   
   # Проверка структуры директорий
   find . -name "*.py" | head -10
   ```

3. **Высокое использование памяти**
   ```bash
   # Мониторинг памяти
   htop
   
   # Перезапуск сервиса
   docker-compose restart
   ```

4. **Медленные ответы**
   ```bash
   # Профилирование
   python -m cProfile -o profile.stats src/api/kidney_displacement_api.py
   
   # Проверка загрузки CPU
   top -p $(pgrep -f kidney_displacement_api)
   ```

### Логирование ошибок

```python
import traceback
from fastapi import Request

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

---

## 📞 Поддержка

### Мониторинг в продакшене

```bash
# Скрипт мониторинга
#!/bin/bash
API_URL="http://localhost:8000/health"

while true; do
    response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)
    if [ $response -ne 200 ]; then
        echo "$(date): API health check failed with status $response"
        # Отправка уведомления
    fi
    sleep 60
done
```

### Алерты

Настройте алерты для:
- Недоступность API (>5 минут)
- Высокое использование памяти (>90%)
- Медленные ответы (>2 секунды)
- Ошибки 5xx в логах

---

## 📚 Дополнительные ресурсы

- [FastAPI документация](https://fastapi.tiangolo.com/)
- [Uvicorn deployment](https://www.uvicorn.org/deployment/)
- [Docker best practices](https://docs.docker.com/develop/dev-best-practices/)
- [Gunicorn configuration](https://docs.gunicorn.org/en/latest/settings.html)

---

**Версия руководства**: 1.0  
**Дата обновления**: 26 марта 2026  
**Поддерживаемые версии Python**: 3.11+
