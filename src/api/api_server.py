"""
AR-navigation and clinical workflow API (sensors, validation, metrics).

NOT the canonical kidney displacement prediction API. For ML predict
endpoints use ``src/api/kidney_displacement_api.py`` (FastAPI, port 8000).
This module serves a different contract (age/bmi/sex, AR matrices, etc.).
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import numpy as np
import logging
import time
import json
from datetime import datetime
from pathlib import Path
import sys

# Добавляем src в Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from ar_system.kidney_ar_system import KidneyARSystem
from validation.data_validator import DataValidator
from metrics.clinical_metrics import ClinicalMetrics
from system_logging.system_logger import SystemLogger

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic модели для API
class PatientData(BaseModel):
    age: float = Field(..., ge=0, le=100, description="Возраст пациента")
    bmi: float = Field(..., ge=10, le=60, description="Индекс массы тела")
    sex_encoded: float = Field(..., ge=0, le=1, description="Пол (0-женский, 1-мужской)")
    kidney_left_center_x_mm: float = Field(..., description="Координата X левой почки")
    kidney_left_center_y_mm: float = Field(..., description="Координата Y левой почки")
    kidney_left_center_z_mm: float = Field(..., description="Координата Z левой почки")
    kidney_right_center_x_mm: float = Field(..., description="Координата X правой почки")
    kidney_right_center_y_mm: float = Field(..., description="Координата Y правой почки")
    kidney_right_center_z_mm: float = Field(..., description="Координата Z правой почки")
    
    # Опциональные поля
    weight_kg: Optional[float] = Field(None, ge=30, le=200, description="Вес в кг")
    height_m: Optional[float] = Field(None, ge=1.0, le=2.5, description="Рост в метрах")
    body_type_encoded: Optional[float] = Field(None, ge=0, le=3, description="Тип телосложения")
    
    # Дополнительные признаки из КТ
    kidney_left_length_mm: Optional[float] = Field(None, description="Длина левой почки")
    kidney_right_length_mm: Optional[float] = Field(None, description="Длина правой почки")
    kidney_left_bbox_width_mm: Optional[float] = Field(None, description="Ширина левой почки")
    kidney_right_bbox_width_mm: Optional[float] = Field(None, description="Ширина правой почки")
    kidney_left_bbox_height_mm: Optional[float] = Field(None, description="Глубина левой почки")
    kidney_right_bbox_height_mm: Optional[float] = Field(None, description="Глубина правой почки")

class SensorData(BaseModel):
    position: List[float] = Field(..., min_items=3, max_items=3, description="Положение [x, y, z]")
    orientation: List[float] = Field(..., min_items=4, max_items=4, description="Ориентация (quaternion)")
    tilt: Optional[float] = Field(0.0, description="Наклон")
    rotation: Optional[float] = Field(0.0, description="Поворот")

class ARSystemData(BaseModel):
    world_to_ar_matrix: Optional[List[List[float]]] = Field(None, description="Матрица трансформации в AR")
    scale_factor: Optional[float] = Field(1.0, description="Масштабный фактор")

class PredictionRequest(BaseModel):
    patient_data: PatientData
    sensor_data: SensorData
    ar_system_data: Optional[ARSystemData] = ARSystemData()
    patient_id: Optional[str] = Field(None, description="ID пациента для логирования")

class KidneyPosition(BaseModel):
    center: List[float]
    polygon: List[List[float]]
    displacement: List[float]

class PredictionResponse(BaseModel):
    success: bool
    confidence: float
    processing_time_ms: float
    left_kidney: Optional[KidneyPosition]
    right_kidney: Optional[KidneyPosition]
    warnings: Optional[List[str]]
    error: Optional[str]
    details: Optional[List[str]]

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    system_ready: bool
    last_prediction_time: Optional[str]

# Создание FastAPI приложения
app = FastAPI(
    title="Kidney AR Prediction API",
    description="API для предсказания смещения почек в AR-навигации",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS для работы с веб-клиентами
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
kidney_system = None
system_logger = None
data_validator = None
clinical_metrics = None
start_time = time.time()
last_prediction_time = None

@app.on_event("startup")
async def startup_event():
    """Инициализация системы при запуске"""
    global kidney_system, system_logger, data_validator, clinical_metrics
    
    try:
        # Инициализация основной системы
        kidney_system = KidneyARSystem()
        
        # Инициализация логирования
        system_logger = SystemLogger("logs/kidney_ar_api.log")
        
        # Инициализация валидатора
        data_validator = DataValidator()
        
        # Инициализация клинических метрик
        clinical_metrics = ClinicalMetrics()
        
        logger.info("Kidney AR API успешно запущен")
        
    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")
        raise

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о системе"""
    return {
        "message": "Kidney AR Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict", 
            "metrics": "/metrics",
            "version": "/version",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья системы"""
    global start_time, last_prediction_time
    
    uptime = time.time() - start_time
    system_ready = kidney_system is not None
    
    return HealthResponse(
        status="healthy" if system_ready else "unhealthy",
        version="1.0.0",
        uptime_seconds=uptime,
        system_ready=system_ready,
        last_prediction_time=last_prediction_time
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_kidney_displacement(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Основной эндпоинт для предсказания смещения почек
    
    Args:
        request: данные пациента, датчиков и AR системы
        
    Returns:
        PredictionResponse: результаты предсказания
    """
    global last_prediction_time
    
    start_processing = time.time()
    patient_id = request.patient_id or f"patient_{int(time.time())}"
    
    try:
        # 1. Валидация входных данных
        validation_result = data_validator.validate_patient_data(request.patient_data.dict())
        if not validation_result['is_valid']:
            raise HTTPException(
                status_code=400, 
                detail=f"Validation error: {validation_result['errors']}"
            )
        
        # 2. Подготовка данных
        patient_data = request.patient_data.dict()
        sensor_data = request.sensor_data.dict()
        ar_system_data = request.ar_system_data.dict() if request.ar_system_data else {}
        
        # Добавление матрицы трансформации по умолчанию
        if ar_system_data.get('world_to_ar_matrix') is None:
            ar_system_data['world_to_ar_matrix'] = np.eye(4).tolist()
        
        # 3. Предсказание
        result = kidney_system.predict_kidney_displacement(
            patient_data, sensor_data, ar_system_data
        )
        
        # 4. Формирование ответа
        processing_time = (time.time() - start_processing) * 1000  # в мс
        last_prediction_time = datetime.now().isoformat()
        
        if result['success']:
            response = PredictionResponse(
                success=True,
                confidence=result['confidence'],
                processing_time_ms=processing_time,
                left_kidney=KidneyPosition(**result['left_kidney']) if result['left_kidney'] else None,
                right_kidney=KidneyPosition(**result['right_kidney']) if result['right_kidney'] else None,
                warnings=result.get('warnings', [])
            )
        else:
            response = PredictionResponse(
                success=False,
                confidence=0.0,
                processing_time_ms=processing_time,
                left_kidney=None,
                right_kidney=None,
                warnings=[],
                error=result.get('error', 'Unknown error'),
                details=result.get('details', [])
            )
        
        # 5. Логирование в фоновом режиме
        background_tasks.add_task(
            log_prediction,
            patient_id,
            request.patient_data.dict(),
            result,
            processing_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        
        return PredictionResponse(
            success=False,
            confidence=0.0,
            processing_time_ms=(time.time() - start_processing) * 1000,
            left_kidney=None,
            right_kidney=None,
            warnings=[],
            error=str(e),
            details=[]
        )

@app.post("/reset_smoothing")
async def reset_temporal_smoothing():
    """Сброс временного сглаживания"""
    if kidney_system:
        kidney_system.reset_smoothing()
        return {"message": "Temporal smoothing reset successfully"}
    else:
        raise HTTPException(status_code=503, detail="System not ready")

@app.get("/metrics")
async def get_system_metrics():
    """Получение метрик системы"""
    if not clinical_metrics:
        raise HTTPException(status_code=503, detail="Metrics not available")
    
    # В реальной системе здесь были бы накопленные метрики
    return {
        "total_predictions": 0,  # TODO: реализовать счетчик
        "average_confidence": 0.0,
        "average_processing_time_ms": 0.0,
        "success_rate": 0.0,
        "uptime_seconds": time.time() - start_time
    }

@app.get("/version")
async def get_version():
    """Получение информации о версии"""
    return {
        "api_version": "1.0.0",
        "model_version": "model_v1",
        "features_version": "features_v1",
        "pipeline_version": "pipeline_v1"
    }

async def log_prediction(patient_id: str, patient_data: Dict, result: Dict, processing_time: float):
    """Фоновое логирование предсказания"""
    try:
        if system_logger:
            # Логирование входных данных
            system_logger.log_input_data(patient_id, patient_data, processing_time / 1000)
            
            # Логирование результата
            if result['success']:
                system_logger.log_prediction(
                    patient_id,
                    np.array(result.get('left_kidney', {}).get('displacement', [0, 0, 0])),
                    result.get('confidence', 0.0),
                    result.get('constraints_applied', False)
                )
            else:
                system_logger.log_error(
                    patient_id,
                    "prediction_failed",
                    result.get('details', [])
                )
    except Exception as e:
        logger.error(f"Ошибка логирования: {e}")

# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request, call_next):
    start_time_req = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time_req
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

if __name__ == "__main__":
    import uvicorn
    
    # Создание директории для логов
    Path("logs").mkdir(exist_ok=True)
    
    # Запуск сервера
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
