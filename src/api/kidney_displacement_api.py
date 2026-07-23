#!/usr/bin/env python3
"""
Canonical FastAPI kidney displacement prediction API.

This is the production predict contract (``/predict``, ``/model_info``, …).
Legacy Flask API: ``models/phase1/api_kidney_predictor.py`` (deprecated).
AR / sensors workflow lives in ``src/api/api_server.py`` (different contract).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime
import logging

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'phase1'))
from adaptive_ensemble import AdaptiveEnsembleTrainer
from src.features.na_trend_features import NaTrendStore
from src.features.pipeline import predict_targets

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical production artifact (honest clinical training with na_trends).
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "models",
    "adaptive_ensemble_clinical_honest.pkl",
)
LEGACY_MODEL_NAME = "adaptive_ensemble.pkl"

# Инициализация FastAPI
app = FastAPI(
    title="Kidney Displacement Prediction API",
    description="API для предсказания смещения почек на основе оптимизированной адаптивной модели",
    version="1.0.0"
)

# Глобальные переменные
model_data = None
trainer = None
feature_names = None

class PatientData(BaseModel):
    """Модель данных пациента"""
    kidney_left_center_x_rel: float = Field(..., description="Относительная X-координата центра левой почки (мм)")
    kidney_left_center_y_rel: float = Field(..., description="Относительная Y-координата центра левой почки (мм)")
    kidney_left_center_z_rel: float = Field(..., description="Относительная Z-координата центра левой почки (мм)")
    kidney_right_center_x_rel: float = Field(..., description="Относительная X-координата центра правой почки (мм)")
    kidney_right_center_y_rel: float = Field(..., description="Относительная Y-координата центра правой почки (мм)")
    kidney_right_center_z_rel: float = Field(..., description="Относительная Z-координата центра правой почки (мм)")
    kidney_left_length_mm: float = Field(..., description="Длина левой почки (мм)")
    kidney_left_volume_cm3: float = Field(..., description="Объем левой почки (см³)")
    kidney_right_length_mm: float = Field(..., description="Длина правой почки (мм)")
    kidney_right_volume_cm3: float = Field(..., description="Объем правой почки (см³)")
    body_width_mm: float = Field(..., description="Ширина тела пациента (мм)")
    body_depth_mm: float = Field(..., description="Глубина тела пациента (мм)")
    body_area_mm2: float = Field(..., description="Площадь поперечного сечения (мм²)")
    kidney_left_to_spine_distance: float = Field(..., description="Расстояние от левой почки до позвоночника (мм)")
    kidney_right_to_spine_distance: float = Field(..., description="Расстояние от правой почки до позвоночника (мм)")
    kidney_left_to_body_center_distance: float = Field(..., description="Расстояние от левой почки до центра масс тела (мм)")
    kidney_right_to_body_center_distance: float = Field(..., description="Расстояние от правой почки до центра масс тела (мм)")
    spine_center_x: float = Field(0.0, description="X-координата центра позвоночника (мм)")
    spine_center_y: float = Field(0.0, description="Y-координата центра позвоночника (мм)")
    spine_center_z: float = Field(0.0, description="Z-координата центра позвоночника (мм)")
    body_com_x: float = Field(0.0, description="X-координата центра масс тела (мм)")
    body_com_y: float = Field(0.0, description="Y-координата центра масс тела (мм)")
    body_com_z: float = Field(0.0, description="Z-координата центра масс тела (мм)")
    scan_position: Optional[str] = Field(
        None,
        description="DICOM PatientPosition / scan_position (HFS, FFS, ...). "
        "Используется для patient_position_encoded при инжиниринге.",
    )
    sex: Optional[float] = Field(
        None,
        description="Пол пациента, код (1.0 = М, 2.0 = Ж). Опционально — "
        "при отсутствии импутируется медианой обучающей выборки.",
    )
    age: Optional[float] = Field(None, description="Возраст пациента, лет. Опционально.")
    bmi: Optional[float] = Field(None, description="Индекс массы тела (BMI). Опционально.")
    body_type: Optional[float] = Field(
        None,
        description="Тип телосложения, код (0=нормостеническое, 1=астеническое, "
        "2=гиперстеническое). Опционально.",
    )
    has_previous_surgery: Optional[float] = Field(
        None,
        description="Были ли ранее операции (0/1). Опционально.",
    )

class PredictRequest(BaseModel):
    """Запрос на предсказание"""
    patient_data: PatientData

class BatchPatientEntry(BaseModel):
    """Одна запись в пакетном запросе на предсказание."""
    patient_id: Optional[str] = Field(None, description="Идентификатор пациента")
    patient_data: PatientData = Field(..., description="Данные пациента")


class BatchPredictRequest(BaseModel):
    """Запрос на пакетное предсказание.

    Контракт согласован с обработчиком: каждый элемент списка содержит
    ``patient_id`` (опционально) и ``patient_data``. Ранее схема была
    объявлена как ``List[Dict[str, PatientData]]``, что Pydantic не мог
    корректно валидировать и приводило к рассинхрону с фактической
    обработкой.
    """
    patients: List[BatchPatientEntry]

class PredictResponse(BaseModel):
    """Ответ предсказания"""
    success: bool
    predictions: Dict[str, float]
    metadata: Dict[str, Any]

def load_model():
    """Загрузка модели при старте сервера"""
    global model_data, trainer, feature_names
    
    try:
        model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
        if os.path.basename(model_path) == LEGACY_MODEL_NAME:
            logger.warning(
                "Using legacy model path '%s'. Prefer canonical "
                "'models/adaptive_ensemble_clinical_honest.pkl'.",
                model_path,
            )
        model_data = joblib.load(model_path)

        enrichment_mode = model_data.get("enrichment_mode", "projection")
        store_payload = model_data.get("na_trend_store")
        na_trend_store = (
            NaTrendStore.from_dict(store_payload) if store_payload else None
        )
        trainer = AdaptiveEnsembleTrainer(
            enrichment_mode=enrichment_mode,
            na_trend_store=na_trend_store,
            z_head=model_data.get("z_head", "ensemble"),
        )
        
        feature_names = model_data['feature_names']
        trainer.feature_names = feature_names

        logger.info(
            "Модель успешно загружена. Признаков: %s, enrichment_mode=%s, "
            "na_trend_store=%s",
            len(feature_names),
            enrichment_mode,
            na_trend_store is not None,
        )
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return False

def predict_displacement(patient_data: PatientData) -> Dict[str, float]:
    """Выполнение предсказания смещения почек.

    Пайплайн признаков СТРОГО соответствует обучающему:
      base + engineered + cross features -> imputer.transform -> scaler.transform -> model.predict

    `imputer` в пакете модели опционален (для обратной совместимости
    со старыми pkl, сохранёнными до добавления imputer в save_model).
    Если его нет, выводим warning один раз и пропускаем шаг — но это
    означает, что любые NaN после feature-engineering приведут к NaN в
    предсказании.

    Семантика HTTP-кодов:
      - 400: проблема в данных клиента (невалидные/отсутствующие признаки);
      - 503: модель не загружена (артефакты не доступны);
      - 500: непредвиденная серверная ошибка.
    """
    if model_data is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    if hasattr(patient_data, "model_dump"):
        patient_dict = patient_data.model_dump()
    else:
        patient_dict = patient_data.dict()

    try:
        predictions = predict_targets(trainer, model_data, patient_dict)
        return predictions
    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Несовместимая форма входных данных при предсказании")
        raise HTTPException(
            status_code=400,
            detail=f"Несовместимые входные данные: {exc}",
        )
    except Exception as exc:
        logger.exception("Внутренняя ошибка при выполнении предсказания")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка предсказания: {exc}",
        )

@app.on_event("startup")
async def startup_event():
    """Инициализация при старте сервера"""
    success = load_model()
    if not success:
        raise RuntimeError("Не удалось загрузить модель")

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "ok",
        "model_version": "optimized_adaptive_ensemble_v1.0",
        "features_count": len(feature_names) if feature_names else 0,
        "targets_count": len(model_data['models']) if model_data else 0,
        "timestamp": datetime.now().isoformat()
    }

def _performance_from_training_meta(payload: dict) -> dict:
    """Prefer metrics stored in training_meta; never invent hardcoded MAE."""
    meta = payload.get("training_meta")
    if not isinstance(meta, dict):
        return {
            "status": "unavailable",
            "average_mae_mm": None,
            "average_r2": None,
            "accuracy_5mm": None,
            "accuracy_10mm": None,
            "detail": "training_meta missing from model payload",
        }

    perf = meta.get("performance")
    if isinstance(perf, dict) and any(
        perf.get(k) is not None
        for k in ("average_mae_mm", "mae_avg_mm", "average_r2", "r2_avg")
    ):
        return {
            "status": "from_training_meta",
            "average_mae_mm": perf.get("average_mae_mm", perf.get("mae_avg_mm")),
            "average_r2": perf.get("average_r2", perf.get("r2_avg")),
            "accuracy_5mm": perf.get("accuracy_5mm", perf.get("within_5mm_ratio")),
            "accuracy_10mm": perf.get("accuracy_10mm", perf.get("within_10mm_ratio")),
        }

    return {
        "status": "unavailable",
        "average_mae_mm": None,
        "average_r2": None,
        "accuracy_5mm": None,
        "accuracy_10mm": None,
        "detail": "training_meta present but no performance metrics",
        "training_meta_keys": sorted(meta.keys()),
    }


@app.get("/model_info")
async def get_model_info():
    """Детальная информация о модели"""
    if not model_data:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    # Получение оптимизированных весов
    optimized_weights = getattr(trainer, '_optimized_weights', {})
    training_meta = model_data.get("training_meta") if isinstance(model_data, dict) else None
    
    return {
        "model_info": {
            "name": "Optimized Adaptive Ensemble",
            "version": "1.0",
            "features_count": len(feature_names),
            "targets_count": len(model_data['models']),
            "data_sources": "DICOMS+Vybor+KiTS19",
            "performance": _performance_from_training_meta(model_data),
            "training_meta": training_meta,
            "feature_types": {
                "base_features": 23,
                "engineered_features": 13,
                "cross_features": 15
            },
            "optimized_weights": optimized_weights
        },
        "feature_names": feature_names
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Предсказание смещения почек."""
    try:
        predictions = predict_displacement(request.patient_data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Неожиданная ошибка в эндпоинте /predict")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {exc}",
        )

    confidence = {
        target: min(0.95, max(0.5, 1.0 - abs(pred) / 50.0))
        for target, pred in predictions.items()
    }

    return PredictResponse(
        success=True,
        predictions=predictions,
        metadata={
            "model_version": "optimized_adaptive_ensemble_v1.0",
            "features_used": len(feature_names),
            "prediction_confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        },
    )

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictRequest):
    """Пакетное предсказание для нескольких пациентов.

    Поведение HTTP-кодов:
      - 200: хотя бы один прогноз выполнен; в теле ответа `success`
        отражает фактический результат (`True` тогда и только тогда, когда
        ВСЕ пациенты обработаны без ошибок);
      - 400: пустой список пациентов или невалидный формат запроса;
      - 503: модель не загружена;
      - 500: непредвиденная серверная ошибка.
    """
    if model_data is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    if not request.patients:
        raise HTTPException(status_code=400, detail="Список пациентов пуст")

    results: List[Dict[str, Any]] = []
    successful_predictions = 0

    for idx, entry in enumerate(request.patients, start=1):
        patient_id = entry.patient_id or f"patient_{idx}"
        try:
            predictions = predict_displacement(entry.patient_data)
            results.append({"patient_id": patient_id, "predictions": predictions})
            successful_predictions += 1
        except HTTPException as http_exc:
            logger.warning(
                "Ошибка предсказания для пациента %s: %s",
                patient_id,
                http_exc.detail,
            )
            results.append({
                "patient_id": patient_id,
                "error": http_exc.detail,
                "status_code": http_exc.status_code,
            })
        except Exception as exc:
            logger.exception("Неожиданная ошибка для пациента %s", patient_id)
            results.append({
                "patient_id": patient_id,
                "error": str(exc),
                "status_code": 500,
            })

    total = len(request.patients)
    return {
        "success": successful_predictions == total,
        "results": results,
        "metadata": {
            "total_patients": total,
            "successful_predictions": successful_predictions,
            "failed_predictions": total - successful_predictions,
            "model_version": "optimized_adaptive_ensemble_v1.0",
            "timestamp": datetime.now().isoformat(),
        },
    }

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о canonical predict-API."""
    return {
        "message": "Kidney Displacement Prediction API (canonical)",
        "version": "1.0.0",
        "service_role": "kidney_displacement_prediction",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model_info",
        "endpoints": {
            "predict": "POST /predict",
            "predict_batch": "POST /predict_batch",
        },
        "deprecated_alternatives": {
            "flask_legacy": "models/phase1/api_kidney_predictor.py (do not use for new integrations)",
        },
        "related_not_canonical": {
            "ar_navigation": "src/api/api_server.py (different domain: AR + sensors, not displacement predict)",
        },
    }

if __name__ == "__main__":
    import uvicorn
    
    # Загрузка модели перед запуском
    if not load_model():
        print("❌ Не удалось загрузить модель. Выход.")
        sys.exit(1)
    
    print("🚀 Запуск API сервера...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
