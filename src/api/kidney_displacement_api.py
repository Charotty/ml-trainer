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
import numpy as np
import pandas as pd
import os
from datetime import datetime
import logging

from src.models.runtime import RuntimePredictor, default_model_path
from src.models.uncertainty import (
    extract_conformal_from_model_data,
    intervals_for_point_predictions,
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical production artifact (honest clinical training with na_trends).
DEFAULT_MODEL_PATH = str(default_model_path())
LEGACY_MODEL_NAME = "adaptive_ensemble.pkl"

# Инициализация FastAPI
app = FastAPI(
    title="Kidney Displacement Prediction API",
    description="API для предсказания смещения почек на основе оптимизированной адаптивной модели",
    version="1.0.0"
)

# Глобальные переменные
runtime_predictor: Optional[RuntimePredictor] = None
model_data = None
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
        "при отсутствии остаётся NaN до persisted imputer; не подменяется 0.",
    )
    age: Optional[float] = Field(
        None,
        description="Возраст пациента, лет. Опционально; missing stays missing (not 50).",
    )
    bmi: Optional[float] = Field(
        None,
        description="Индекс массы тела (BMI). Опционально; missing stays missing (not 25).",
    )
    body_type: Optional[float] = Field(
        None,
        description="Тип телосложения, код (0=нормостеническое, 1=астеническое, "
        "2=гиперстеническое). 0 is a real class, not 'unknown'. Опционально.",
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
    global model_data, feature_names, runtime_predictor

    try:
        model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
        if os.path.basename(model_path) == LEGACY_MODEL_NAME:
            logger.warning(
                "Using legacy model path '%s'. Prefer canonical "
                "'models/adaptive_ensemble_clinical_honest.pkl'.",
                model_path,
            )
        runtime_predictor = RuntimePredictor.load(model_path)
        model_data = runtime_predictor.payload
        feature_names = list(runtime_predictor.bundle.feature_names)
        logger.info(
            "Модель успешно загружена. Признаков: %s, enrichment_mode=%s",
            len(feature_names),
            runtime_predictor.enrichment_mode(),
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
    if runtime_predictor is None or model_data is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    if hasattr(patient_data, "model_dump"):
        patient_dict = patient_data.model_dump()
    else:
        patient_dict = patient_data.dict()

    try:
        predictions = runtime_predictor.predict_row(patient_dict)
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
    """Liveness: report loaded feature_count. Does not name a production winner."""
    n_feat = len(feature_names) if feature_names else 0
    n_targets = len(model_data["models"]) if model_data else 0
    return {
        "status": "ok",
        "model_loaded": model_data is not None,
        "feature_count": n_feat,
        "features_count": n_feat,
        "targets_count": n_targets,
        "timestamp": datetime.now().isoformat(),
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
    optimized_weights = (model_data or {}).get("adaptive_weights") or {}
    training_meta = model_data.get("training_meta") if isinstance(model_data, dict) else None
    
    return {
        "model_info": {
            "name": "Adaptive Ensemble (loaded alias, not a production winner)",
            "version": None,
            "features_count": len(feature_names),
            "targets_count": len(model_data['models']),
            "data_sources": (training_meta or {}).get("data_sources") if isinstance(training_meta, dict) else None,
            "production_winner": False,
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

def prediction_uncertainty_payload(predictions: Dict[str, float]) -> Dict[str, Any]:
    """OOF residual conformal intervals, or an explicit unavailable payload.

    Never returns a magnitude-based fake confidence score.
    """
    conformal = extract_conformal_from_model_data(model_data)
    return intervals_for_point_predictions(predictions, conformal)


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

    uncertainty = prediction_uncertainty_payload(predictions)

    loaded_name = (
        runtime_predictor.model_path.name
        if runtime_predictor is not None
        else "unloaded"
    )
    return PredictResponse(
        success=True,
        predictions=predictions,
        metadata={
            "model_id": loaded_name,
            "features_used": len(feature_names) if feature_names is not None else 0,
            "uncertainty": uncertainty,
            "prediction_confidence": None,
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
    if runtime_predictor is None or model_data is None or feature_names is None:
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
            "model_id": (
                runtime_predictor.model_path.name
                if runtime_predictor is not None
                else None
            ),
            "feature_count": len(feature_names) if feature_names is not None else 0,
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
        "production_winner": False,
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
