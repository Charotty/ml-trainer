#!/usr/bin/env python3
"""
API сервер для предсказания смещения почек на основе оптимизированной адаптивной модели
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime
import logging

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'phase1'))
from adaptive_ensemble import AdaptiveEnsembleTrainer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class PredictRequest(BaseModel):
    """Запрос на предсказание"""
    patient_data: PatientData

class BatchPredictRequest(BaseModel):
    """Запрос на пакетное предсказание"""
    patients: List[Dict[str, PatientData]]

class PredictResponse(BaseModel):
    """Ответ предсказания"""
    success: bool
    predictions: Dict[str, float]
    metadata: Dict[str, any]

def load_model():
    """Загрузка модели при старте сервера"""
    global model_data, trainer, feature_names
    
    try:
        # Загрузка модели
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'adaptive_ensemble.pkl')
        model_data = joblib.load(model_path)
        
        # Инициализация тренера для создания признаков
        trainer = AdaptiveEnsembleTrainer()
        
        # Получение имен признаков
        feature_names = model_data['feature_names']
        
        logger.info(f"Модель успешно загружена. Признаков: {len(feature_names)}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return False

def create_features(patient_data: PatientData) -> pd.DataFrame:
    """Создание всех необходимых признаков из базовых данных"""
    # Преобразуем в DataFrame
    data_dict = patient_data.dict()
    df = pd.DataFrame([data_dict])
    
    # Создаем инженерные признаки
    df = trainer._create_engineered_features(df)
    
    # Создаем cross-features
    df = trainer._create_cross_features(df)
    
    return df

def predict_displacement(patient_data: PatientData) -> Dict[str, float]:
    """Выполнение предсказания смещения почек"""
    try:
        # Создание признаков
        df = create_features(patient_data)
        
        # Проверка наличия всех необходимых признаков
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Отсутствующие признаки: {missing_features}")
        
        # Подготовка данных
        X = df[feature_names].values
        X_scaled = model_data['scaler'].transform(X)
        
        # Предсказание
        predictions = {}
        for target_name, model in model_data['models'].items():
            pred = model.predict(X_scaled)[0]
            predictions[target_name] = float(pred)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

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

@app.get("/model_info")
async def get_model_info():
    """Детальная информация о модели"""
    if not model_data:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    # Получение оптимизированных весов
    optimized_weights = getattr(trainer, '_optimized_weights', {})
    
    return {
        "model_info": {
            "name": "Optimized Adaptive Ensemble",
            "version": "1.0",
            "features_count": len(feature_names),
            "targets_count": len(model_data['models']),
            "data_sources": "DICOMS+Vybor+KiTS19",
            "performance": {
                "average_mae_mm": 2.140,
                "average_r2": 0.139,
                "accuracy_5mm": 89.2,
                "accuracy_10mm": 97.6
            },
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
    """Предсказание смещения почек"""
    try:
        # Выполнение предсказания
        predictions = predict_displacement(request.patient_data)
        
        # Расчет доверительных интервалов (упрощенный)
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
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictRequest):
    """Пакетное предсказание для нескольких пациентов"""
    try:
        results = []
        successful_predictions = 0
        
        for patient_data in request.patients:
            try:
                patient_id = patient_data.get("patient_id", f"patient_{len(results)+1}")
                patient_info = patient_data["patient_data"]
                
                # Выполнение предсказания
                predictions = predict_displacement(PatientData(**patient_info))
                
                results.append({
                    "patient_id": patient_id,
                    "predictions": predictions
                })
                successful_predictions += 1
                
            except Exception as e:
                logger.error(f"Ошибка предсказания для пациента {patient_data.get('patient_id', 'unknown')}: {e}")
                results.append({
                    "patient_id": patient_data.get("patient_id", f"patient_{len(results)+1}"),
                    "error": str(e)
                })
        
        return {
            "success": True,
            "results": results,
            "metadata": {
                "total_patients": len(request.patients),
                "successful_predictions": successful_predictions,
                "failed_predictions": len(request.patients) - successful_predictions,
                "model_version": "optimized_adaptive_ensemble_v1.0",
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Ошибка пакетного предсказания: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка пакетного предсказания: {str(e)}")

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией"""
    return {
        "message": "Kidney Displacement Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model_info"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Загрузка модели перед запуском
    if not load_model():
        print("❌ Не удалось загрузить модель. Выход.")
        sys.exit(1)
    
    print("🚀 Запуск API сервера...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
