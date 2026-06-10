#!/usr/bin/env python3
"""
[DEPRECATED] Flask REST API for Kidney Displacement Predictor.

Canonical predict-API for this project is the FastAPI service in
``src/api/kidney_displacement_api.py``. Reasons:

  * Production stack standardised on FastAPI + Pydantic + Uvicorn.
  * FastAPI service is the one that ships the corrected feature
    pipeline (imputer + scaler) wired by the audit fixes 1.1, 1.2, 1.7.
  * This Flask file exposes a different feature contract (30 fields
    with ``_rel``/``_norm`` suffixes + ``patient_position_encoded``)
    that no longer matches the trained ensemble.

This module is kept ONLY for legacy clients. New integrations MUST use
the FastAPI service. If you find yourself editing this file in 2026 or
later, prefer porting the change to ``src/api/kidney_displacement_api.py``
instead.
"""

import sys
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.phase1_schema import BASE_FEATURES

warnings.warn(
    "models.phase1.api_kidney_predictor is deprecated; use the FastAPI "
    "service at src/api/kidney_displacement_api.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import traceback
from pathlib import Path
import json

from kidney_displacement_predictor import KidneyDisplacementPredictor, PredictionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global predictor instance
predictor = None
model_loaded = False

def load_model():
    """Load the trained model"""
    global predictor, model_loaded
    
    try:
        model_path = Path("models")
        if model_path.exists():
            predictor = KidneyDisplacementPredictor(model_path=str(model_path))
            predictor.load_model(str(model_path))
            model_loaded = True
            logger.info("Model loaded successfully")
            return True
        else:
            logger.warning("Model directory not found. Please train the model first.")
            return False
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def validate_request_data(data):
    """Validate request data"""
    errors = []
    
    if not isinstance(data, dict):
        errors.append("Request data must be a JSON object")
        return False, errors
    
    # Check required features
    required_features = list(BASE_FEATURES)
    
    missing_features = set(required_features) - set(data.keys())
    if missing_features:
        errors.append(f"Missing required features: {list(missing_features)}")
    
    # Check data types
    for feature in required_features:
        if feature in data:
            try:
                float(data[feature])
            except (ValueError, TypeError):
                errors.append(f"Feature '{feature}' must be numeric")
    
    return len(errors) == 0, errors

def format_prediction_response(result: PredictionResult) -> dict:
    """Format prediction result for API response"""
    response = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "coordinate_system": {
            "type": "DICOM",
            "orientation": "RAS",
            "origin": [0.0, 0.0, 0.0],
            "spacing": [1.0, 1.0, 1.0],
            "description": "Patient-based coordinate system in Right-Anterior-Superior orientation"
        },
        "reference_point": {
            "type": "spine_center",
            "anatomical_level": "L3 vertebral body",
            "coordinates": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "unit": "mm"
            },
            "description": "Center of spinal canal at L3 vertebral level, used as reference for displacement calculations"
        },
        "predictions": {},
        "displacement_vectors": {},
        "confidence_intervals": {},
        "model_confidence": {},
        "metadata": result.prediction_metadata
    }
    
    # Extract kidney displacement values
    left_x = result.predictions.get('kidney_left_delta_x', 0.0)
    left_y = result.predictions.get('kidney_left_delta_y', 0.0)
    left_z = result.predictions.get('kidney_left_delta_z', 0.0)
    right_x = result.predictions.get('kidney_right_delta_x', 0.0)
    right_y = result.predictions.get('kidney_right_delta_y', 0.0)
    right_z = result.predictions.get('kidney_right_delta_z', 0.0)
    
    # Calculate vector magnitudes and directions
    import numpy as np
    
    left_vector = [left_x, left_y, left_z]
    right_vector = [right_x, right_y, right_z]
    left_magnitude = np.linalg.norm(left_vector)
    right_magnitude = np.linalg.norm(right_vector)
    
    def get_direction_description(vector):
        """Get anatomical direction description from vector components"""
        x, y, z = vector
        directions = []
        
        # X-axis direction
        if abs(x) > 0.1:
            directions.append("right" if x > 0 else "left")
        
        # Y-axis direction
        if abs(y) > 0.1:
            directions.append("anterior" if y > 0 else "posterior")
        
        # Z-axis direction
        if abs(z) > 0.1:
            directions.append("superior" if z > 0 else "inferior")
        
        return "-".join(directions) if directions else "minimal"
    
    # Format predictions with full displacement description
    response["predictions"]["left_kidney"] = {
        "displacement": {
            "x": round(left_x, 3),
            "y": round(left_y, 3),
            "z": round(left_z, 3),
            "unit": "mm"
        },
        "vector": {
            "components": [round(left_x, 3), round(left_y, 3), round(left_z, 3)],
            "magnitude": round(left_magnitude, 3),
            "unit": "mm",
            "direction": get_direction_description(left_vector)
        }
    }
    
    response["predictions"]["right_kidney"] = {
        "displacement": {
            "x": round(right_x, 3),
            "y": round(right_y, 3),
            "z": round(right_z, 3),
            "unit": "mm"
        },
        "vector": {
            "components": [round(right_x, 3), round(right_y, 3), round(right_z, 3)],
            "magnitude": round(right_magnitude, 3),
            "unit": "mm",
            "direction": get_direction_description(right_vector)
        }
    }
    
    # Add displacement vectors for compatibility
    response["displacement_vectors"]["left_kidney"] = {
        "components": [round(left_x, 3), round(left_y, 3), round(left_z, 3)],
        "magnitude": round(left_magnitude, 3),
        "unit": "mm",
        "direction": get_direction_description(left_vector)
    }
    
    response["displacement_vectors"]["right_kidney"] = {
        "components": [round(right_x, 3), round(right_y, 3), round(right_z, 3)],
        "magnitude": round(right_magnitude, 3),
        "unit": "mm",
        "direction": get_direction_description(right_vector)
    }
    
    # Add clinical metrics
    asymmetry_magnitude = abs(left_magnitude - right_magnitude)
    max_displacement = max(left_magnitude, right_magnitude)
    
    response["clinical_metrics"] = {
        "total_displacement": {
            "left_kidney": round(left_magnitude, 3),
            "right_kidney": round(right_magnitude, 3),
            "unit": "mm"
        },
        "asymmetry": {
            "magnitude": round(asymmetry_magnitude, 3),
            "direction": "left_greater" if left_magnitude > right_magnitude else "right_greater",
            "clinical_significance": "minimal" if asymmetry_magnitude < 1.0 else "moderate" if asymmetry_magnitude < 3.0 else "significant"
        },
        "risk_assessment": {
            "category": "low" if max_displacement < 5.0 else "moderate" if max_displacement < 10.0 else "high",
            "probability": round(min(max_displacement / 10.0, 1.0), 3),
            "recommendation": "standard follow-up" if max_displacement < 5.0 else "enhanced monitoring" if max_displacement < 10.0 else "urgent evaluation"
        }
    }
    
    # Format confidence intervals for each kidney
    response["confidence_intervals"]["left_kidney"] = {}
    response["confidence_intervals"]["right_kidney"] = {}
    
    for target in ['kidney_left_delta_x', 'kidney_left_delta_y', 'kidney_left_delta_z']:
        if target in result.confidence_intervals:
            ci_low, ci_high = result.confidence_intervals[target]
            axis = target.split('_')[-1]
            response["confidence_intervals"]["left_kidney"][axis] = {
                "lower": round(ci_low, 3),
                "upper": round(ci_high, 3),
                "level": "95%"
            }
    
    for target in ['kidney_right_delta_x', 'kidney_right_delta_y', 'kidney_right_delta_z']:
        if target in result.confidence_intervals:
            ci_low, ci_high = result.confidence_intervals[target]
            axis = target.split('_')[-1]
            response["confidence_intervals"]["right_kidney"][axis] = {
                "lower": round(ci_low, 3),
                "upper": round(ci_high, 3),
                "level": "95%"
            }
    
    # Add model confidence
    response["model_confidence"]["left_kidney"] = {}
    response["model_confidence"]["right_kidney"] = {}
    
    for target in ['kidney_left_delta_x', 'kidney_left_delta_y', 'kidney_left_delta_z']:
        if target in result.model_confidence:
            axis = target.split('_')[-1]
            response["model_confidence"]["left_kidney"][axis] = round(result.model_confidence[target], 3)
    
    for target in ['kidney_right_delta_x', 'kidney_right_delta_y', 'kidney_right_delta_z']:
        if target in result.model_confidence:
            axis = target.split('_')[-1]
            response["model_confidence"]["right_kidney"][axis] = round(result.model_confidence[target], 3)
    
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "version": "1.0.0"
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if not model_loaded:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 503
    
    info = predictor.get_model_info()
    return jsonify(info)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict kidney displacement"""
    if not model_loaded:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Please train and load the model first."
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        # Validate request data
        is_valid, errors = validate_request_data(data)
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": "Invalid request data",
                "errors": errors
            }), 400
        
        # Make prediction
        logger.info(f"Making prediction for patient data")
        result = predictor.predict(data)
        
        # Format response
        response = format_prediction_response(result)
        
        logger.info(f"Prediction completed successfully")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "status": "error",
            "message": "Internal server error during prediction",
            "details": str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict kidney displacement for multiple patients"""
    if not model_loaded:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Please train and load the model first."
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        if not data or 'patients' not in data:
            return jsonify({
                "status": "error",
                "message": "Request must contain 'patients' array"
            }), 400
        
        patients = data['patients']
        if not isinstance(patients, list):
            return jsonify({
                "status": "error",
                "message": "'patients' must be an array"
            }), 400
        
        if len(patients) > 100:
            return jsonify({
                "status": "error",
                "message": "Maximum 100 patients per batch request"
            }), 400
        
        # Validate each patient
        for i, patient in enumerate(patients):
            is_valid, errors = validate_request_data(patient)
            if not is_valid:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid data for patient {i+1}",
                    "errors": errors
                }), 400
        
        # Make batch predictions
        logger.info(f"Making batch prediction for {len(patients)} patients")
        results = predictor.predict_batch(patients)
        
        # Format response
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "batch_size": len(patients),
            "results": []
        }
        
        for i, result in enumerate(results):
            patient_result = {
                "patient_index": i,
                "predictions": {},
                "confidence_intervals": {},
                "model_confidence": {}
            }
            
            for target, pred in result.predictions.items():
                patient_result["predictions"][target] = round(pred, 3)
                
                ci_low, ci_high = result.confidence_intervals[target]
                patient_result["confidence_intervals"][target] = {
                    "lower": round(ci_low, 3),
                    "upper": round(ci_high, 3)
                }
                
                patient_result["model_confidence"][target] = round(result.model_confidence[target], 3)
            
            response["results"].append(patient_result)
        
        logger.info(f"Batch prediction completed for {len(patients)} patients")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "status": "error",
            "message": "Internal server error during batch prediction",
            "details": str(e)
        }), 500

@app.route('/validate', methods=['POST'])
def validate():
    """Validate patient data without making predictions"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        # Validate request data
        is_valid, errors = validate_request_data(data)
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "valid": is_valid,
            "errors": errors
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        
        return jsonify({
            "status": "error",
            "message": "Internal server error during validation",
            "details": str(e)
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Get required features list"""
    if not model_loaded:
        return jsonify({
            "status": "error",
            "message": "Model not loaded"
        }), 503
    
    return jsonify({
        "status": "success",
        "required_features": predictor.required_features,
        "target_columns": predictor.target_columns,
        "feature_count": len(predictor.required_features),
        "target_count": len(predictor.target_columns)
    })

@app.route('/docs', methods=['GET'])
def api_docs():
    """API documentation"""
    docs = {
        "title": "Kidney Displacement Predictor API",
        "version": "1.0.0",
        "description": "REST API for predicting kidney displacement using Adaptive Ensemble model",
        "endpoints": {
            "GET /health": "Health check endpoint",
            "GET /model/info": "Get model information and performance metrics",
            "POST /predict": "Predict kidney displacement for single patient",
            "POST /predict/batch": "Predict kidney displacement for multiple patients",
            "POST /validate": "Validate patient data format",
            "GET /features": "Get list of required features",
            "GET /docs": "API documentation"
        },
        "example_request": {
            "kidney_left_center_x_rel": 0.5,
            "kidney_left_center_y_rel": 0.3,
            "kidney_left_center_z_rel": 0.2,
            "kidney_left_length_mm": 100,
            "kidney_left_volume_cm3": 150,
            "body_width_mm": 400,
            "patient_position_encoded": 0
        },
        "example_response": {
            "status": "success",
            "predictions": {
                "kidney_left_delta_x": {"value": 1.234, "unit": "mm"},
                "kidney_left_delta_y": {"value": 0.567, "unit": "mm"}
            },
            "confidence_intervals": {
                "kidney_left_delta_x": {"lower": 0.5, "upper": 2.0, "level": "95%"}
            },
            "model_confidence": {
                "kidney_left_delta_x": 0.85
            }
        }
    }
    
    return jsonify(docs)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "GET /model/info", 
            "POST /predict",
            "POST /predict/batch",
            "POST /validate",
            "GET /features",
            "GET /docs"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500

def main():
    """Main function to run the API"""
    print("="*60)
    print("KIDNEY DISPLACEMENT PREDICTOR API")
    print("="*60)
    
    # Load model
    print("Loading model...")
    if load_model():
        print("✅ Model loaded successfully")
    else:
        print("❌ Failed to load model")
        print("Please train the model first using kidney_displacement_predictor.py")
        return
    
    # Run API
    print("\nStarting API server...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /model/info - Model information")
    print("  POST /predict - Single prediction")
    print("  POST /predict/batch - Batch prediction")
    print("  POST /validate - Validate data")
    print("  GET  /features - Required features")
    print("  GET  /docs - API documentation")
    print("\nAPI will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")

if __name__ == "__main__":
    main()
