#!/usr/bin/env python3
"""
Unit Tests for Kidney Displacement Predictor
Phase 1: Testing and Validation
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from kidney_displacement_predictor import (
    KidneyDisplacementPredictor, 
    PredictionResult, 
    ModelMetadata
)

class TestKidneyDisplacementPredictor(unittest.TestCase):
    """Test suite for KidneyDisplacementPredictor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.predictor = KidneyDisplacementPredictor(model_path=self.temp_dir)
        
        # Sample test data
        self.sample_patient = {
            'kidney_left_center_x_rel': 0.5,
            'kidney_left_center_y_rel': 0.3,
            'kidney_left_center_z_rel': 0.2,
            'kidney_left_center_x_norm': 0.45,
            'kidney_left_center_y_norm': 0.35,
            'kidney_left_center_z_norm': 0.25,
            'kidney_right_center_x_rel': 0.6,
            'kidney_right_center_y_rel': 0.4,
            'kidney_right_center_z_rel': 0.3,
            'kidney_right_center_x_norm': 0.55,
            'kidney_right_center_y_norm': 0.45,
            'kidney_right_center_z_norm': 0.35,
            'kidney_left_length_mm': 100,
            'kidney_left_volume_cm3': 150,
            'kidney_right_length_mm': 110,
            'kidney_right_volume_cm3': 160,
            'body_width_mm': 400,
            'body_depth_mm': 200,
            'body_area_mm2': 80000,
            'kidney_left_to_spine_distance': 50,
            'kidney_right_to_spine_distance': 60,
            'kidney_left_to_body_center_distance': 80,
            'kidney_right_to_body_center_distance': 90,
            'spine_center_x': 200,
            'spine_center_y': 150,
            'spine_center_z': 100,
            'body_com_x': 210,
            'body_com_y': 155,
            'body_com_z': 105,
            'patient_position_encoded': 0
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test predictor initialization"""
        # Test with default path
        predictor = KidneyDisplacementPredictor()
        self.assertIsNotNone(predictor.model_path)
        self.assertFalse(predictor.is_trained)
        self.assertIsNone(predictor.metadata)
        
        # Test with custom path
        custom_predictor = KidneyDisplacementPredictor(model_path=self.temp_dir)
        self.assertEqual(custom_predictor.model_path, Path(self.temp_dir))
    
    def test_feature_validation(self):
        """Test input validation"""
        # Test valid input
        is_valid, errors = self.predictor.validate_input(self.sample_patient)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test missing features
        invalid_patient = self.sample_patient.copy()
        del invalid_patient['kidney_left_center_x_rel']
        is_valid, errors = self.predictor.validate_input(invalid_patient)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('kidney_left_center_x_rel' in error for error in errors))
        
        # Test NaN values
        nan_patient = self.sample_patient.copy()
        nan_patient['kidney_left_center_x_rel'] = np.nan
        is_valid, errors = self.predictor.validate_input(nan_patient)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_prediction_without_training(self):
        """Test prediction without training should raise error"""
        with self.assertRaises(ValueError):
            self.predictor.predict(self.sample_patient)
    
    def test_model_info_untrained(self):
        """Test model info for untrained model"""
        info = self.predictor.get_model_info()
        self.assertEqual(info['status'], 'not_trained')
        self.assertIn('message', info)
    
    def test_save_load_model_untrained(self):
        """Test saving/loading untrained model should raise error"""
        with self.assertRaises(ValueError):
            self.predictor.save_model()
        
        with self.assertRaises(FileNotFoundError):
            self.predictor.load_model(self.temp_dir)
    
    def test_prediction_result_structure(self):
        """Test PredictionResult data structure"""
        # Create a mock prediction result
        predictions = {'kidney_left_delta_x': 1.5}
        confidence_intervals = {'kidney_left_delta_x': (1.0, 2.0)}
        model_confidence = {'kidney_left_delta_x': 0.8}
        metadata = {'model_version': '1.0.0'}
        
        result = PredictionResult(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            model_confidence=model_confidence,
            prediction_metadata=metadata
        )
        
        self.assertEqual(result.predictions, predictions)
        self.assertEqual(result.confidence_intervals, confidence_intervals)
        self.assertEqual(result.model_confidence, model_confidence)
        self.assertEqual(result.prediction_metadata, metadata)
    
    def test_model_metadata_structure(self):
        """Test ModelMetadata data structure"""
        metadata = ModelMetadata(
            model_name="TestModel",
            version="1.0.0",
            training_date="2024-01-01",
            dataset_info={"total_cases": 100},
            performance_metrics={"mae": 2.5},
            feature_names=["feature1"],
            target_names=["target1"]
        )
        
        self.assertEqual(metadata.model_name, "TestModel")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.dataset_info["total_cases"], 100)

class TestKidneyPredictorIntegration(unittest.TestCase):
    """Integration tests for KidneyDisplacementPredictor"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.predictor = KidneyDisplacementPredictor(model_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skip("Requires training data - run manually")
    def test_full_training_cycle(self):
        """Test complete training cycle"""
        # This test requires actual training data
        # Skip for automated testing, run manually
        
        # Train model
        metrics = self.predictor.train(save_model=True)
        
        # Check training completed
        self.assertTrue(self.predictor.is_trained)
        self.assertIsNotNone(self.predictor.metadata)
        
        # Check metrics
        self.assertIn('average_mae', metrics)
        self.assertGreater(metrics['average_mae'], 0)
        
        # Check model info
        info = self.predictor.get_model_info()
        self.assertEqual(info['status'], 'trained')
        
        # Test prediction
        sample_patient = {
            'kidney_left_center_x_rel': 0.5,
            'kidney_left_center_y_rel': 0.3,
            # ... add all required features
            'patient_position_encoded': 0
        }
        
        result = self.predictor.predict(sample_patient)
        
        # Check result structure
        self.assertIsInstance(result, PredictionResult)
        self.assertIn('kidney_left_delta_x', result.predictions)
        self.assertIn('kidney_left_delta_x', result.confidence_intervals)
        self.assertIn('kidney_left_delta_x', result.model_confidence)
    
    def test_model_save_load_cycle(self):
        """Test model saving and loading cycle"""
        # Create mock metadata
        self.predictor.metadata = ModelMetadata(
            model_name="TestModel",
            version="1.0.0",
            training_date="2024-01-01",
            dataset_info={"total_cases": 100},
            performance_metrics={"mae": 2.5},
            feature_names=["feature1"],
            target_names=["target1"]
        )
        self.predictor.is_trained = True
        
        # Save model
        save_path = self.predictor.save_model()
        self.assertTrue(Path(save_path).exists())
        
        # Check files exist
        model_file = Path(save_path) / "kidney_displacement_model.pkl"
        metadata_file = Path(save_path) / "model_metadata.json"
        feature_file = Path(save_path) / "features.json"
        
        self.assertTrue(model_file.exists())
        self.assertTrue(metadata_file.exists())
        self.assertTrue(feature_file.exists())
        
        # Load model
        new_predictor = KidneyDisplacementPredictor(model_path=self.temp_dir)
        new_predictor.load_model(save_path)
        
        # Check loaded model
        self.assertTrue(new_predictor.is_trained)
        self.assertIsNotNone(new_predictor.metadata)
        self.assertEqual(new_predictor.metadata.version, "1.0.0")

class TestPredictionMethods(unittest.TestCase):
    """Test prediction methods"""
    
    def setUp(self):
        """Set up prediction test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.predictor = KidneyDisplacementPredictor(model_path=self.temp_dir)
        
        # Mock trained state
        self.predictor.is_trained = True
        self.predictor.metadata = ModelMetadata(
            model_name="TestModel",
            version="1.0.0",
            training_date="2024-01-01",
            dataset_info={"total_cases": 100},
            performance_metrics={"mae": 2.5},
            feature_names=self.predictor.required_features,
            target_names=self.predictor.target_columns
        )
    
    def tearDown(self):
        """Clean up prediction test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation"""
        pred = 2.0
        target_name = 'kidney_left_delta_x'
        
        ci_low, ci_high = self.predictor._calculate_confidence_interval(pred, target_name)
        
        # Check that interval is symmetric around prediction
        self.assertAlmostEqual((ci_low + ci_high) / 2, pred, places=2)
        self.assertLess(ci_low, pred)
        self.assertGreater(ci_high, pred)
    
    def test_model_confidence_scores(self):
        """Test model confidence scores"""
        # Test all targets
        for target in self.predictor.target_columns:
            confidence = self.predictor._get_model_confidence(target)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
        
        # Test specific targets
        right_z_confidence = self.predictor._get_model_confidence('kidney_right_delta_z')
        left_x_confidence = self.predictor._get_model_confidence('kidney_left_delta_x')
        
        # Right Z should have higher confidence (easier target)
        self.assertGreater(right_z_confidence, left_x_confidence)
    
    def test_training_metrics_calculation(self):
        """Test training metrics calculation"""
        mock_results = {
            'target1': {
                'Adaptive_MAE': 2.0,
                'Adaptive_RMSE': 3.0,
                'Adaptive_R2': 0.5,
                'Adaptive_Error_5mm': 85.0,
                'Adaptive_Error_10mm': 95.0,
                'Standard_MAE': 2.1,
                'Improvement_vs_Best': 5.0
            },
            'target2': {
                'Adaptive_MAE': 3.0,
                'Adaptive_RMSE': 4.0,
                'Adaptive_R2': 0.3,
                'Adaptive_Error_5mm': 75.0,
                'Adaptive_Error_10mm': 90.0,
                'Standard_MAE': 3.2,
                'Improvement_vs_Best': -2.0
            }
        }
        
        metrics = self.predictor._calculate_training_metrics(mock_results)
        
        self.assertEqual(metrics['average_mae'], 2.5)
        self.assertEqual(metrics['average_rmse'], 3.5)
        self.assertEqual(metrics['average_r2'], 0.4)
        self.assertEqual(metrics['min_mae'], 2.0)
        self.assertEqual(metrics['max_mae'], 3.0)

def run_performance_benchmark():
    """Run performance benchmark for the predictor"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    predictor = KidneyDisplacementPredictor()
    
    # Generate test data
    test_patients = []
    for i in range(10):
        patient = {
            'kidney_left_center_x_rel': np.random.uniform(0, 1),
            'kidney_left_center_y_rel': np.random.uniform(0, 1),
            'kidney_left_center_z_rel': np.random.uniform(0, 1),
            'kidney_left_center_x_norm': np.random.uniform(0, 1),
            'kidney_left_center_y_norm': np.random.uniform(0, 1),
            'kidney_left_center_z_norm': np.random.uniform(0, 1),
            'kidney_right_center_x_rel': np.random.uniform(0, 1),
            'kidney_right_center_y_rel': np.random.uniform(0, 1),
            'kidney_right_center_z_rel': np.random.uniform(0, 1),
            'kidney_right_center_x_norm': np.random.uniform(0, 1),
            'kidney_right_center_y_norm': np.random.uniform(0, 1),
            'kidney_right_center_z_norm': np.random.uniform(0, 1),
            'kidney_left_length_mm': np.random.uniform(80, 120),
            'kidney_left_volume_cm3': np.random.uniform(100, 200),
            'kidney_right_length_mm': np.random.uniform(80, 120),
            'kidney_right_volume_cm3': np.random.uniform(100, 200),
            'body_width_mm': np.random.uniform(300, 500),
            'body_depth_mm': np.random.uniform(150, 250),
            'body_area_mm2': np.random.uniform(60000, 100000),
            'kidney_left_to_spine_distance': np.random.uniform(30, 70),
            'kidney_right_to_spine_distance': np.random.uniform(30, 70),
            'kidney_left_to_body_center_distance': np.random.uniform(60, 100),
            'kidney_right_to_body_center_distance': np.random.uniform(60, 100),
            'spine_center_x': np.random.uniform(150, 250),
            'spine_center_y': np.random.uniform(100, 200),
            'spine_center_z': np.random.uniform(50, 150),
            'body_com_x': np.random.uniform(150, 250),
            'body_com_y': np.random.uniform(100, 200),
            'body_com_z': np.random.uniform(50, 150),
            'patient_position_encoded': np.random.choice([0, 1])
        }
        test_patients.append(patient)
    
    print(f"Generated {len(test_patients)} test patients")
    
    # Test validation performance
    import time
    start_time = time.time()
    
    for patient in test_patients:
        is_valid, errors = predictor.validate_input(patient)
        if not is_valid:
            print(f"Validation failed: {errors}")
    
    validation_time = time.time() - start_time
    print(f"Validation time: {validation_time:.4f}s ({validation_time/len(test_patients)*1000:.2f}ms per patient)")
    
    print("\n✅ Performance benchmark completed!")

def main():
    """Run all tests"""
    print("="*60)
    print("KIDNEY DISPLACEMENT PREDICTOR - TEST SUITE")
    print("="*60)
    
    # Run unit tests
    print("\n1. Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmark
    print("\n2. Running performance benchmark...")
    run_performance_benchmark()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
