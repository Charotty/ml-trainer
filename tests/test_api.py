import json
import unittest
from pathlib import Path

import pandas as pd
import requests


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:8000"
        
        # Load test data
        val_df = pd.read_csv(Path(__file__).parents[1] / "data" / "processed" / "validation.csv")
        cls.test_features = val_df.iloc[0].to_dict()
        
        # Load feature names
        with open(Path(__file__).parents[1] / "models" / "production" / "feature_names.json", "r", encoding="utf-8") as f:
            cls.feature_names = json.load(f)
        
        # Filter features
        cls.features = {k: cls.test_features[k] for k in cls.feature_names}
    
    def test_health(self):
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
    
    def test_model_info(self):
        response = requests.get(f"{self.base_url}/model_info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("feature_count", data)
        self.assertIn("target_count", data)
        self.assertIn("val_metrics", data)
    
    def test_predict_displacement(self):
        payload = {"features": self.features}
        response = requests.post(f"{self.base_url}/predict_displacement", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 9)
    
    def test_plan_trocars(self):
        payload = {
            "upper": {"x": 120.5, "y": 85.2, "z": 45.7},
            "middle": {"x": 118.0, "y": 84.0, "z": 44.5},
            "lower": {"x": 115.5, "y": 82.8, "z": 43.3}
        }
        response = requests.post(f"{self.base_url}/plan_trocars", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("trocars", data)
        self.assertEqual(len(data["trocars"]), 3)
    
    def test_full_pipeline(self):
        payload = {
            "features": self.features,
            "kidney_points_mm": {
                "upper": {"x": 120.5, "y": 85.2, "z": 45.7},
                "middle": {"x": 118.0, "y": 84.0, "z": 44.5},
                "lower": {"x": 115.5, "y": 82.8, "z": 43.3}
            }
        }
        response = requests.post(f"{self.base_url}/full_pipeline", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("displacement", data)
        self.assertIn("trocars", data)
        self.assertIn("predictions", data["displacement"])
        self.assertIn("trocars", data["trocars"])
    
    def test_predict_displacement_invalid_features(self):
        payload = {"features": {"age": 65}}  # Missing most features
        response = requests.post(f"{self.base_url}/predict_displacement", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())
    
    def test_plan_trocars_invalid_points(self):
        payload = {"upper": {"x": 120.5}}  # Missing y, z
        response = requests.post(f"{self.base_url}/plan_trocars", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())


if __name__ == "__main__":
    unittest.main()
