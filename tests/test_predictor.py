import json
import unittest
from pathlib import Path

import pandas as pd

from models.production.predictor import KidneyDisplacementPredictor


class TestKidneyDisplacementPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prod_dir = Path(__file__).parents[1] / "models" / "production"
        cls.predictor = KidneyDisplacementPredictor.load_model(cls.prod_dir)
        
        # Load test data
        val_df = pd.read_csv(cls.prod_dir.parents[1] / "data" / "processed" / "validation.csv")
        cls.test_features = val_df.iloc[0].to_dict()
        
        # Load feature names
        with open(cls.prod_dir / "feature_names.json", "r", encoding="utf-8") as f:
            cls.feature_names = json.load(f)
    
    def test_load_model(self):
        self.assertIsNotNone(self.predictor)
        self.assertEqual(len(self.predictor.feature_names), 37)
        self.assertEqual(len(self.predictor.target_names), 9)
    
    def test_predict_shape(self):
        result = self.predictor.predict(self.test_features)
        self.assertEqual(len(result.predictions), 9)
        for target in self.predictor.target_names:
            self.assertIn(target, result.predictions)
    
    def test_predict_values(self):
        result = self.predictor.predict(self.test_features)
        for delta_val in result.predictions.values():
            self.assertIsInstance(delta_val, float)
            # Reasonable displacement range in mm
            self.assertGreater(abs(delta_val), 0)
            self.assertLess(abs(delta_val), 100)
    
    def test_feature_validation(self):
        # Test with missing features
        incomplete_features = {"age": 65}
        with self.assertRaises(KeyError):
            self.predictor.predict(incomplete_features)
    
    def test_model_types(self):
        self.assertEqual(self.predictor.model_type, "ensemble")
        self.assertIsNotNone(self.predictor.rf_model)
        self.assertEqual(len(self.predictor.xgb_models), 9)


if __name__ == "__main__":
    unittest.main()
