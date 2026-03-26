import unittest
from pathlib import Path

from backend.trocar_planner import plan_trocars


class TestTrocarPlanner(unittest.TestCase):
    def test_plan_trocars_basic(self):
        kidney_points = {
            "upper": {"x": 120.5, "y": 85.2, "z": 45.7},
            "middle": {"x": 118.0, "y": 84.0, "z": 44.5},
            "lower": {"x": 115.5, "y": 82.8, "z": 43.3}
        }
        
        result = plan_trocars(kidney_points)
        
        # Should return 3 trocars
        self.assertEqual(len(result), 3)
        
        # Check required fields
        for trocar in result:
            self.assertIn("name", trocar)
            self.assertIn("position_mm", trocar)
            self.assertIn("depth_mm", trocar)
            self.assertIn("entry_angle_deg", trocar)
            self.assertIn("safety_score", trocar)
            
            # Check position structure
            pos = trocar["position_mm"]
            self.assertIn("x", pos)
            self.assertIn("y", pos)
            self.assertIn("z", pos)
            
            # Check value types
            self.assertIsInstance(trocar["depth_mm"], float)
            self.assertIsInstance(trocar["entry_angle_deg"], float)
            self.assertIsInstance(trocar["safety_score"], float)
            self.assertGreaterEqual(trocar["safety_score"], 0.0)
            self.assertLessEqual(trocar["safety_score"], 1.0)
    
    def test_plan_trocars_names(self):
        kidney_points = {
            "upper": {"x": 120.5, "y": 85.2, "z": 45.7},
            "lower": {"x": 115.5, "y": 82.8, "z": 43.3}
        }
        
        result = plan_trocars(kidney_points)
        names = [t["name"] for t in result]
        
        self.assertIn("camera", names)
        self.assertIn("working_1", names)
        self.assertIn("working_2", names)
    
    def test_plan_trocars_missing_middle(self):
        kidney_points = {
            "upper": {"x": 120.5, "y": 85.2, "z": 45.7},
            "lower": {"x": 115.5, "y": 82.8, "z": 43.3}
        }
        
        result = plan_trocars(kidney_points)
        self.assertEqual(len(result), 3)
    
    def test_plan_trocars_invalid_input(self):
        # Empty kidney points
        with self.assertRaises(ValueError):
            plan_trocars({})
        
        # Missing required coordinates
        with self.assertRaises(KeyError):
            plan_trocars({"upper": {"x": 120.5}})


if __name__ == "__main__":
    unittest.main()
