"""
ApiTests.py
Unit tests for the Flask REST API endpoints (/health, /predict, /train, /logs, /monitoring).
Verifies functionality for both specific country and all countries combined,
and ensures tests are executed in isolated test mode.
"""

import unittest
import json
import os
import sys

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from model import model_train
from logger import clear_test_logs

class ApiTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        clear_test_logs()
        # Pre-train isolated test models for tests
        model_train(country="all", test=True)
        model_train(country="united_kingdom", test=True)

    @classmethod
    def tearDownClass(cls):
        clear_test_logs()

    def test_01_health_endpoint(self):
        """Test GET /health returns 200 OK and healthy status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("supported_countries", data)

    def test_02_predict_all_countries(self):
        """Test POST /predict for all countries combined in test mode."""
        payload = {
            "country": "all",
            "target_date": "2019-05-15",
            "test": True
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["country"], "all")
        self.assertIsInstance(data["y_pred"], (int, float))
        self.assertGreater(data["y_pred"], 0)
        self.assertIsInstance(data["y_proba"], list)
        self.assertEqual(len(data["y_proba"]), 2)

    def test_03_predict_specific_country(self):
        """Test POST /predict for a specific country (united_kingdom)."""
        payload = {
            "country": "united_kingdom",
            "target_date": "2019-05-15",
            "test": True
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["country"], "united_kingdom")
        self.assertIsInstance(data["y_pred"], (int, float))
        self.assertGreater(data["y_pred"], 0)

    def test_04_predict_missing_date(self):
        """Test POST /predict with missing target_date returns 400 Bad Request."""
        payload = {"country": "all"}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")

    def test_05_predict_invalid_country(self):
        """Test POST /predict with invalid country returns 400 Bad Request."""
        payload = {
            "country": "invalid_country_xyz",
            "target_date": "2019-05-15"
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")

    def test_06_train_endpoint_isolated(self):
        """Test POST /train triggers model training in test mode."""
        payload = {
            "country": "united_kingdom",
            "test": True
        }
        response = self.client.post("/train", json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["test_mode"])
        self.assertIn("united_kingdom", data["models_trained"])

    def test_07_logs_endpoints(self):
        """Test GET /logs/predict and /logs/train in isolated test mode."""
        # Ensure at least one prediction log has been written
        self.client.post("/predict", json={"country": "all", "target_date": "2019-05-15", "test": True})

        resp_pred = self.client.get("/logs/predict?test=true")
        self.assertEqual(resp_pred.status_code, 200)
        data_pred = json.loads(resp_pred.data)
        self.assertEqual(data_pred["status"], "success")
        self.assertTrue(data_pred["test_mode"])
        self.assertGreater(data_pred["count"], 0)

        resp_train = self.client.get("/logs/train?test=true")
        self.assertEqual(resp_train.status_code, 200)
        data_train = json.loads(resp_train.data)
        self.assertEqual(data_train["status"], "success")
        self.assertTrue(data_train["test_mode"])

    def test_08_monitoring_endpoint(self):
        """Test GET /monitoring returns model performance & drift report."""
        response = self.client.get("/monitoring?country=all&test=true")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("status", data)
        self.assertIn("baseline_metrics", data)
        self.assertIn("feature_drift", data)

if __name__ == "__main__":
    unittest.main()
