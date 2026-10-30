"""
ModelTests.py
Unit tests for the model training, evaluation, loading, and prediction functions.
Ensures tests isolate read/write artifacts using the test flag and test paths.
"""

import unittest
import os
import sys
import glob

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import model_train, model_load, model_predict, compare_models, _get_model_path
from data_ingestion import fetch_data, convert_to_ts

class ModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure a clean isolated test model is trained
        model_train(country="all", test=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up isolated test model artifacts
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        models_dir = os.path.join(base_dir, "models")
        for test_model in glob.glob(os.path.join(models_dir, "test-model-*.joblib")):
            if os.path.exists(test_model):
                os.remove(test_model)

    def test_01_model_train_isolated(self):
        """Test model_train with test=True creates isolated test model artifact."""
        result = model_train(country="united_kingdom", test=True)
        self.assertIn("united_kingdom", result)
        test_path = _get_model_path(tag="united_kingdom", test=True)
        self.assertTrue(os.path.exists(test_path))

    def test_02_model_load(self):
        """Test model_load retrieves valid pipeline and metadata bundle."""
        bundle = model_load(country="all", test=True)
        self.assertIn("pipeline", bundle)
        self.assertIn("features", bundle)
        self.assertIn("target", bundle)
        self.assertIn("eval_test", bundle)
        self.assertGreater(len(bundle["features"]), 5)

    def test_03_model_predict_all(self):
        """Test model_predict returns valid positive revenue prediction for 'all'."""
        pred = model_predict(country="all", target_date="2019-05-15", test=True)
        self.assertEqual(pred["country"], "all")
        self.assertIsInstance(pred["y_pred"], float)
        self.assertGreater(pred["y_pred"], 0.0)
        self.assertEqual(len(pred["y_proba"]), 2)
        self.assertLessEqual(pred["y_proba"][0], pred["y_proba"][1])

    def test_04_model_predict_specific_country(self):
        """Test model_predict returns valid revenue prediction for specific country."""
        pred = model_predict(country="united_kingdom", target_date="2019-05-15", test=True)
        self.assertEqual(pred["country"], "united_kingdom")
        self.assertIsInstance(pred["y_pred"], float)
        self.assertGreater(pred["y_pred"], 0.0)

    def test_05_convert_to_ts_shape_and_features(self):
        """Test convert_to_ts generates non-empty dataframe with all engineered features."""
        df_raw = fetch_data()
        ts_df = convert_to_ts(df_raw, country="germany")
        self.assertGreater(len(ts_df), 300)
        expected_cols = ["date", "revenue", "target", "revenue_lag_7", "revenue_roll_mean_7", "month"]
        for col in expected_cols:
            self.assertIn(col, ts_df.columns)

    def test_06_model_comparison_multiple_models(self):
        """Test compare_models executes multi-model cross-validation."""
        df_raw = fetch_data()
        ts_df = convert_to_ts(df_raw, country="all")
        comp = compare_models(ts_df)
        self.assertIn("Baseline (Mean)", comp)
        self.assertIn("Random Forest", comp)
        self.assertIn("Gradient Boosting", comp)
        for model_name, metrics in comp.items():
            self.assertIn("RMSE", metrics)
            self.assertIn("MAE", metrics)
            self.assertIn("R2", metrics)

if __name__ == "__main__":
    unittest.main()
