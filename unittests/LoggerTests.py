"""
LoggerTests.py
Unit tests for the training and prediction logging system.
Verifies entry structure, timestamp validity, and strict isolation between
production logs and unit test logs.
"""

import unittest
import os
import sys

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import (
    update_train_log,
    update_predict_log,
    read_train_logs,
    read_predict_logs,
    clear_test_logs,
    _get_log_paths
)

class LoggerTests(unittest.TestCase):

    def setUp(self):
        clear_test_logs()

    def tearDown(self):
        clear_test_logs()

    def test_01_update_train_log(self):
        """Test train logging writes valid formatted entry."""
        tag = "test_country"
        date_range = ("2018-01-01", "2018-12-31")
        eval_test = {"rmse": 105.5, "mae": 82.3}
        runtime = 1.25

        entry = update_train_log(
            tag=tag,
            date_range=date_range,
            eval_test=eval_test,
            runtime=runtime,
            model_version="0.1",
            model_version_note="Unit test model",
            test=True
        )

        self.assertIn("timestamp", entry)
        self.assertEqual(entry["tag"], tag)
        self.assertEqual(entry["eval_test"], eval_test)

        logs = read_train_logs(test=True)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["tag"], tag)

    def test_02_update_predict_log(self):
        """Test predict logging writes valid formatted entry."""
        y_pred = 15420.50
        y_proba = [12000.0, 18000.0]
        query = {"country": "all", "target_date": "2019-06-01"}
        runtime = 0.045

        entry = update_predict_log(
            y_pred=y_pred,
            y_proba=y_proba,
            query=query,
            runtime=runtime,
            model_version="0.1",
            test=True
        )

        self.assertIn("timestamp", entry)
        self.assertEqual(entry["y_pred"], y_pred)
        self.assertEqual(entry["query"], query)

        logs = read_predict_logs(test=True)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["y_pred"], y_pred)

    def test_03_test_isolation_from_production(self):
        """Test that writing to test logs strictly avoids altering production logs."""
        prod_train_path, prod_predict_path = _get_log_paths(test=False)
        
        # Record initial sizes of production log files
        initial_train_size = os.path.getsize(prod_train_path) if os.path.exists(prod_train_path) else 0
        initial_predict_size = os.path.getsize(prod_predict_path) if os.path.exists(prod_predict_path) else 0

        # Perform test writes
        update_train_log("isolated_test", ("2018-01-01", "2018-01-10"), {"rmse": 10.0}, 0.5, test=True)
        update_predict_log(999.0, None, {"country": "test"}, 0.01, test=True)

        # Check production log sizes remain identical
        post_train_size = os.path.getsize(prod_train_path) if os.path.exists(prod_train_path) else 0
        post_predict_size = os.path.getsize(prod_predict_path) if os.path.exists(prod_predict_path) else 0

        self.assertEqual(initial_train_size, post_train_size)
        self.assertEqual(initial_predict_size, post_predict_size)

if __name__ == "__main__":
    unittest.main()
