"""
monitoring.py
Production monitoring module for model performance and data drift tracking.
Analyzes operational metrics, calculates out-of-sample RMSE/MAE against ground truth,
and detects covariate shift using Kolmogorov-Smirnov statistical tests and Wasserstein distance.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from data_ingestion import fetch_data, convert_to_ts
from model import model_load, FEATURES, TARGET
from logger import read_predict_logs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "data", "cs-train")
PROD_DATA_DIR = os.path.join(BASE_DIR, "data", "cs-production")

def monitor_performance(train_dir=TRAIN_DATA_DIR, prod_dir=PROD_DATA_DIR, country="all", test=False):
    """
    Comprehensive performance and data drift monitoring pipeline.
    
    Args:
        train_dir (str): Baseline training data path.
        prod_dir (str): Incoming production streaming data path.
        country (str): Specific country or 'all'.
        test (bool): Test isolation flag.
        
    Returns:
        dict: Detailed monitoring report including metrics, drift statistics, and alert status.
    """
    model_bundle = model_load(country=country, test=test)
    pipeline = model_bundle["pipeline"]
    baseline_eval = model_bundle.get("eval_test", {})

    # Ingest baseline and production time-series
    df_train_raw = fetch_data(train_dir)
    ts_train = convert_to_ts(df_train_raw, country=country)
    clean_train = ts_train.dropna(subset=FEATURES + [TARGET]).iloc[30:-30]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "country": country,
        "model_version": model_bundle.get("model_version", "0.1"),
        "baseline_metrics": baseline_eval,
        "feature_drift": {},
        "production_eval": {},
        "status": "HEALTHY",
        "drift_detected": False,
        "alerts": []
    }

    # Evaluate against production data if available
    if os.path.exists(prod_dir) and any(os.scandir(prod_dir)):
        df_prod_raw = fetch_data(prod_dir)
        ts_prod = convert_to_ts(df_prod_raw, country=country)
        clean_prod = ts_prod.dropna(subset=FEATURES + [TARGET])

        if not clean_prod.empty:
            X_prod = clean_prod[FEATURES]
            y_prod = clean_prod[TARGET]

            y_prod_preds = pipeline.predict(X_prod)
            prod_rmse = root_mean_squared_error(y_prod, y_prod_preds)
            prod_mae = mean_absolute_error(y_prod, y_prod_preds)

            report["production_eval"] = {
                "records_evaluated": len(clean_prod),
                "rmse": round(float(prod_rmse), 2),
                "mae": round(float(prod_mae), 2)
            }

            # Check for metric degradation
            base_rmse = baseline_eval.get("rmse", 1.0)
            if base_rmse > 0 and (prod_rmse > base_rmse * 1.5):
                report["alerts"].append(f"RMSE degraded by >50% (Baseline: {base_rmse:.2f}, Production: {prod_rmse:.2f})")
                report["status"] = "WARNING"

            # Compute covariate drift per feature
            for feat in FEATURES:
                train_vals = clean_train[feat].values
                prod_vals = clean_prod[feat].values

                ks_stat, p_val = ks_2samp(train_vals, prod_vals)
                w_dist = wasserstein_distance(train_vals, prod_vals)

                drift_flag = bool(p_val < 0.01 and ks_stat > 0.25)
                report["feature_drift"][feat] = {
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "wasserstein_distance": round(float(w_dist), 4),
                    "drift_detected": drift_flag
                }

                if drift_flag:
                    report["drift_detected"] = True
                    report["alerts"].append(f"Statistical drift detected in feature '{feat}' (KS p-val: {p_val:.4f})")

    # Add API operational logging metrics
    predict_logs = read_predict_logs(test=test)
    report["operational_metrics"] = {
        "total_inferences_logged": len(predict_logs),
        "avg_latency_seconds": round(float(np.mean([x.get("runtime_seconds", 0) for x in predict_logs])), 4) if predict_logs else 0.0
    }

    if report["drift_detected"] and report["status"] == "HEALTHY":
        report["status"] = "DRIFT_DETECTED"

    return report

if __name__ == "__main__":
    print("[*] Running continuous monitoring check...")
    res = monitor_performance(country="all")
    print(json.dumps(res, indent=2))
