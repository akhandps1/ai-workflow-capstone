"""
logger.py
Enterprise logging module for model training and prediction audit trails.
Strictly isolates production logs from test logs using the `test` parameter flag.
"""

import os
import json
import time
from datetime import datetime, timezone
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOGS_DIR, exist_ok=True)

def _get_log_paths(test=False):
    """
    Returns train and predict log paths based on test flag.
    Ensures absolute isolation between test logs and production logs.
    """
    prefix = "test" if test else "production"
    train_log = os.path.join(LOGS_DIR, f"train-{prefix}.log")
    predict_log = os.path.join(LOGS_DIR, f"predict-{prefix}.log")
    return train_log, predict_log

def update_train_log(tag, date_range, eval_test, runtime, model_version="0.1", model_version_note="", test=False):
    """
    Logs model training events.
    """
    train_log_file, _ = _get_log_paths(test=test)
    
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": str(tag),
        "start_date": str(date_range[0]) if len(date_range) > 0 else "N/A",
        "end_date": str(date_range[1]) if len(date_range) > 1 else "N/A",
        "eval_test": eval_test,
        "runtime_seconds": round(float(runtime), 4) if isinstance(runtime, (int, float)) else str(runtime),
        "model_version": str(model_version),
        "model_version_note": str(model_version_note)
    }

    with open(train_log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return log_entry

def update_predict_log(y_pred, y_proba, query, runtime, model_version="0.1", test=False):
    """
    Logs model prediction queries and responses.
    """
    _, predict_log_file = _get_log_paths(test=test)

    clean_pred = [round(float(x), 2) for x in y_pred] if isinstance(y_pred, (list, tuple)) else round(float(y_pred), 2)
    clean_proba = [round(float(x), 2) for x in y_proba] if isinstance(y_proba, (list, tuple)) else y_proba

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "y_pred": clean_pred,
        "y_proba": clean_proba,
        "query": query,
        "runtime_seconds": round(float(runtime), 4) if isinstance(runtime, (int, float)) else str(runtime),
        "model_version": str(model_version)
    }

    with open(predict_log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry

def read_train_logs(test=False):
    """
    Reads all logged training events into a list of dicts.
    """
    train_log_file, _ = _get_log_paths(test=test)
    if not os.path.exists(train_log_file):
        return []

    entries = []
    with open(train_log_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    return entries

def read_predict_logs(test=False):
    """
    Reads all logged prediction events into a list of dicts.
    """
    _, predict_log_file = _get_log_paths(test=test)
    if not os.path.exists(predict_log_file):
        return []

    entries = []
    with open(predict_log_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    return entries

def clear_test_logs():
    """
    Safely clears only the test log files, leaving production logs intact.
    """
    train_test, predict_test = _get_log_paths(test=True)
    for p in [train_test, predict_test]:
        if os.path.exists(p):
            os.remove(p)
