"""
model.py
Production machine learning training, evaluation, comparison, and prediction pipeline
for the IBM AI Enterprise Workflow Capstone.

Supports multi-country models, 'all' countries combined, comparative evaluation against
baseline models, time-series feature engineering, and complete read/write test isolation.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from data_ingestion import fetch_data, convert_to_ts
from logger import update_train_log, update_predict_log

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "cs-train")

os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_VERSION = "0.1"
MODEL_VERSION_NOTE = "RandomForestRegressor with TimeSeries Rolling Features & StandardScaler"

FEATURES = [
    "revenue_lag_7", "revenue_lag_14", "revenue_lag_30",
    "revenue_roll_mean_7", "revenue_roll_mean_30",
    "revenue_roll_std_7", "revenue_roll_std_30",
    "purchases_roll_sum_7", "purchases_roll_sum_30",
    "views_roll_sum_30", "day_of_week", "day_of_month", "month", "is_weekend"
]
TARGET = "target"

def _get_model_path(tag="all", test=False):
    prefix = "test-" if test else ""
    return os.path.join(MODELS_DIR, f"{prefix}model-{tag}.joblib")

def compare_models(ts_df):
    """
    Evaluates multiple candidate models vs baseline using TimeSeriesSplit cross-validation.
    Compares:
      1. Baseline (Dummy Mean)
      2. Ridge Regression
      3. Random Forest Regressor
      4. Gradient Boosting Regressor
      
    Returns:
        dict: Summary metrics for each model.
    """
    clean_df = ts_df.dropna(subset=FEATURES + [TARGET]).copy()
    clean_df = clean_df.iloc[30:-30].copy()
    
    X = clean_df[FEATURES]
    y = clean_df[TARGET]

    candidate_models = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=10.0))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42))
        ])
    }

    tscv = TimeSeriesSplit(n_splits=4)
    comparison_results = {}

    for name, pipeline in candidate_models.items():
        rmses, maes, r2s = [], [], []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_val)

            rmses.append(root_mean_squared_error(y_val, preds))
            maes.append(mean_absolute_error(y_val, preds))
            r2s.append(r2_score(y_val, preds))

        comparison_results[name] = {
            "RMSE": float(np.mean(rmses)),
            "MAE": float(np.mean(maes)),
            "R2": float(np.mean(r2s))
        }

    return comparison_results

def build_model_pipeline():
    """
    Creates the production Scikit-Learn pipeline.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_split=4, random_state=42))
    ])

def model_train(data_dir=DATA_DIR, country=None, test=False):
    """
    Trains and saves forecasting models.
    Supports isolation: when test=True, saves with 'test-' prefix and logs to test log.
    """
    start_time = time.time()
    df_raw = fetch_data(data_dir)

    target_countries = [country] if country else ["all", "united_kingdom", "germany", "eire", "france", "spain"]
    trained_models = {}

    for tag in target_countries:
        ts_df = convert_to_ts(df_raw, country=tag)
        clean_df = ts_df.dropna(subset=FEATURES + [TARGET]).iloc[30:-30].copy()

        X = clean_df[FEATURES]
        y = clean_df[TARGET]

        split_idx = int(len(clean_df) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        pipeline = build_model_pipeline()
        pipeline.fit(X_train, y_train)

        val_preds = pipeline.predict(X_val)
        val_rmse = root_mean_squared_error(y_val, val_preds)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)

        eval_test = {
            "rmse": round(float(val_rmse), 2),
            "mae": round(float(val_mae), 2),
            "r2": round(float(val_r2), 4)
        }

        model_payload = {
            "pipeline": pipeline,
            "features": FEATURES,
            "target": TARGET,
            "country": tag,
            "date_range": (str(clean_df["date"].min().date()), str(clean_df["date"].max().date())),
            "eval_test": eval_test,
            "model_version": MODEL_VERSION,
            "last_ts_snapshot": ts_df.tail(60).copy()
        }

        model_file = _get_model_path(tag=tag, test=test)
        joblib.dump(model_payload, model_file)
        trained_models[tag] = model_payload

        runtime = time.time() - start_time
        update_train_log(
            tag=tag,
            date_range=model_payload["date_range"],
            eval_test=eval_test,
            runtime=runtime,
            model_version=MODEL_VERSION,
            model_version_note=MODEL_VERSION_NOTE,
            test=test
        )

    return trained_models

def model_load(country="all", test=False):
    """
    Loads saved model bundle for the specified country.
    If in test mode and file does not exist, auto-trains test model on the fly.
    """
    tag = country.strip().lower().replace(" ", "_") if country else "all"
    model_file = _get_model_path(tag=tag, test=test)

    if test and not os.path.exists(model_file):
        model_train(country=tag, test=True)

    if not os.path.exists(model_file):
        fallback_file = _get_model_path(tag="all", test=test)
        if test and not os.path.exists(fallback_file):
            model_train(country="all", test=True)
        if os.path.exists(fallback_file):
            model_file = fallback_file

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}. Please run model_train first.")

    return joblib.load(model_file)

def model_predict(country="all", target_date="2019-06-01", data_dir=DATA_DIR, test=False):
    """
    Generates 30-day forward revenue forecast for a country and target date.
    Accepts specific country or 'all'.
    """
    start_time = time.time()
    tag = country.strip().lower().replace(" ", "_") if country else "all"
    
    model_bundle = model_load(country=tag, test=test)
    pipeline = model_bundle["pipeline"]
    features_list = model_bundle["features"]

    df_raw = fetch_data(data_dir)
    ts_df = convert_to_ts(df_raw, country=tag)
    
    target_dt = pd.to_datetime(target_date)
    matching = ts_df[ts_df["date"] == target_dt]

    if not matching.empty:
        feat_vector = matching[features_list].iloc[0:1]
    else:
        feat_vector = ts_df[features_list].tail(1)

    y_pred = pipeline.predict(feat_vector)
    pred_val = float(y_pred[0])

    val_rmse = model_bundle.get("eval_test", {}).get("rmse", 500.0)
    lower_bound = max(0.0, pred_val - 1.96 * val_rmse)
    upper_bound = pred_val + 1.96 * val_rmse

    runtime = time.time() - start_time
    query_payload = {"country": tag, "target_date": str(target_date)}

    update_predict_log(
        y_pred=pred_val,
        y_proba=[lower_bound, upper_bound],
        query=query_payload,
        runtime=runtime,
        model_version=MODEL_VERSION,
        test=test
    )

    return {
        "country": tag,
        "target_date": str(target_date),
        "y_pred": round(pred_val, 2),
        "y_proba": [round(lower_bound, 2), round(upper_bound, 2)],
        "model_version": MODEL_VERSION,
        "runtime_seconds": round(runtime, 4)
    }
