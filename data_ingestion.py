"""
data_ingestion.py
Automated data ingestion and feature engineering module for IBM AI Enterprise Workflow.
Reads raw transactional invoice JSON files, cleans data, aggregates into regular
time-series records per country and for 'all' countries combined, and generates
engineered lag/rolling features for predictive modeling.
"""

import os
import glob
import json
import argparse
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "cs-train")

def fetch_data(data_dir=DEFAULT_DATA_DIR):
    """
    Ingests all raw JSON invoice files from data_dir.
    Standardizes schema, handles nulls, and parses dates.
    
    Returns:
        pd.DataFrame: Cleaned transactional DataFrame.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON invoice files found in {data_dir}")

    records = []
    for filepath in json_files:
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
                records.extend(data)
            except Exception as e:
                print(f"[!] Warning: failed to parse {filepath}: {e}")

    df = pd.DataFrame(records)

    # Standardize column naming variations if present
    column_mapping = {
        "StreamID": "stream_id",
        "total_price": "price",
        "Price": "price",
        "Country": "country",
        "Invoice": "invoice",
        "TimesViewed": "times_viewed",
        "Customer_ID": "customer_id",
        "CustomerID": "customer_id"
    }
    df = df.rename(columns=column_mapping)

    # Data validation & cleaning
    df = df.dropna(subset=["year", "month", "day", "price"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce").fillna(1).astype(int)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"] = df["day"].astype(int)
    
    # Standardize country string
    df["country"] = df["country"].astype(str).str.strip().str.lower().str.replace(" ", "_")

    # Filter invalid records
    df = df[df["price"] > 0]
    
    # Construct exact datetime
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.sort_values(by="date").reset_index(drop=True)
    
    return df

def convert_to_ts(df, country=None):
    """
    Transforms transactional records into a continuous daily time-series DataFrame
    for a specific country or all countries combined.
    
    Args:
        df (pd.DataFrame): Clean transactional DataFrame from fetch_data.
        country (str): Specific country name (e.g. 'united_kingdom') or None/'all'.
        
    Returns:
        pd.DataFrame: Daily time-series DataFrame with aggregated and engineered features.
    """
    if country and country.lower() not in ("all", "all_countries"):
        country_norm = country.strip().lower().replace(" ", "_")
        filtered_df = df[df["country"] == country_norm].copy()
    else:
        country_norm = "all"
        filtered_df = df.copy()

    if filtered_df.empty:
        raise ValueError(f"No records found for country: '{country}'")

    # Daily aggregation
    daily = filtered_df.groupby("date").agg(
        revenue=("price", "sum"),
        purchases=("invoice", "count"),
        total_views=("times_viewed", "sum"),
        unique_streams=("stream_id", "nunique"),
        unique_customers=("customer_id", "nunique")
    ).reset_index()

    # Reindex to continuous calendar range
    full_date_range = pd.date_range(start=daily["date"].min(), end=daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_date_range).fillna(0.0).reset_index()
    daily = daily.rename(columns={"index": "date"})

    # Feature Engineering (Lags, Rolling windows, Calendar)
    daily["revenue_lag_7"] = daily["revenue"].shift(7).fillna(0.0)
    daily["revenue_lag_14"] = daily["revenue"].shift(14).fillna(0.0)
    daily["revenue_lag_30"] = daily["revenue"].shift(30).fillna(0.0)

    daily["revenue_roll_mean_7"] = daily["revenue"].rolling(window=7, min_periods=1).mean()
    daily["revenue_roll_mean_30"] = daily["revenue"].rolling(window=30, min_periods=1).mean()
    daily["revenue_roll_std_7"] = daily["revenue"].rolling(window=7, min_periods=1).std().fillna(0.0)
    daily["revenue_roll_std_30"] = daily["revenue"].rolling(window=30, min_periods=1).std().fillna(0.0)

    daily["purchases_roll_sum_7"] = daily["purchases"].rolling(window=7, min_periods=1).sum()
    daily["purchases_roll_sum_30"] = daily["purchases"].rolling(window=30, min_periods=1).sum()
    daily["views_roll_sum_30"] = daily["total_views"].rolling(window=30, min_periods=1).sum()

    daily["day_of_week"] = daily["date"].dt.day_of_week
    daily["day_of_month"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)

    # Target: 30-day forward cumulative revenue
    rev_forward_30 = daily["revenue"].iloc[::-1].rolling(window=30, min_periods=1).sum().iloc[::-1]
    daily["target"] = rev_forward_30.values
    
    # Store country tag
    daily["country"] = country_norm

    return daily

def ingest_data(data_dir=DEFAULT_DATA_DIR, output_dir=None):
    """
    Complete automated ingestion pipeline:
    Fetches raw transactions, builds TS datasets for top countries and 'all',
    and optionally saves processed files.
    """
    print(f"[*] Ingesting transactional data from: {data_dir}")
    df_raw = fetch_data(data_dir)
    print(f"[+] Loaded {len(df_raw)} raw transaction records across {df_raw['country'].nunique()} countries.")

    top_countries = (
        df_raw["country"].value_counts().head(5).index.tolist()
    )
    all_targets = ["all"] + top_countries
    ts_data_dict = {}

    for c in all_targets:
        ts_df = convert_to_ts(df_raw, country=c)
        ts_data_dict[c] = ts_df
        print(f"  -> Generated time-series for '{c}': {len(ts_df)} daily points.")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        raw_out = os.path.join(output_dir, "transactions_cleaned.csv")
        df_raw.to_csv(raw_out, index=False)
        for c, ts_df in ts_data_dict.items():
            ts_out = os.path.join(output_dir, f"ts_{c}.csv")
            ts_df.to_csv(ts_out, index=False)
        print(f"[+] Exported processed datasets to {output_dir}")

    return df_raw, ts_data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Data Ingestion for AI Enterprise Workflow")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to raw invoice json files")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "data", "processed"), help="Output path for processed CSVs")
    args = parser.parse_args()

    ingest_data(data_dir=args.data_dir, output_dir=args.output_dir)
