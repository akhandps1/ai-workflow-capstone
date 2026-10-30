"""
generate_sample_data.py
Creates realistic training and production transactional JSON invoice files
matching the IBM AI Enterprise Workflow Capstone schema.
"""

import json
import os
import random
from datetime import datetime, timedelta
import numpy as np

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "cs-train")
PROD_DIR = os.path.join(BASE_DIR, "data", "cs-production")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(PROD_DIR, exist_ok=True)

COUNTRIES = [
    "united_kingdom", "germany", "eire", "france", "spain", 
    "netherlands", "portugal", "australia", "belgium", "norway"
]

COUNTRY_WEIGHTS = [0.65, 0.10, 0.08, 0.06, 0.04, 0.03, 0.015, 0.01, 0.01, 0.005]

STREAM_IDS = [f"{random.randint(10000, 99999)}" for _ in range(500)]
CUSTOMER_BASE = {c: [random.randint(10000, 19999) for _ in range(250)] for c in COUNTRIES}

def generate_records_for_month(year, month, records_count=2500):
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
    num_days = (end_date - start_date).days + 1
    records = []
    
    # Monthly growth trend and seasonal factor
    base_trend = (year - 2017) * 12 + month
    seasonal_factor = 1.0 + 0.35 * np.sin(2 * np.pi * (month / 12.0))
    if month in (11, 12):  # Holiday shopping surge
        seasonal_factor += 0.4
        
    actual_records = int(records_count * (1.0 + 0.02 * base_trend) * seasonal_factor)
    invoice_seq = 489000 + base_trend * 1000
    
    for _ in range(actual_records):
        day = random.randint(1, num_days)
        cur_date = datetime(year, month, day)
        day_of_week = cur_date.weekday()
        
        # Weekend multiplier
        weekend_boost = 1.25 if day_of_week in (4, 5, 6) else 1.0
        
        country = random.choices(COUNTRIES, weights=COUNTRY_WEIGHTS)[0]
        customer_id = float(random.choice(CUSTOMER_BASE[country]))
        invoice = str(invoice_seq + random.randint(0, 800))
        stream_id = random.choice(STREAM_IDS)
        
        # Base price and views
        base_price = np.random.exponential(scale=4.5) + 0.5
        price = round(float(base_price * weekend_boost), 2)
        times_viewed = int(max(1, np.random.poisson(lam=5) * weekend_boost))
        
        records.append({
            "country": country,
            "customer_id": customer_id,
            "invoice": invoice,
            "price": price,
            "stream_id": stream_id,
            "times_viewed": times_viewed,
            "year": year,
            "month": month,
            "day": day
        })
        
    return records

def build_datasets():
    print("[*] Generating training invoice dataset (2017-11 through 2019-05)...")
    # Training months: Nov 2017 to May 2019
    train_dates = []
    for y in [2017, 2018, 2019]:
        for m in range(1, 13):
            if y == 2017 and m < 11:
                continue
            if y == 2019 and m > 5:
                continue
            train_dates.append((y, m))
            
    for y, m in train_dates:
        records = generate_records_for_month(y, m, records_count=2200)
        filename = f"invoices-{y}-{m:02d}.json"
        filepath = os.path.join(TRAIN_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(records, f)
        print(f"  -> Generated {filename} ({len(records)} records)")
        
    print("[*] Generating production invoice dataset (2019-06 through 2019-08)...")
    prod_dates = [(2019, 6), (2019, 7), (2019, 8)]
    for y, m in prod_dates:
        # Slight drift in pricing / views for production monitoring demonstration
        records = generate_records_for_month(y, m, records_count=2600)
        filename = f"invoices-{y}-{m:02d}.json"
        filepath = os.path.join(PROD_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(records, f)
        print(f"  -> Generated {filename} ({len(records)} records)")

    print("[+] All sample datasets successfully generated.")

if __name__ == "__main__":
    build_datasets()
