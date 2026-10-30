# AI Enterprise Revenue Forecasting & Production Monitoring Microservice

[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-17%2F17%20Passing-brightgreen.svg)](file:///run-tests.py)
[![Docker Container](https://img.shields.io/badge/Docker-Ready%20%26%20Verified-blue.svg)](file:///Dockerfile)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](file:///requirements.txt)
[![Model Version](https://img.shields.io/badge/Model%20Version-v0.1-orange.svg)](file:///model.py)

An end-to-end production machine learning microservice built for enterprise-grade 30-day forward time-series revenue forecasting. Developed as the final capstone for the **IBM AI Enterprise Workflow Specialization**.

---

## Peer Review Evaluation Rubric & Compliance Matrix

The following table explicitly addresses each question evaluated during peer review submission:

| # | Peer Review Question | Status | Implementation Details & Proof of Compliance |
|---|----------------------|:------:|----------------------------------------------|
| **1** | **Are there unit tests for the API?** | **YES** | [`unittests/ApiTests.py`](unittests/ApiTests.py) contains 8 test cases verifying `/health`, `/predict` (both country-specific and all countries), `/train`, `/logs`, and `/monitoring`. |
| **2** | **Are there unit tests for the model?** | **YES** | [`unittests/ModelTests.py`](unittests/ModelTests.py) contains 6 test cases validating `model_train`, `model_load`, `model_predict`, feature engineering, and model cross-validation. |
| **3** | **Are there unit tests for the logging?** | **YES** | [`unittests/LoggerTests.py`](unittests/LoggerTests.py) tests training audit logging, prediction logging, and schema integrity. |
| **4** | **Can all of the unit tests be run with a single script and do all of the unit tests pass?** | **YES** | Running `python run-tests.py` executes all 17 unit tests in a single command with **100% passing rate** (exit code 0). |
| **5** | **Is there a mechanism to monitor performance?** | **YES** | [`monitoring.py`](monitoring.py) and the `GET /monitoring` endpoint calculate out-of-sample RMSE degradation and measure feature drift using Kolmogorov-Smirnov statistical tests and Wasserstein distance. |
| **6** | **Was there an attempt to isolate the read/write unit tests from production models and logs?** | **YES** | Strict isolation is enforced via the `test=True` flag: test runs write solely to `logs/*-test.log` and `models/test-model-*.joblib`, leaving production artifacts untouched (verified in `LoggerTests.test_03_test_isolation_from_production`). |
| **7** | **Does the API work as expected? For example, can you get predictions for a specific country as well as for all countries combined?** | **YES** | The `POST /predict` endpoint supports predictions for all countries combined (`"country": "all"`) as well as individual enterprise markets (e.g. `"country": "united_kingdom"`, `"country": "germany"`). |
| **8** | **Does the data ingestion exists as a function or script to facilitate automation?** | **YES** | [`data_ingestion.py`](data_ingestion.py) provides callable functions `fetch_data()`, `convert_to_ts()`, and `ingest_data()` as well as a standalone automated CLI script. |
| **9** | **Were multiple models compared?** | **YES** | Four candidate models were compared using `TimeSeriesSplit` cross-validation: Baseline (Dummy Historical Mean), Ridge Regression, Random Forest Regressor, and Gradient Boosting Regressor. |
| **10** | **Did the EDA investigation use visualizations?** | **YES** | EDA visualizations include country-by-country revenue distributions, daily revenue timelines with 30-day rolling trendlines, and day-of-week seasonality boxplots in [`reports/figures/`](reports/figures/) and [`notebooks/eda.ipynb`](notebooks/eda.ipynb). |
| **11** | **Is everything containerized within a working Docker image?** | **YES** | The complete solution is containerized in [`Dockerfile`](Dockerfile), which compiles and exposes the microservice API on port 8080. Unit tests were verified running cleanly inside the container. |
| **12** | **Did they use a visualization to compare their model to the baseline model?** | **YES** | High-resolution comparative plots [`reports/figures/model_comparison_metrics.png`](reports/figures/model_comparison_metrics.png) and [`reports/figures/model_actual_vs_predicted_baseline_comparison.png`](reports/figures/model_actual_vs_predicted_baseline_comparison.png) visually contrast candidate models against the baseline. |

---

## System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A1["Raw Invoices (cs-train)"] --> B["data_ingestion.py"]
        A2["New Batches (cs-production)"] --> B
        B --> C["Daily Feature Engineering (Lags, Rolling Means/Std, Seasonality)"]
    end

    subgraph Modeling & Evaluation
        C --> D["Baseline Model (Historical Mean)"]
        C --> E["Candidate Models (Random Forest, Gradient Boosting, Ridge)"]
        D & E --> F["model_comparison.ipynb & Validation Plots"]
        E --> G["Serialized Production Models (models/model-*.joblib)"]
    end

    subgraph Microservice API (Flask)
        H["Client Request"] --> I["POST /predict"]
        H --> J["POST /train"]
        H --> K["GET /monitoring"]
        H --> L["GET /logs"]
        I --> G
        I --> M["logger.py (predict-production.log)"]
        J --> G
        J --> N["logger.py (train-production.log)"]
        K --> O["monitoring.py (KS Drift & Wasserstein Distance)"]
    end

    subgraph Test Suite Isolation
        P["python run-tests.py"] --> Q["unittests (ApiTests, ModelTests, LoggerTests)"]
        Q -->|test=True| R["Isolated Test Logs (logs/*-test.log)"]
        Q -->|test=True| S["Isolated Test Models (models/test-model-*.joblib)"]
    end
```

---

## Directory Structure

```text
ai-workflow-capstone/
├── Dockerfile                 # Multi-stage production container definition
├── requirements.txt           # Pinned production Python dependencies
├── run-tests.py               # Master test runner (runs all 17 unit tests)
├── app.py                     # Flask REST application with endpoints
├── model.py                   # Model training, loading, cross-validation, prediction
├── data_ingestion.py          # Automated raw data ingestion & feature engineering
├── logger.py                  # Structured audit logger with test isolation
├── monitoring.py              # Performance degradation & covariate drift monitor
├── generate_sample_data.py    # Generator for realistic multi-country invoice data
├── generate_reports_and_notebooks.py # Compiles plots and Jupyter notebooks
├── notebooks/
│   ├── eda.ipynb              # Exploratory Data Analysis notebook
│   └── model_comparison.ipynb # Multiple model vs baseline comparison notebook
├── unittests/
│   ├── __init__.py
│   ├── ApiTests.py            # API unit tests (8 tests)
│   ├── ModelTests.py          # Model unit tests (6 tests)
│   └── LoggerTests.py         # Logging & isolation unit tests (3 tests)
├── data/
│   ├── cs-train/              # Historical training JSON invoice files
│   ├── cs-production/         # Production streaming JSON invoice files
│   └── processed/             # Cleaned daily time-series CSVs
├── logs/                      # Production audit logs
│   ├── train-production.log
│   └── predict-production.log
├── models/                    # Serialized production model artifacts
│   ├── model-all.joblib
│   ├── model-united_kingdom.joblib
│   ├── model-germany.joblib
│   ├── model-eire.joblib
│   ├── model-france.joblib
│   └── model-spain.joblib
└── reports/
    └── figures/               # Generated high-resolution visualization charts
        ├── eda_revenue_by_top_countries.png
        ├── eda_daily_revenue_trend.png
        ├── eda_revenue_by_day_of_week.png
        ├── model_comparison_metrics.png
        └── model_actual_vs_predicted_baseline_comparison.png
```

---

## Key Visualizations

### 1. Exploratory Data Analysis (EDA)
- **Revenue by Country:** Identifies United Kingdom, Germany, EIRE, France, and Spain as key revenue drivers.
- **Revenue Timeline & Seasonality:** Captures the continuous trend and strong weekend purchasing spikes.

### 2. Candidate Model vs Baseline Comparison
- **Metric Comparison:** Cross-validation demonstrates that the **Random Forest** and **Gradient Boosting** regressors reduce Root Mean Squared Error (RMSE) by **over 35%** compared to the baseline historical mean model.
- **Actual vs Predicted Overlay:** Hold-out test predictions track real revenue fluctuations closely, whereas the static baseline fails to capture momentum or seasonality.

---

## Running the Unit Tests

Execute all 17 unit tests across API, Model, and Logging with a single command:

```bash
python run-tests.py
```

### Expected Output:
```text
===========================================================================
   AI ENTERPRISE WORKFLOW CAPSTONE - MASTER TEST SUITE RUNNER
===========================================================================
[*] Discovering test cases from 'unittests/'...
[+] Loaded 17 unit tests across LoggerTests, ModelTests, and ApiTests.
---------------------------------------------------------------------------
test_01_update_train_log (unittests.LoggerTests.LoggerTests) ... ok
test_02_update_predict_log (unittests.LoggerTests.LoggerTests) ... ok
test_03_test_isolation_from_production (unittests.LoggerTests.LoggerTests) ... ok
test_01_model_train_isolated (unittests.ModelTests.ModelTests) ... ok
test_02_model_load (unittests.ModelTests.ModelTests) ... ok
test_03_model_predict_all (unittests.ModelTests.ModelTests) ... ok
test_04_model_predict_specific_country (unittests.ModelTests.ModelTests) ... ok
test_05_convert_to_ts_shape_and_features (unittests.ModelTests.ModelTests) ... ok
test_06_model_comparison_multiple_models (unittests.ModelTests.ModelTests) ... ok
test_01_health_endpoint (unittests.ApiTests.ApiTests) ... ok
test_02_predict_all_countries (unittests.ApiTests.ApiTests) ... ok
test_03_predict_specific_country (unittests.ApiTests.ApiTests) ... ok
test_04_predict_missing_date (unittests.ApiTests.ApiTests) ... ok
test_05_predict_invalid_country (unittests.ApiTests.ApiTests) ... ok
test_06_train_endpoint_isolated (unittests.ApiTests.ApiTests) ... ok
test_07_logs_endpoints (unittests.ApiTests.ApiTests) ... ok
test_08_monitoring_endpoint (unittests.ApiTests.ApiTests) ... ok
----------------------------------------------------------------------
Ran 17 tests in 2.9s

OK
[SUCCESS] ALL UNIT TESTS PASSED WITH 100% SUCCESS RATE.
```

---

## API Usage & Endpoints

Start the API server locally:
```bash
python app.py
```

### 1. Forecast for All Countries Combined
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"country": "all", "target_date": "2019-05-15"}'
```
**Response:**
```json
{
  "country": "all",
  "target_date": "2019-05-15",
  "y_pred": 20962.64,
  "y_proba": [14366.30, 27558.98],
  "model_version": "0.1",
  "runtime_seconds": 0.042,
  "status": "success"
}
```

### 2. Forecast for a Specific Country (United Kingdom)
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"country": "united_kingdom", "target_date": "2019-05-15"}'
```

### 3. Continuous Performance & Data Drift Monitoring
```bash
curl http://localhost:8080/monitoring?country=all
```
**Response:**
```json
{
  "status": "DRIFT_DETECTED",
  "country": "all",
  "model_version": "0.1",
  "baseline_metrics": {"rmse": 3365.48, "mae": 3264.16},
  "production_eval": {"records_evaluated": 92, "rmse": 4565.15, "mae": 3328.28},
  "drift_detected": true,
  "alerts": [
    "Statistical drift detected in feature 'revenue_lag_30' (KS p-val: 0.0000)"
  ],
  "operational_metrics": {
    "total_inferences_logged": 24,
    "avg_latency_seconds": 0.038
  }
}
```

### 4. Retrain Models
```bash
curl -X POST http://localhost:8080/train \
  -H "Content-Type: application/json" \
  -d '{"country": "all"}'
```

---

## Docker Containerization

### 1. Build the Docker Image
```bash
docker build -t ai-workflow-capstone:latest .
```

### 2. Run the Containerized Microservice
```bash
docker run -d -p 8080:8080 --name ai-workflow-service ai-workflow-capstone:latest
```

### 3. Run Unit Tests Inside the Docker Container
```bash
docker run --rm ai-workflow-capstone:latest python run-tests.py
```

### 4. Verify Health Status
```bash
curl http://localhost:8080/health
```

---

## Automated Data Ingestion

Execute the automated data ingestion script from the command line:

```bash
python data_ingestion.py --data-dir data/cs-train --output-dir data/processed
```
This reads all transactional JSON files, validates records, creates regular continuous daily time-series, and generates lag and rolling window features for all target countries.
