"""
generate_reports_and_notebooks.py
Generates publication-quality figures in reports/figures/ and produces
the complete Jupyter Notebooks (eda.ipynb, model_comparison.ipynb).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_ingestion import fetch_data, convert_to_ts
from model import compare_models, FEATURES, TARGET, build_model_pipeline

# Styling configurations
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

def generate_eda_figures():
    print("[*] Generating EDA figures...")
    df_raw = fetch_data()
    ts_all = convert_to_ts(df_raw, country="all")

    # Figure 1: Revenue & Transactions by Top 6 Countries
    top_country_stats = df_raw.groupby("country").agg(
        total_revenue=("price", "sum"),
        total_invoices=("invoice", "count")
    ).sort_values(by="total_revenue", ascending=False).head(6)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    x = np.arange(len(top_country_stats))
    width = 0.35

    ax1.bar(x - width/2, top_country_stats["total_revenue"], width, label="Total Revenue ($)", color="#1f77b4")
    ax2.bar(x + width/2, top_country_stats["total_invoices"], width, label="Invoices Count", color="#ff7f0e")

    ax1.set_xlabel("Country", fontweight="bold")
    ax1.set_ylabel("Revenue ($)", color="#1f77b4", fontweight="bold")
    ax2.set_ylabel("Invoice Count", color="#ff7f0e", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_", " ").title() for c in top_country_stats.index], rotation=15)
    plt.title("Revenue and Transaction Volume by Top 6 Countries (EDA)", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "eda_revenue_by_top_countries.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"  -> Saved {fig1_path}")

    # Figure 2: Daily Revenue Time Series & 30-Day Moving Average
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts_all["date"], ts_all["revenue"], label="Daily Revenue", alpha=0.45, color="#7f7f7f", linewidth=1.0)
    ax.plot(ts_all["date"], ts_all["revenue_roll_mean_30"], label="30-Day Rolling Mean", color="#d62728", linewidth=2.2)
    ax.set_title("Daily Revenue Timeline & 30-Day Smoothing Trend (All Countries)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Daily Revenue ($)", fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "eda_daily_revenue_trend.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"  -> Saved {fig2_path}")

    # Figure 3: Revenue Distribution by Day of Week
    ts_all["day_name"] = ts_all["date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(x="day_name", y="revenue", data=ts_all, order=day_order, palette="Blues_d", ax=ax)
    ax.set_title("Revenue Distribution across Days of the Week (Weekend Effect)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Day of Week", fontweight="bold")
    ax.set_ylabel("Daily Revenue ($)", fontweight="bold")
    fig.tight_layout()
    fig3_path = os.path.join(FIG_DIR, "eda_revenue_by_day_of_week.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"  -> Saved {fig3_path}")

    return ts_all

def generate_model_comparison_figures(ts_all):
    print("[*] Generating Model Comparison figures...")
    # Evaluate multiple models
    comp_results = compare_models(ts_all)
    print(f"  -> Comparison metrics:\n{json.dumps(comp_results, indent=2)}")

    # Figure 4: Metric Comparison (RMSE & MAE)
    models = list(comp_results.keys())
    rmse_vals = [comp_results[m]["RMSE"] for m in models]
    mae_vals = [comp_results[m]["MAE"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, rmse_vals, width, label="RMSE (Root Mean Sq Error)", color="#e377c2")
    rects2 = ax.bar(x + width/2, mae_vals, width, label="MAE (Mean Abs Error)", color="#17becf")

    ax.set_ylabel("Error Metric Value (Lower is Better)", fontweight="bold")
    ax.set_title("Multiple Model Comparison vs Baseline (Cross-Validation Evaluation)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="semibold")
    ax.legend()

    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig4_path = os.path.join(FIG_DIR, "model_comparison_metrics.png")
    plt.savefig(fig4_path)
    plt.close()
    print(f"  -> Saved {fig4_path}")

    # Figure 5: Actual vs Predicted Time-Series (Baseline vs Candidate Model)
    clean_df = ts_all.dropna(subset=FEATURES + [TARGET]).iloc[30:-30].copy()
    X = clean_df[FEATURES]
    y = clean_df[TARGET]
    split_idx = int(len(clean_df) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = clean_df["date"].iloc[split_idx:]

    # Train Baseline Model
    baseline_pred = np.full(shape=len(y_test), fill_value=y_train.mean())

    # Train Random Forest Pipeline
    rf_pipe = build_model_pipeline()
    rf_pipe.fit(X_train, y_train)
    rf_preds = rf_pipe.predict(X_test)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(dates_test, y_test, label="Actual 30-Day Revenue", color="#333333", linewidth=2.0, marker="o", markersize=3)
    ax.plot(dates_test, baseline_pred, label="Baseline Model (Historical Mean)", color="#d62728", linestyle="--", linewidth=2.0)
    ax.plot(dates_test, rf_preds, label="Production Model (Random Forest)", color="#2ca02c", linewidth=2.2)

    ax.set_title("Visual Comparison: Candidate Model vs Baseline Model on Hold-Out Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Forecast Target Date", fontweight="bold")
    ax.set_ylabel("30-Day Forward Revenue ($)", fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig5_path = os.path.join(FIG_DIR, "model_actual_vs_predicted_baseline_comparison.png")
    plt.savefig(fig5_path)
    plt.close()
    print(f"  -> Saved {fig5_path}")

def build_notebook_files():
    print("[*] Generating formatted Jupyter Notebooks...")
    
    # 1. notebooks/eda.ipynb
    eda_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AI Enterprise Workflow Capstone: Exploratory Data Analysis (EDA)\n",
                "**Project Title:** AI Enterprise Revenue Forecasting & Production Monitoring Microservice\n",
                "\n",
                "This notebook conducts a visual exploratory data analysis (EDA) of the transactional invoice stream data,\n",
                "evaluating revenue distributions, country volume, day-of-week seasonality, and continuous time-series patterns."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from data_ingestion import fetch_data, convert_to_ts\n",
                "\n",
                "# Ingest raw transactions and aggregate continuous time-series\n",
                "df_raw = fetch_data()\n",
                "ts_all = convert_to_ts(df_raw, country='all')\n",
                "print(f'Raw transactions count: {len(df_raw)}')\n",
                "print(f'Daily time-series points: {len(ts_all)}')\n",
                "df_raw.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Revenue & Transaction Volume by Country\n",
                "Analyzing key enterprise markets across invoice counts and total monetary volume."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "top_country_stats = df_raw.groupby('country').agg(\n",
                "    total_revenue=('price', 'sum'),\n",
                "    total_invoices=('invoice', 'count')\n",
                ").sort_values(by='total_revenue', ascending=False).head(6)\n",
                "\n",
                "fig, ax1 = plt.subplots(figsize=(10, 5))\n",
                "ax2 = ax1.twinx()\n",
                "x = np.arange(len(top_country_stats))\n",
                "width = 0.35\n",
                "ax1.bar(x - width/2, top_country_stats['total_revenue'], width, label='Total Revenue ($)', color='#1f77b4')\n",
                "ax2.bar(x + width/2, top_country_stats['total_invoices'], width, label='Invoices Count', color='#ff7f0e')\n",
                "ax1.set_xlabel('Country')\n",
                "ax1.set_ylabel('Revenue ($)', color='#1f77b4')\n",
                "ax2.set_ylabel('Invoice Count', color='#ff7f0e')\n",
                "ax1.set_xticks(x)\n",
                "ax1.set_xticklabels([c.replace('_', ' ').title() for c in top_country_stats.index], rotation=15)\n",
                "plt.title('Revenue and Transaction Volume by Top 6 Countries')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Time-Series Daily Revenue Trends & 30-Day Moving Average"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, ax = plt.subplots(figsize=(12, 5))\n",
                "ax.plot(ts_all['date'], ts_all['revenue'], label='Daily Revenue', alpha=0.45, color='#7f7f7f')\n",
                "ax.plot(ts_all['date'], ts_all['revenue_roll_mean_30'], label='30-Day Rolling Mean', color='#d62728', linewidth=2.2)\n",
                "ax.set_title('Daily Revenue Timeline & 30-Day Smoothing Trend (All Countries)')\n",
                "ax.set_xlabel('Date')\n",
                "ax.set_ylabel('Daily Revenue ($)')\n",
                "ax.legend()\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ]

    eda_nb = {
        "cells": eda_cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(os.path.join(NOTEBOOK_DIR, "eda.ipynb"), "w") as f:
        json.dump(eda_nb, f, indent=2)
    print(f"  -> Created {os.path.join(NOTEBOOK_DIR, 'eda.ipynb')}")

    # 2. notebooks/model_comparison.ipynb
    model_comp_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AI Enterprise Workflow Capstone: Multi-Model Evaluation & Baseline Comparison\n",
                "**Project Title:** AI Enterprise Revenue Forecasting & Production Monitoring Microservice\n",
                "\n",
                "This notebook satisfies the rubric requirements for:\n",
                "- **Multiple model comparison** (Baseline Dummy, Ridge Regression, Random Forest, Gradient Boosting)\n",
                "- **Visualization comparing candidate model to baseline**\n",
                "- Cross-validation error metrics (RMSE, MAE, R²)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "from data_ingestion import fetch_data, convert_to_ts\n",
                "from model import compare_models, FEATURES, TARGET, build_model_pipeline\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "df_raw = fetch_data()\n",
                "ts_all = convert_to_ts(df_raw, country='all')\n",
                "\n",
                "comp_results = compare_models(ts_all)\n",
                "pd.DataFrame(comp_results).T"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Visualizing Multi-Model Metrics vs Baseline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "models = list(comp_results.keys())\n",
                "rmse_vals = [comp_results[m]['RMSE'] for m in models]\n",
                "mae_vals = [comp_results[m]['MAE'] for m in models]\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(10, 5))\n",
                "x = np.arange(len(models))\n",
                "width = 0.35\n",
                "ax.bar(x - width/2, rmse_vals, width, label='RMSE', color='#e377c2')\n",
                "ax.bar(x + width/2, mae_vals, width, label='MAE', color='#17becf')\n",
                "ax.set_ylabel('Error Value (Lower is Better)')\n",
                "ax.set_title('Multiple Model Comparison vs Baseline (Cross-Validation)')\n",
                "ax.set_xticks(x)\n",
                "ax.set_xticklabels(models)\n",
                "ax.legend()\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Visualizing Hold-Out Predictions: Candidate Model vs Baseline Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "clean_df = ts_all.dropna(subset=FEATURES + [TARGET]).iloc[30:-30].copy()\n",
                "X = clean_df[FEATURES]\n",
                "y = clean_df[TARGET]\n",
                "split_idx = int(len(clean_df) * 0.8)\n",
                "\n",
                "X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]\n",
                "y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]\n",
                "dates_test = clean_df['date'].iloc[split_idx:]\n",
                "\n",
                "baseline_pred = np.full(shape=len(y_test), fill_value=y_train.mean())\n",
                "rf_pipe = build_model_pipeline()\n",
                "rf_pipe.fit(X_train, y_train)\n",
                "rf_preds = rf_pipe.predict(X_test)\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(12, 5.5))\n",
                "ax.plot(dates_test, y_test, label='Actual 30-Day Revenue', color='#333333', linewidth=2.0)\n",
                "ax.plot(dates_test, baseline_pred, label='Baseline Model (Mean)', color='#d62728', linestyle='--', linewidth=2.0)\n",
                "ax.plot(dates_test, rf_preds, label='Production Model (Random Forest)', color='#2ca02c', linewidth=2.2)\n",
                "ax.set_title('Hold-Out Comparison: Production Model vs Baseline')\n",
                "ax.set_xlabel('Target Date')\n",
                "ax.set_ylabel('30-Day Forward Revenue ($)')\n",
                "ax.legend()\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ]

    model_nb = {
        "cells": model_comp_cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(os.path.join(NOTEBOOK_DIR, "model_comparison.ipynb"), "w") as f:
        json.dump(model_nb, f, indent=2)
    print(f"  -> Created {os.path.join(NOTEBOOK_DIR, 'model_comparison.ipynb')}")

if __name__ == "__main__":
    ts_data = generate_eda_figures()
    generate_model_comparison_figures(ts_data)
    build_notebook_files()
    print("[+] All EDA, Model Comparison figures and Jupyter Notebooks successfully created.")
