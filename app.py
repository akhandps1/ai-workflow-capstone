"""
app.py
Production Flask REST API for IBM AI Enterprise Workflow Capstone.
Provides endpoints for inference (/predict), retraining (/train), audit logs (/logs),
and continuous model & data monitoring (/monitoring).
"""

import os
from flask import Flask, request, jsonify
from model import model_predict, model_train, MODEL_VERSION
from logger import read_train_logs, read_predict_logs
from monitoring import monitor_performance

app = Flask(__name__)

SUPPORTED_COUNTRIES = ["all", "united_kingdom", "germany", "eire", "france", "spain"]

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    """
    Service healthcheck and capability metadata.
    """
    return jsonify({
        "status": "healthy",
        "service": "AI Enterprise Revenue Prediction Service",
        "model_version": MODEL_VERSION,
        "supported_countries": SUPPORTED_COUNTRIES
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Generates 30-day forward revenue prediction.
    Accepts:
        {
            "country": "all" | "united_kingdom" | ...,
            "target_date": "YYYY-MM-DD",
            "test": true | false (optional)
        }
    """
    payload = request.get_json()
    if not payload:
        return jsonify({"status": "error", "message": "Missing JSON request body"}), 400

    country = payload.get("country", "all")
    target_date = payload.get("target_date")
    test_mode = bool(payload.get("test", False))

    if not target_date:
        return jsonify({"status": "error", "message": "Field 'target_date' is required (YYYY-MM-DD)"}), 400

    # Country validation
    country_norm = country.strip().lower().replace(" ", "_")
    if country_norm not in SUPPORTED_COUNTRIES and country_norm != "all":
        return jsonify({
            "status": "error",
            "message": f"Unsupported country '{country}'. Supported: {SUPPORTED_COUNTRIES}"
        }), 400

    try:
        result = model_predict(country=country_norm, target_date=target_date, test=test_mode)
        result["status"] = "success"
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    """
    Triggers model training.
    Accepts:
        {
            "country": "all" | "united_kingdom" | ... (optional),
            "test": true | false (optional)
        }
    """
    payload = request.get_json() or {}
    country = payload.get("country")
    test_mode = bool(payload.get("test", False))

    try:
        trained_bundle = model_train(country=country, test=test_mode)
        summary = {k: v.get("eval_test", {}) for k, v in trained_bundle.items()}
        return jsonify({
            "status": "success",
            "message": "Models successfully trained",
            "models_trained": list(trained_bundle.keys()),
            "eval_summary": summary,
            "test_mode": test_mode
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/logs/predict", methods=["GET"])
def get_predict_logs():
    """
    Retrieves prediction audit logs.
    Query param ?test=true retrieves test logs instead of production logs.
    """
    test_mode = request.args.get("test", "false").lower() == "true"
    entries = read_predict_logs(test=test_mode)
    return jsonify({
        "status": "success",
        "test_mode": test_mode,
        "count": len(entries),
        "logs": entries[-100:]  # Return most recent 100 entries
    }), 200

@app.route("/logs/train", methods=["GET"])
def get_train_logs():
    """
    Retrieves training audit logs.
    Query param ?test=true retrieves test logs instead of production logs.
    """
    test_mode = request.args.get("test", "false").lower() == "true"
    entries = read_train_logs(test=test_mode)
    return jsonify({
        "status": "success",
        "test_mode": test_mode,
        "count": len(entries),
        "logs": entries[-100:]
    }), 200

@app.route("/monitoring", methods=["GET"])
def monitoring():
    """
    Evaluates current model performance against new data & checks feature drift.
    Query params:
        country: 'all' | country_name
        test: true | false
    """
    country = request.args.get("country", "all")
    test_mode = request.args.get("test", "false").lower() == "true"

    try:
        report = monitor_performance(country=country, test=test_mode)
        return jsonify(report), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
