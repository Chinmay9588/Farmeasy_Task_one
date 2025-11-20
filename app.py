# app.py

from flask import Flask, request, jsonify, render_template
from datetime import datetime
from stage_disease_model import (
    calculate_days_since_sowing,
    predict_stage,
    get_possible_diseases,
    explain_disease,
    predict_disease_from_features
)
from gemini_integration import GeminiLLM
import os

# Flask App Initialization  (IMPORT MUST COME FIRST!!)
app = Flask(__name__, template_folder="templates", static_folder="static")

# Optional: Gemini API Key
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
gemini_llm = None
if GEMINI_KEY:
    try:
        gemini_llm = GeminiLLM(GEMINI_KEY)
    except Exception as e:
        print("Gemini init error:", e)
        gemini_llm = None


# -------------------- ROUTES -------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------------------------
# PREDICT STAGE + POSSIBLE DISEASES BASED ON DATES
# -----------------------------------------------------
@app.route("/predict_stage_disease", methods=["POST"])
def predict_stage_disease():
    data = request.json or request.form.to_dict()
    sowing_date_str = data.get("sowing_date")
    current_date_str = data.get("current_date")

    if not sowing_date_str:
        return jsonify({"error": "Provide sowing_date in DD-MM-YYYY format."}), 400

    try:
        sow = datetime.strptime(sowing_date_str, "%d-%m-%Y")
        cur = datetime.strptime(current_date_str, "%Y-%m-%d") if current_date_str else datetime.now()
    except Exception:
        return jsonify({"error": "Invalid date format. Use DD-MM-YYYY for sowing_date and YYYY-MM-DD for current_date."}), 400

    days = calculate_days_since_sowing(sow, cur)
    stage = predict_stage(days)
    diseases = get_possible_diseases(stage)
    disease_details = {d: explain_disease(d) for d in diseases}

    return jsonify({
        "days_since_sowing": days,
        "stage": stage,
        "possible_diseases": disease_details
    })


# -----------------------------------------------------
# ML MODEL BASED DISEASE PREDICTION
# -----------------------------------------------------
@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    payload = request.json
    if not payload:
        return jsonify({"error": "Send JSON body."}), 400

    # If user provides explicit features
    features = payload.get("features")

    if not features:
        features = {}

        # Add sensor or numeric inputs
        for key, value in payload.items():
            if key not in ("sowing_date", "current_date", "crop_name", "features"):
                features[key] = value

        # Add days_since_sowing if dates provided
        sowing_date_str = payload.get("sowing_date")
        current_date_str = payload.get("current_date")

        if sowing_date_str:
            try:
                # Use the same parsing and calculation as the /predict_stage_disease
                # route so days_since_sowing is consistent across endpoints.
                sow = datetime.strptime(sowing_date_str, "%d-%m-%Y")
                cur = datetime.strptime(current_date_str, "%Y-%m-%d") if current_date_str else datetime.now()
                # calculate_days_since_sowing is imported from stage_disease_model
                features["days_since_sowing"] = calculate_days_since_sowing(sow, cur)
            except Exception:
                # If parsing fails, leave it out and let the model imputers handle missing values
                pass

    # Run ML model prediction
    result = predict_disease_from_features(features)
    return jsonify(result)


# -----------------------------------------------------
# GEMINI INSIGHTS ENDPOINT (OPTIONAL)
# -----------------------------------------------------
@app.route("/get_insights", methods=["POST"])
def get_insights():
    data = request.json or {}
    disease = data.get("disease")
    stage = data.get("stage")
    crop = data.get("crop_name", "Chickpea")

    if not disease:
        return jsonify({"error": "Provide disease name for insights."}), 400

    if gemini_llm is None:
        return jsonify({"error": "Gemini is not configured. Set GEMINI_API_KEY environment variable."}), 500

    prompt = (
        f"Crop: {crop}\n"
        f"Stage: {stage}\n"
        f"Disease: {disease}\n"
        f"Explain causes, symptoms, and chemical/organic treatments clearly."
    )

    insights = gemini_llm.query_llm(prompt)
    return jsonify({"insights": insights})


# -----------------------------------------------------
# RUN FLASK APP
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
