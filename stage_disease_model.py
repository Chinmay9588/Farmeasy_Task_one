# stage_disease_model.py
import joblib
import os
from datetime import datetime
import pandas as pd
from disease_data import stage_durations, stage_diseases, disease_info

MODEL_PATH = "model_pipeline.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"

def calculate_days_since_sowing(sowing_date: datetime, current_date: datetime) -> int:
    return (current_date - sowing_date).days

def predict_stage(days_since: int) -> str:
    for stage, (low, high) in stage_durations.items():
        if low <= days_since <= high:
            return stage
    if days_since < 0:
        return "Unknown"
    return "Maturity"

def get_possible_diseases(stage: str):
    return stage_diseases.get(stage, [])

def explain_disease(disease: str):
    return disease_info.get(disease, "No detailed info available.")

def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def load_label_encoder():
    if not os.path.exists(LABEL_ENCODER_PATH):
        return None
    return joblib.load(LABEL_ENCODER_PATH)

def predict_disease_from_features(features: dict):
    """
    features: dict mapping feature_name -> value.
    Expected to include numeric keys (temperature, humidity, days_since_sowing, etc.)
    """
    pipeline = load_pipeline()
    if pipeline is None:
        return {"error": "Model not found. Run `python model_train.py` first."}

    # Build single-row DataFrame
    X = pd.DataFrame([features])

    # Ensure DataFrame has all columns the pipeline expects. If columns are
    # missing (e.g. 'days_since_sowing', 'Crop', 'Crop_Stage'), add them with
    # NaN/None so the pipeline's imputers can handle them.
    try:
        preprocessor = pipeline.named_steps.get('preprocessor')
        if preprocessor is not None:
            expected_cols = []
            for name, transformer, cols in preprocessor.transformers:
                # cols can be a list of names, a single name, or 'remainder'
                if cols == 'remainder':
                    continue
                if isinstance(cols, (list, tuple)):
                    expected_cols.extend(cols)
                else:
                    expected_cols.append(cols)

            # Add any missing expected columns with NaN/None
            for c in expected_cols:
                if c not in X.columns:
                    X[c] = None

    except Exception:
        # If anything goes wrong while introspecting the pipeline, continue
        # and let the pipeline raise a helpful error during prediction.
        pass

    try:
        pred_enc = pipeline.predict(X)
    except Exception as e:
        return {"error": f"Prediction error: {e}"}

    le = load_label_encoder()
    if le is not None:
        try:
            pred_label = le.inverse_transform(pred_enc.astype(int))[0]
            return {"prediction": pred_label}
        except Exception:
            # fallback
            return {"prediction": str(pred_enc[0])}
    else:
        return {"prediction": str(pred_enc[0])}
