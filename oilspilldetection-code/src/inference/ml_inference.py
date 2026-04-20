import os
import joblib
import numpy as np

# Use absolute paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(BASE_DIR, "models")

rf = joblib.load(os.path.join(MODELS_DIR, "rf.pkl"))
knn = joblib.load(os.path.join(MODELS_DIR, "knn.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

# Tuned threshold for stability
THRESHOLD = 0.55

def classify_regions(features):
    features_scaled = scaler.transform(features)

    rf_prob = rf.predict_proba(features_scaled)[:, 1]
    knn_prob = knn.predict_proba(features_scaled)[:, 1]

    # Soft ensemble
    avg_prob = (rf_prob + knn_prob) / 2

    preds = (avg_prob > THRESHOLD).astype(int)
    return preds
