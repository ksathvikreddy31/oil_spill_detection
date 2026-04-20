import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

LOG_FILE = "inference_logs/inference_features.csv"

def build_metrics_from_logs():
    try:
        df = pd.read_csv(LOG_FILE)
    except:
        df = pd.read_csv(LOG_FILE, header=None)
        df = df.iloc[:, :10]
        df.columns = [
            "image_name","timestamp","area","aspect_ratio","mean_intensity",
            "std_intensity","skewness","kurtosis","ml_prediction","ground_truth"
        ]

    if "ground_truth" not in df.columns:
        return None, None

    # --- ML-ONLY SAFE CLEANING ---

    ml_df = df[["ml_prediction", "ground_truth"]].copy()

    # Force everything numeric
    ml_df["ml_prediction"] = pd.to_numeric(ml_df["ml_prediction"], errors="coerce")
    ml_df["ground_truth"] = pd.to_numeric(ml_df["ground_truth"], errors="coerce")

    # Remove inf and invalid rows
    ml_df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)
    ml_df.dropna(inplace=True)

    if len(ml_df) == 0:
        return None, None

    y_true = ml_df["ground_truth"].astype(int)
    y_pred = ml_df["ml_prediction"].astype(int)

    # --- ML MODEL METRICS ONLY ---

    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0)
        ]
    }

    table = pd.DataFrame(metrics)
    cm = confusion_matrix(y_true, y_pred)

    return table, cm
