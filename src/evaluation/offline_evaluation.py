import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset used for training
df = pd.read_csv("ml_dataset/features.csv")

X = df.drop(columns=["label"])
y = df["label"]

# Load models
rf = joblib.load("models/rf.pkl")
knn = joblib.load("models/knn.pkl")
scaler = joblib.load("models/scaler.pkl")

# Scale features
X_scaled = scaler.transform(X)

# Ensemble prediction
rf_prob = rf.predict_proba(X_scaled)[:, 1]
knn_prob = knn.predict_proba(X_scaled)[:, 1]
avg_prob = (rf_prob + knn_prob) / 2
y_pred = (avg_prob > 0.55).astype(int)

# Metrics
metrics = {
    "Accuracy": accuracy_score(y, y_pred),
    "Precision": precision_score(y, y_pred),
    "Recall": recall_score(y, y_pred),
    "F1-Score": f1_score(y, y_pred)
}

cm = confusion_matrix(y, y_pred, labels=[0, 1])

# Save results
pd.DataFrame({"Metric": metrics.keys(), "Value": metrics.values()}) \
  .to_csv("offline_model_metrics.csv", index=False)

pd.DataFrame(cm).to_csv("offline_confusion_matrix.csv", index=False)

print("[SUCCESS] Offline model performance saved successfully")
