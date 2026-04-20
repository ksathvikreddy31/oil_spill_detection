
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("evaluation_dataset.csv")

# Balanced sampling for stable metrics
df0 = df[df["ground_truth"] == 0].sample(n=300, random_state=42)
df1 = df[df["ground_truth"] == 1].sample(n=300, random_state=42)
df = pd.concat([df0, df1])

y_true = df["ground_truth"]
y_pred = df["ml_prediction"]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n[STAT] FINAL SYSTEM PERFORMANCE\n")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1 Score : {f1:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
