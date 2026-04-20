import pandas as pd
import numpy as np

# DATASET_PATH = "final_features_dataset.csv"
DATASET_PATH=r"D:\major-project\TRAIL-4\ml_dataset\features.csv"

df = pd.read_csv(DATASET_PATH)

FEATURE_COLS = [
    "area",
    "aspect_ratio",
    "mean_intensity",
    "std_intensity",
    "skewness",
    "kurtosis"
]

X = df[FEATURE_COLS].values
y = df["label"].values

def get_true_label(feature_vector):
    diff = np.linalg.norm(X - feature_vector, axis=1)
    idx = diff.argmin()
    return int(y[idx])
