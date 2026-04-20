import os
import numpy as np
import cv2
import pandas as pd

from src.inference.feature_extraction import extract_features

BASE_DIR = r"D:\major-project\TRAIL-4\oil-spill"
IMG_DIR = os.path.join(BASE_DIR, "train", "images")
MASK_DIR = os.path.join(BASE_DIR, "train", "labels")

feature_rows = []

image_files = os.listdir(IMG_DIR)

for img_name in image_files:
    img_path = os.path.join(IMG_DIR, img_name)

    # Match mask filename correctly
    mask_name = os.path.splitext(img_name)[0] + ".png"
    mask_path = os.path.join(MASK_DIR, mask_name)

    if not os.path.exists(mask_path):
        print(f"[WARN] Mask missing for {img_name}, skipping")
        continue

    # Read and binarize mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype("uint8")

    # -----------------------------
    # 1️⃣ OIL REGIONS (label = 1)
    # -----------------------------
    oil_features, _ = extract_features(img_path, mask)

    for f in oil_features:
        feature_rows.append([
            f[0],  # area
            f[1],  # aspect_ratio
            f[2],  # mean_intensity
            f[3],  # std_intensity
            f[4],  # skewness
            f[5],  # kurtosis
            1      # label = oil
        ])

    # --------------------------------
    # 2️⃣ NON-OIL REGIONS (label = 0)
    # --------------------------------
    inv_mask = 1 - mask
    bg_features, _ = extract_features(img_path, inv_mask)

    # (Optional) limit background samples to avoid dominance
    bg_features = bg_features[:len(oil_features)]

    for f in bg_features:
        feature_rows.append([
            f[0],
            f[1],
            f[2],
            f[3],
            f[4],
            f[5],
            0      # label = non-oil
        ])

# Create DataFrame
columns = [
    "area",
    "aspect_ratio",
    "mean_intensity",
    "std_intensity",
    "skewness",
    "kurtosis",
    "label"
]

df = pd.DataFrame(feature_rows, columns=columns)

os.makedirs("ml_dataset", exist_ok=True)
csv_path = "ml_dataset/features.csv"
df.to_csv(csv_path, index=False)

print("[SUCCESS] ML dataset CSV created successfully")
print("CSV path:", csv_path)
print("Shape:", df.shape)
print("Label distribution:")
print(df["label"].value_counts())
