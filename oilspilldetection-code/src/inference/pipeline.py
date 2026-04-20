# import os
# import cv2
# import numpy as np
# import pandas as pd
# from datetime import datetime

# from src.inference.ensemble_segmentation import ensemble_mask
# from src.inference.feature_extraction import extract_features
# from src.inference.ml_inference import classify_regions

# LOG_DIR = "inference_logs"
# LOG_FILE = os.path.join(LOG_DIR, "inference_features.csv")

# def run_pipeline(image_path):
#     os.makedirs(LOG_DIR, exist_ok=True)

#     # Read original image
#     original_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     original_color = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)

#     # -------------------------------
#     # 1️⃣ DL ENSEMBLE SEGMENTATION
#     # -------------------------------
#     dl_mask = ensemble_mask(image_path)

#     # 🔥 VERY IMPORTANT: resize mask to original image size
#     dl_mask = cv2.resize(
#         dl_mask,
#         (original_gray.shape[1], original_gray.shape[0]),
#         interpolation=cv2.INTER_NEAREST
#     )

#     # Red overlay for DL prediction
#     dl_overlay = original_color.copy()
#     dl_overlay[dl_mask == 1] = [0, 0, 255]

#     # -------------------------------
#     # 2️⃣ FEATURE EXTRACTION
#     # -------------------------------
#     features, contours = extract_features(image_path, dl_mask)

#     final_overlay = original_color.copy()
#     oil_detected = False

#     if len(features) == 0:
#         decision_text = "NO OIL DETECTED"
#         return original_color, dl_overlay, final_overlay, decision_text

#     # -------------------------------
#     # 3️⃣ ML CLASSIFICATION
#     # -------------------------------
#     # -------------------------------
# # 3️⃣ ML CLASSIFICATION
# # -------------------------------
#     predictions = classify_regions(features)

#     oil_detected = False
#     verified_mask = np.zeros_like(dl_mask)

#     for pred, cnt in zip(predictions, contours):
#         if pred == 1:
#             oil_detected = True
#             cv2.drawContours(verified_mask, [cnt], -1, 1, -1)

#     # Draw verified oil regions in GREEN
#     final_overlay = original_color.copy()
#     final_overlay[verified_mask == 1] = [0, 255, 0]


#     # -------------------------------
#     # 4️⃣ FINAL DECISION
#     # -------------------------------
#     decision_text = "OIL DETECTED" if oil_detected else "NO OIL DETECTED"

#     # -------------------------------
#     # 5️⃣ LOG FEATURES + PREDICTIONS
#     # -------------------------------
#     rows = []
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     for f, pred in zip(features, predictions):
#         rows.append([
#             os.path.basename(image_path),
#             timestamp,
#             f[0],  # area
#             f[1],  # aspect_ratio
#             f[2],  # mean_intensity
#             f[3],  # std_intensity
#             f[4],  # skewness
#             f[5],  # kurtosis
#             int(pred)
#         ])

#     columns = [
#         "image_name",
#         "timestamp",
#         "area",
#         "aspect_ratio",
#         "mean_intensity",
#         "std_intensity",
#         "skewness",
#         "kurtosis",
#         "ml_prediction"
#     ]

#     df = pd.DataFrame(rows, columns=columns)

#     if os.path.exists(LOG_FILE):
#         df.to_csv(LOG_FILE, mode="a", header=False, index=False)
#     else:
#         df.to_csv(LOG_FILE, index=False)

#     return original_color, dl_overlay, final_overlay, decision_text
import os, cv2, numpy as np, pandas as pd
from datetime import datetime
from src.inference.ensemble_segmentation import ensemble_mask
from src.inference.feature_extraction import extract_features
from src.inference.ml_inference import classify_regions

# Use absolute paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOG_DIR = os.path.join(BASE_DIR, "inference_logs")
LOG_FILE = os.path.join(LOG_DIR, "inference_features.csv")

def apply_colored_overlay(image, mask, color, alpha=0.6):
    overlay = image.copy()
    if np.any(mask == 1):
        color_layer = np.zeros_like(image)
        color_layer[:] = color
        overlay[mask == 1] = cv2.addWeighted(
            image[mask == 1], 1-alpha, color_layer[mask == 1], alpha, 0
        )
    return overlay

def run_pipeline(image_path):
    os.makedirs(LOG_DIR, exist_ok=True)

    original_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_color = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)

    dl_mask = ensemble_mask(image_path)
    dl_mask = cv2.resize(dl_mask, (original_gray.shape[1], original_gray.shape[0]), cv2.INTER_NEAREST)
    dl_mask = 1 - dl_mask

    dl_overlay = apply_colored_overlay(original_color, dl_mask, [0,0,255])

    features, contours = extract_features(image_path, dl_mask)

    if len(features) == 0:
        return original_color, dl_overlay, original_color.copy(), "NO OIL DETECTED"

    predictions = classify_regions(features)
    oil_detected = np.any(predictions == 1)

    final_overlay = original_color.copy()
    if oil_detected:
        final_overlay = apply_colored_overlay(original_color, dl_mask, [0,255,0])

    decision = "OIL DETECTED" if oil_detected else "NO OIL DETECTED"

    # Logging (for research only, not metrics)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [[os.path.basename(image_path), timestamp, *f, int(p)] for f, p in zip(features, predictions)]

    cols = ["image","time","area","aspect","mean","std","skew","kurt","pred"]
    pd.DataFrame(rows, columns=cols).to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)

    return original_color, dl_overlay, final_overlay, decision
