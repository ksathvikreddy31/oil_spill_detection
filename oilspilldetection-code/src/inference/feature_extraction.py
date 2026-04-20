import cv2
import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(image_path, mask):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        region = img[y:y+h, x:x+w]

        feature_vector = [
            area,
            w / (h + 1e-6),
            np.mean(region),
            np.std(region),
            skew(region.flatten()),
            kurtosis(region.flatten())
        ]

        features.append(feature_vector)

    return np.array(features), contours
