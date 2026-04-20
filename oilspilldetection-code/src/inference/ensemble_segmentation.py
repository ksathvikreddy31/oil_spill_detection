import os
import numpy as np
from src.inference.dl_inference import predict_mask

# Use absolute paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def ensemble_mask(image_path):
    m1 = predict_mask(os.path.join(MODELS_DIR, "unet_final.h5"), image_path)
    m2 = predict_mask(os.path.join(MODELS_DIR, "linknet_final.h5"), image_path)
    m3 = predict_mask(os.path.join(MODELS_DIR, "dilated_unet_final.h5"), image_path)

    stacked = np.stack([m1, m2, m3], axis=0)
    final_mask = (np.sum(stacked, axis=0) >= 2).astype("uint8")

    return final_mask
