import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 256

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def predict_mask(model_path, image_path):
    model = load_model(model_path, compile=False)
    img = preprocess_image(image_path)
    pred = model.predict(img)[0, :, :, 0]
    return (pred > 0.5).astype("uint8")
