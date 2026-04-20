import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import cv2
import base64
from src.inference.pipeline import run_pipeline


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def encode_image(img):
    _, buffer = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode()

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()

    temp_file = os.path.join(os.path.dirname(__file__), "temp.png")
    with open(temp_file, "wb") as f:
        f.write(contents)

    original, dl_pred, final_pred, decision = run_pipeline(temp_file)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    metrics_path = os.path.join(base_dir, "offline_model_metrics.csv")
    cm_path = os.path.join(base_dir, "offline_confusion_matrix.csv")

    metrics = pd.read_csv(metrics_path).to_dict(orient="records")
    cm = pd.read_csv(cm_path).values.tolist()

    return {
        "original": encode_image(original),
        "dl_pred": encode_image(dl_pred),
        "final_pred": encode_image(final_pred),
        "decision": decision,
        "metrics": metrics,
        "cm": cm
    }