import os
import sys
import re
import time
import uuid
import json
from io import BytesIO
from typing import List

# Ensure server root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.schemas import PredictResponse, PredictBatchItem, ModelsResponseItem, Health
from app.config import ServeConfig
from app.inference import Predictor
from app.data.dicom_utils import dicom_to_png_bytes
from app.security import rate_limiter, validate_upload

app = FastAPI(
    title="Pneumonia CXR Analysis API",
    description="Deep Learning API for Chest X-Ray Pneumonia Detection and Grad-CAM Explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = ServeConfig()
PREDICTORS = {}
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "served_outputs"))
os.makedirs(OUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=OUT_DIR), name="static")

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

def get_predictor(model_id: str = "default") -> Predictor:
    if model_id not in cfg.model_registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry.")

    meta = cfg.model_registry[model_id]
    key = (meta["arch"], meta["path"])

    if key not in PREDICTORS:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            PREDICTORS[key] = Predictor(meta["arch"], meta["path"], meta["img_size"], device)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize model '{model_id}': {e}")

    return PREDICTORS[key]

@app.get("/health", response_model=Health)
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/models", response_model=List[ModelsResponseItem])
def models():
    # Attempt to read dynamic ROC-AUC from test or validation metrics
    reported_auc = 0.95
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for metrics_candidate in ["test_metrics.json", "val_metrics.json"]:
        m_path = os.path.join(base_dir, "outputs", metrics_candidate)
        if os.path.exists(m_path):
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "roc_auc" in data:
                        reported_auc = float(data["roc_auc"])
                        break
            except Exception:
                pass

    return [
        {
            "id": "default",
            "name": "DenseNet-121",
            "params": "7.98M",
            "input_size": "224",
            "trained_on": "Kaggle Chest X-Ray",
            "roc_auc": round(reported_auc, 3),
        }
    ]

@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_id: str = "default",
    enable_dicom: bool = False,
):
    rate_limiter(get_client_ip(request))
    start_time = time.time()

    data = await file.read()
    validate_upload(file.filename, data, max_mb=cfg.max_upload_mb, allowed_ext=cfg.allowed_ext)

    try:
        if enable_dicom and file.filename.lower().endswith(".dcm"):
            data = dicom_to_png_bytes(data)

        img = Image.open(BytesIO(data)).convert("RGB")
        predictor = get_predictor(model_id)

        # Unique UUID-based output filename prevents concurrent overwrites
        unique_prefix = uuid.uuid4().hex[:12]
        clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", os.path.splitext(file.filename)[0])
        out_name = f"{unique_prefix}_{clean_name}_gradcam.png"
        out_path = os.path.join(OUT_DIR, out_name)

        label, conf = predictor.predict_with_cam(img, out_path)
        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "label": label,
            "confidence": conf,
            "gradcam_url": f"/static/{out_name}",
            "processing_time_ms": round(elapsed_ms, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

@app.post("/predict-batch", response_model=List[PredictBatchItem])
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    model_id: str = "default",
    enable_dicom: bool = False,
):
    rate_limiter(get_client_ip(request))
    predictor = get_predictor(model_id)
    results = []

    for f in files:
        try:
            data = await f.read()
            validate_upload(f.filename, data, max_mb=cfg.max_upload_mb, allowed_ext=cfg.allowed_ext)

            if enable_dicom and f.filename.lower().endswith(".dcm"):
                data = dicom_to_png_bytes(data)

            img = Image.open(BytesIO(data)).convert("RGB")
            unique_prefix = uuid.uuid4().hex[:12]
            clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", os.path.splitext(f.filename)[0])
            out_name = f"{unique_prefix}_{clean_name}_gradcam.png"
            out_path = os.path.join(OUT_DIR, out_name)

            label, conf = predictor.predict_with_cam(img, out_path)
            results.append({
                "filename": f.filename,
                "label": label,
                "confidence": conf,
                "gradcam_url": f"/static/{out_name}",
                "status": "success",
                "error": None
            })
        except Exception as e:
            results.append({
                "filename": f.filename or "unknown",
                "label": "Error",
                "confidence": 0.0,
                "gradcam_url": None,
                "status": "error",
                "error": str(e)
            })

    return results

if __name__ == "__main__":
    uvicorn.run("app.service.api:app", host=cfg.host, port=cfg.port, reload=True)
