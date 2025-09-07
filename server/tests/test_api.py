import io
import sys
import os

# Ensure server root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from PIL import Image
from app.service.api import app, PREDICTORS, cfg

client = TestClient(app)

class DummyPredictor:
    def predict_with_cam(self, img, out_path):
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        img.save(out_path, format="PNG")
        return "Normal", 0.98

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data

def test_models():
    res = client.get("/models")
    assert res.status_code == 200
    models_list = res.json()
    assert len(models_list) > 0
    assert models_list[0]["id"] == "default"

def test_predict_single():
    PREDICTORS.clear()
    meta = cfg.model_registry["default"]
    PREDICTORS[(meta["arch"], meta["path"])] = DummyPredictor()

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(100, 100, 100)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    res = client.post(
        "/predict",
        files={"file": ("test_xray.png", png_bytes, "image/png")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["label"] == "Normal"
    assert data["confidence"] == 0.98
    assert data["gradcam_url"].startswith("/static/")

def test_predict_batch():
    PREDICTORS.clear()
    meta = cfg.model_registry["default"]
    PREDICTORS[(meta["arch"], meta["path"])] = DummyPredictor()

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(50, 50, 50)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    files = [
        ("files", ("image1.png", png_bytes, "image/png")),
        ("files", ("image2.png", png_bytes, "image/png")),
    ]
    res = client.post("/predict-batch", files=files)
    assert res.status_code == 200
    batch_res = res.json()
    assert len(batch_res) == 2
    assert batch_res[0]["label"] == "Normal"
    assert batch_res[1]["label"] == "Normal"

def test_unsupported_file_format():
    res = client.post(
        "/predict",
        files={"file": ("malicious.exe", b"malicious binary payload", "application/octet-stream")}
    )
    assert res.status_code == 415
