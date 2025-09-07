import os
import sys
import time
import atexit
import signal
import platform
import subprocess
import requests
import streamlit as st
from PIL import Image

# ===============================
# Backend auto-start (helper)
# ===============================
BACKEND_PROC = None

def _kill_backend():
    """Ensure the spawned backend process is terminated when Streamlit stops."""
    global BACKEND_PROC
    if BACKEND_PROC is not None and BACKEND_PROC.poll() is None:
        try:
            if platform.system() == "Windows":
                BACKEND_PROC.terminate()
            else:
                os.killpg(os.getpgid(BACKEND_PROC.pid), signal.SIGTERM)
        except Exception:
            pass
        BACKEND_PROC = None

@st.cache_resource(show_spinner=False)
def ensure_backend(api_base: str = "http://localhost:8000", autostart: bool = True, boot_timeout: float = 20.0):
    """
    Ensure the FastAPI backend is reachable at api_base.
    If not reachable and autostart=True, spawn a uvicorn server in the background
    using DEVNULL to prevent OS pipe deadlocks.
    """
    global BACKEND_PROC

    health_url = f"{api_base}/health"

    def _healthy(timeout=2):
        try:
            r = requests.get(health_url, timeout=timeout)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    if _healthy():
        return True

    if not autostart:
        return False

    st.sidebar.info("Starting FastAPI backend…")
    # Launch uvicorn pointing to server app directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    server_dir = os.path.join(root_dir, "server")

    uvicorn_cmd = [
        sys.executable, "-m", "uvicorn",
        "app.service.api:app",
        "--app-dir", server_dir,
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    popen_kwargs = {}
    if platform.system() != "Windows":
        popen_kwargs.update(dict(preexec_fn=os.setsid))

    try:
        # Use DEVNULL to prevent buffer deadlocks on unconsumed stdout/stderr
        BACKEND_PROC = subprocess.Popen(
            uvicorn_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **popen_kwargs
        )
        atexit.register(_kill_backend)
    except Exception as e:
        st.sidebar.error(f"Failed to launch backend: {e}")
        return False

    # Wait for health check to succeed
    start = time.time()
    while time.time() - start < boot_timeout:
        if _healthy(timeout=1):
            st.sidebar.success("Backend started ✅")
            return True
        time.sleep(0.5)

    st.sidebar.error("Backend did not become healthy within the timeout window.")
    return False


# ===============================
# UI Configuration & Styling
# ===============================
st.set_page_config(
    page_title="Pneumonia CXR Detection",
    page_icon="🫁",
    layout="wide"
)

st.sidebar.title("🫁 Pneumonia Detection")
st.sidebar.caption("Research & Education Pipeline (DenseNet-121)")
st.sidebar.warning("⚠️ **Disclaimer**: For research & educational purposes only. Not for clinical diagnosis.")

ENV_API = os.environ.get("API_URL", "").strip()
API = ENV_API if ENV_API else "http://localhost:8000"

if not ENV_API:
    backend_ready = ensure_backend(API, autostart=True, boot_timeout=25.0)
    if not backend_ready:
        st.sidebar.warning("Local backend is not running yet. You can also run 'npm run server' from root.")
else:
    try:
        r = requests.get(f"{API}/health", timeout=3)
        if r.status_code == 200:
            st.sidebar.success("Connected to API ✅")
        else:
            st.sidebar.warning(f"Backend status: HTTP {r.status_code}")
    except Exception as e:
        st.sidebar.error(f"Could not reach {API}: {e}")

# Safe model metadata fetch
try:
    models_res = requests.get(f"{API}/models", timeout=4)
    if models_res.status_code == 200 and len(models_res.json()) > 0:
        meta = models_res.json()[0]
        st.sidebar.markdown(f"**Model**: `{meta['name']}`")
        st.sidebar.markdown(f"**Input Size**: `{meta['input_size']}x{meta['input_size']}`")
        st.sidebar.markdown(f"**Parameters**: `{meta['params']}`")
        if "roc_auc" in meta:
            st.sidebar.markdown(f"**Reported ROC-AUC**: `{meta['roc_auc']:.2f}`")
except Exception:
    st.sidebar.info("Model info will appear when backend is connected.")


# ===============================
# Session State Initialization
# ===============================
if "single_result" not in st.session_state:
    st.session_state.single_result = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


# ===============================
# Main UI Tabs
# ===============================
tab_inference, tab_metrics, tab_about = st.tabs(["🔬 Inference & Explainability", "📊 Metrics Dashboard", "ℹ️ System Info"])

with tab_inference:
    st.header("Chest X-Ray Pneumonia Analysis")
    st.markdown("Upload frontal chest X-rays (`.png`, `.jpg`, `.jpeg`, or `.dcm`) to generate classifications and Grad-CAM activation heatmaps.")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploader = st.file_uploader(
            "Select X-Ray Image(s)",
            type=["png", "jpg", "jpeg", "dcm"],
            accept_multiple_files=True,
            help="Supports standard image files and DICOM radiographs."
        )

        enable_dicom = False
        if uploader and any(f.name.lower().endswith(".dcm") for f in uploader):
            enable_dicom = True
            st.info("DICOM file(s) detected. Windowing and photometric correction will be applied automatically.")

        predict_btn = st.button("🚀 Run Prediction", type="primary", disabled=(not uploader))

    def call_api_single(file):
        try:
            files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
            params = {"enable_dicom": file.name.lower().endswith(".dcm")}
            res = requests.post(f"{API}/predict", files=files, params=params, timeout=30)
            if res.status_code == 200:
                return res.json(), None
            return None, f"API Error ({res.status_code}): {res.text}"
        except Exception as e:
            return None, f"Request failed: {e}"

    def call_api_batch(files_list):
        try:
            files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files_list]
            params = {"enable_dicom": any(f.name.lower().endswith(".dcm") for f in files_list)}
            res = requests.post(f"{API}/predict-batch", files=files, params=params, timeout=60)
            if res.status_code == 200:
                return res.json(), None
            return None, f"API Error ({res.status_code}): {res.text}"
        except Exception as e:
            return None, f"Request failed: {e}"

    if predict_btn and uploader:
        with st.spinner("Analyzing radiograph(s) with DenseNet-121 and generating Grad-CAM..."):
            if len(uploader) == 1:
                result, err = call_api_single(uploader[0])
                if err:
                    st.error(err)
                    st.session_state.single_result = None
                else:
                    st.session_state.single_result = (uploader[0], result)
                    st.session_state.batch_results = None
            else:
                results, err = call_api_batch(uploader)
                if err:
                    st.error(err)
                    st.session_state.batch_results = None
                else:
                    st.session_state.batch_results = results
                    st.session_state.single_result = None

    with col2:
        # Single image display
        if st.session_state.single_result:
            uploaded_file, res = st.session_state.single_result
            label = res["label"]
            conf = res["confidence"]
            gradcam_url = f"{API}{res['gradcam_url']}"

            if label.lower() == "pneumonia":
                st.error(f"### Result: **{label}** ({conf:.1%} confidence)")
            else:
                st.success(f"### Result: **{label}** ({conf:.1%} confidence)")

            st.progress(min(max(conf, 0.0), 1.0))
            st.image(gradcam_url, caption=f"Grad-CAM Heatmap Overlay ({uploaded_file.name})", use_container_width=True)

        # Batch images display
        elif st.session_state.batch_results:
            results = st.session_state.batch_results
            st.subheader(f"Batch Results ({len(results)} items)")
            for item in results:
                if item.get("label") == "Error":
                    st.error(f"**{item['filename']}** → Error: {item.get('gradcam_url')}")
                else:
                    color = "red" if item['label'].lower() == "pneumonia" else "green"
                    st.markdown(f"**{item['filename']}**: :{color}[**{item['label']}**] ({item['confidence']:.1%})")
                    st.image(f"{API}{item['gradcam_url']}", width=350)
        else:
            st.info("Upload an image on the left and click **Run Prediction** to view diagnostic findings.")

with tab_metrics:
    st.header("Model Evaluation & Clinical Diagnostics")
    st.markdown("Visualizations generated from test set evaluation (`outputs/` or `server/outputs/`).")

    output_candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server", "outputs")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs")),
        "outputs"
    ]
    out_dir = next((d for d in output_candidates if os.path.exists(d)), "server/outputs")

    metric_cols = st.columns(2)
    metric_files = [
        ("roc.png", "ROC Curve & AUC"),
        ("pr.png", "Precision-Recall Curve"),
        ("loss.png", "Training & Validation Loss"),
        ("acc.png", "Training & Validation Accuracy"),
        ("calibration.png", "Reliability & Calibration")
    ]

    for idx, (filename, title) in enumerate(metric_files):
        img_path = os.path.join(out_dir, filename)
        col = metric_cols[idx % 2]
        with col:
            if os.path.exists(img_path):
                st.image(img_path, caption=title, use_container_width=True)
            else:
                st.caption(f"📊 *{title}* will be displayed once model evaluation (`npm run evaluate`) is completed.")

with tab_about:
    st.header("About PneumoniaCXR")
    st.markdown("""
    **Architecture & Technology Stack**:
    - **Backbone**: PyTorch DenseNet-121 with ImageNet-1K pretrained weights
    - **Optimization**: AdamW, Cosine Annealing, Class-weighted BCE / Focal Loss
    - **Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping) on `denseblock4`
    - **Backend**: FastAPI RESTful microservice with asynchronous request processing
    - **Frontend**: Streamlit interactive dashboard with DICOM handling and batch processing
    """)
