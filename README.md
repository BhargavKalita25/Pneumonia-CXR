# Pneumonia Detection from Chest X-Rays (DenseNet-121, PyTorch)

**For research & education only -- not for clinical use.**

This project provides an end-to-end deep learning pipeline for pneumonia detection from chest X-ray (CXR) radiographs. It features:
- **PyTorch Model Pipeline**: DenseNet-121 (plus ResNet & EfficientNet support) with ImageNet pretrained transfer learning, Automatic Mixed Precision (AMP), class-weighted BCE / Focal loss, and cosine annealing scheduling.
- **Explainable AI (XAI)**: Grad-CAM activation heatmaps showing region-level visual explanations of model focus.
- **FastAPI Backend (`server/`)**: Asynchronous, rate-limited REST API providing single and batch inference endpoints.
- **Streamlit Frontend (`client/`)**: Modern interactive diagnostic dashboard with drag-and-drop upload (PNG, JPEG, and DICOM with windowing and MONOCHROME1 correction).
- **Workspace Orchestrator**: Root-level `package.json` enabling single-command startup, training, testing, and evaluation without opening multiple terminals or installing local node dependencies.

---

## Project Structure

```
PneumoniaCXR/
├── client/                     # Frontend Application
│   ├── .streamlit/             # Streamlit configuration
│   │   └── config.toml         # Headless server & port configuration
│   ├── streamlit_app.py        # Streamlit interface with Grad-CAM visualization
│   └── requirements.txt        # Frontend dependencies
│
├── server/                     # Backend & Deep Learning Pipeline
│   ├── requirements.txt        # Backend dependencies (PyTorch, FastAPI, etc.)
│   ├── launcher.py             # Multi-service terminal orchestrator & dashboard
│   ├── setup_env.py            # Automated virtual environment provisioner
│   ├── run.py                  # Transparent command runner routing via .venv
│   ├── app/
│   │   ├── config.py           # Training and server configurations
│   │   ├── schemas.py          # Pydantic request/response schemas
│   │   ├── security.py         # Rate limiting & file/MIME validation
│   │   ├── train.py            # Model training pipeline
│   │   ├── evaluate.py         # Test set evaluation & diagnostic plots
│   │   ├── inference.py        # Thread-safe Predictor with Grad-CAM
│   │   ├── data/
│   │   │   ├── dataset.py      # CXRFolder dataset loader & patient ID extractor
│   │   │   ├── dicom_utils.py  # DICOM VOI windowing & MONOCHROME1 handling
│   │   │   └── splits.py       # Patient leakage auditing
│   │   ├── models/
│   │   │   ├── builder.py      # Model backbone factory (DenseNet, ResNet, EfficientNet)
│   │   │   ├── losses.py       # Weighted BCE & Focal Loss
│   │   │   └── gradcam.py      # Grad-CAM implementation & heatmap overlay
│   │   ├── profiles/           # Hardware profiles (cpu.yaml, gpu_t4.yaml, gpu_v100.yaml)
│   │   ├── service/
│   │   │   └── api.py          # FastAPI application
│   │   └── utils/
│   │       ├── metrics.py      # Clinical metrics (ROC-AUC, PR-AUC, Brier score)
│   │       ├── plots.py        # ROC, PR, calibration, and training curves
│   │       └── seed.py         # Full reproducibility seeding
│   ├── outputs/                # Checkpoints & evaluation curves (.gitkeep)
│   ├── served_outputs/         # Generated Grad-CAM static files (.gitkeep)
│   └── tests/                  # Automated test suite (API, Preprocessing, Losses, DICOM, Metrics)
│
├── data/                       # Dataset directory (.gitkeep)
├── package.json                # Root workspace orchestrator
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

---

## Quick Start

### 1. Environment Setup

#### Option A: Automated Virtual Environment via NPM (Recommended)
From the project root:
```bash
npm run setup
```
This automatically:
1. Detects or creates a dedicated Python virtual environment in `.venv/`.
2. Installs all backend dependencies from `server/requirements.txt`.
3. Installs all frontend dependencies from `client/requirements.txt`.
4. Configures all subsequent npm commands (`npm run start`, `npm run test`, `npm run train`, etc.) to automatically execute inside `.venv`.

#### Option B: Manual Virtual Environment
If you prefer configuring your virtual environment manually:
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r server/requirements.txt
pip install -r client/requirements.txt
```

---

## Running the Application

### Launch Both Backend & Frontend Together (Single Command)
```bash
npm run start
```

Both services will initialize in the background, and once ready, an aligned dashboard is displayed in the terminal:

```text
╔══════════════════════════════════════════════════════════════════════╗
║         PneumoniaCXR Orchestrator  --  Initializing Services         ║
╚══════════════════════════════════════════════════════════════════════╝

  [1/2] Starting FastAPI backend on port 8000...  [OK] [1/2] FastAPI backend running on port 8000        
  [2/2] Starting Streamlit frontend on port 8501...  [OK] [2/2] Streamlit frontend running on port 8501      

╭──────────────────────────────────────────────────────────────────────╮
│                     PNEUMONIACXR SYSTEM DASHBOARD                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Web Application (Frontend UI):                                     │
│      - Local URL:    http://localhost:8501                           │
│      - Network URL:  http://172.20.10.4:8501                         │
│                                                                      │
│   REST API (Backend Service):                                        │
│      - Local URL:    http://localhost:8000                           │
│      - Network URL:  http://172.20.10.4:8000                         │
│      - API Docs:     http://localhost:8000/docs                      │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│   Status: Active & Listening  |  Press Ctrl+C to Stop                │
╰──────────────────────────────────────────────────────────────────────╯
```

- **Frontend UI**: Open `http://localhost:8501` in your browser.
- **Backend API Docs**: Open `http://localhost:8000/docs` for interactive Swagger documentation.
- **Stop Services**: Press `Ctrl + C` in the terminal to cleanly terminate both processes.

### Running Services Separately
- **Start FastAPI Server only**:
  ```bash
  npm run server
  # Or directly with Python:
  python -m uvicorn app.service.api:app --app-dir server --host 0.0.0.0 --port 8000 --reload
  ```
- **Start Streamlit Client only**:
  ```bash
  npm run client
  # Or directly with Python:
  python -m streamlit run client/streamlit_app.py --server.port 8501 --server.headless true
  ```

---

## Training the Model

1. Download the [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset and extract it to `data/kaggle_chest_xray/`:
   ```text
   data/kaggle_chest_xray/
     train/
       NORMAL/
       PNEUMONIA/
     val/
       NORMAL/
       PNEUMONIA/
     test/
       NORMAL/
       PNEUMONIA/
   ```

2. Run training:
   ```bash
   npm run train
   ```
   - Hardware profiles (`cpu.yaml`, `gpu_t4.yaml`, `gpu_v100.yaml`) adapt batch sizes and AMP settings.
   - Checkpoints and training curves (`loss.png`, `acc.png`) are saved in `server/outputs/`.

---

## Evaluation & Diagnostics

Evaluate the trained checkpoint on the held-out test set:
```bash
npm run evaluate
```

Outputs saved in `server/outputs/`:
- `test_metrics.json` (ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1, Sensitivity @ 90% Specificity, Brier score)
- `roc.png` (Receiver Operating Characteristic)
- `pr.png` (Precision-Recall Curve)
- `calibration.png` (Reliability diagram)

---

## Running Automated Tests

Run the full pytest suite:
```bash
npm run test
```

All 16 test cases covering API endpoints, DICOM parsing, MultiValue windowing, MONOCHROME1 inversion, Focal Loss calculations, patient ID extraction, and clinical metrics serialization will execute.

---

## Clinical Notes & Disclaimer
- **Not for Clinical Diagnosis**: This tool is developed strictly for research, testing, and educational purposes.
- **Pediatric Bias**: The underlying Kaggle dataset consists predominantly of pediatric radiographs (children aged 1 to 5). Predictions may not generalize to adult chest radiographs.
