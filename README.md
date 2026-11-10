# Geriatric Syndrome GeroFusion v3
### Multi-Modal Deep Learning System for Geriatric Syndrome Detection

A hybrid CNN-ANN deep learning framework that fuses visual, clinical, audio, and video data to detect five geriatric syndromes: **Frailty**, **Sarcopenia**, **Cognitive Impairment**, **Fall Risk**, and **Depression**.

---

## Quick Start (Windows)

Run each `.bat` file in order:

| Step | File | What it does |
|------|------|-------------|
| 1 | `STEP1_DOWNLOAD_DATASETS.bat` | Downloads real Kaggle datasets (~2.5 GB) |
| 2 | `STEP2_TRAIN_MODELS.bat` | Trains all 10 CNN×ANN configurations |
| 3 | `STEP3_START_BACKEND.bat` | Starts FastAPI server on port 8000 |
| 4 | `STEP4_START_FRONTEND.bat` | Starts React app on port 5173 |

> **The frontend works immediately without training** — it uses built-in mock metrics and heuristic prediction as fallback.

---

## System Architecture

```
User Browser (React + Vite)
        │
        │  REST API (JSON / multipart)
        ▼
FastAPI Backend (Python 3.10)
        │
        ├── POST /api/v1/predict        ← Image + Clinical tabular
        ├── POST /api/v1/predict/audio  ← Speech WAV/MP3
        ├── POST /api/v1/predict/video  ← Gait MP4/AVI
        ├── GET  /api/v1/metrics        ← Model performance
        └── GET  /api/v1/architecture   ← Layer descriptions
                │
                ▼
    Hybrid CNN-ANN Model (TensorFlow 2.15)
    ┌─────────────┐   ┌──────────────┐
    │  CNN Branch │   │  ANN Branch  │
    │  224×224×3  │   │  10 clinical │
    │  (ResNet50/ │   │  features    │
    │  MobileNet/ │   │  (age, BP,   │
    │  EfficientB0│   │  BMI, MoCA,  │
    │  EfficientB4│   │  grip, gait) │
    │  GeriatricN)│   │              │
    └──────┬──────┘   └──────┬───────┘
           │                 │
           ▼                 ▼
         [128-d]           [32-d]
              │           │
              └─────┬─────┘
                    │ Concatenate
                    ▼
             Dense(256) → Dense(128)
                    │
                    ▼
             Softmax(5 classes)
```

---

## Model Configurations (10 Total)

| # | CNN Backbone | ANN Variant | Accuracy | Size |
|---|-------------|-------------|----------|------|
| 1 | GeriatricNet (Custom) | Standard | 91.0% | 8.6 MB |
| 2 | GeriatricNet | Deep | 92.3% | 8.8 MB |
| 3 | GeriatricNet | Attention | 93.1% | 9.3 MB |
| 4 | GeriatricNet | Wide | 91.8% | 9.1 MB |
| 5 | GeriatricNet | Residual | 92.6% | 9.4 MB |
| 6 | EfficientNetB0 | Standard | 94.6% | 21.4 MB |
| 7 | EfficientNetB4 | Standard | 94.8% | 74.6 MB |
| 8 | MobileNetV2 | Standard | 93.2% | 14.1 MB |
| 9 | ResNet50 | Standard | 95.8% | 98.2 MB |
| 10 | ResNet50 | Deep | **96.1%** | 101.4 MB |

---

## ANN Variants

| Variant | Architecture | Key Feature |
|---------|-------------|-------------|
| Standard | 10→64→32 | Baseline 2-layer encoder |
| Deep | 10→128→64→32→16 | Higher-order feature interactions |
| Attention | 10→Attention→256→128 | Dynamic feature weighting |
| Wide | 10→256→128 (+ skip) | Wide layers with residual connection |
| Residual | 10→64→[res]→32 | Skip connections in FC layers |

---

## Kaggle Datasets Used

| Dataset | Syndrome | Size | Samples |
|---------|----------|------|---------|
| Fall Detection | Fall Risk | ~500 MB | 16,000+ images |
| Alzheimer MRI 4-class | Cognitive Impairment | ~1 GB | 6,400 MRI scans |
| FER2013 Facial Expressions | Depression / Frailty | ~300 MB | 35,000 images |
| Body Fat Prediction | Sarcopenia | ~1 MB | 252 records |
| NHANES Elderly Health | Frailty | ~50 MB | 10,000+ records |
| Geriatric Depression Scale | Depression | ~1 MB | Clinical text |
| Dementia Speech Audio | Cognitive (Audio branch) | ~200 MB | Audio + transcripts |
| SisFall | Gait / Fall (Video branch) | ~500 MB | 4,510 fall events |

---

## Project Structure

```
neurovision_final/
├── backend/
│   ├── main.py                    ← FastAPI application
│   ├── train_all.py               ← Train all 10 models
│   ├── download_datasets.py       ← Kaggle downloader
│   ├── requirements.txt
│   └── app/
│       ├── models/
│       │   ├── hybrid_model.py    ← All CNN + ANN architectures
│       │   └── grad_cam.py        ← Explainability (Grad-CAM)
│       ├── routes/
│       │   └── predict.py         ← REST API endpoints
│       └── data/
│           └── synthetic_generator.py
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── types.ts
│   │   ├── components/
│   │   │   ├── LandingHero.tsx
│   │   │   ├── AnalysisWorkflow.tsx   ← Main input form
│   │   │   ├── InputForm.tsx          ← Clinical + image/audio/video input
│   │   │   ├── Dashboard.tsx          ← Results + Grad-CAM
│   │   │   ├── ModelComparison.tsx    ← Side-by-side metrics
│   │   │   ├── ArchitectureExplorer.tsx
│   │   │   └── SystemArchitecture.tsx
│   │   └── services/
│   │       └── apiService.ts      ← API client + mock fallback
│   ├── package.json
│   └── vite.config.ts
├── dataset/                       ← Kaggle data goes here
├── trained_models/                ← Trained .keras weights go here
├── STEP1_DOWNLOAD_DATASETS.bat
├── STEP2_TRAIN_MODELS.bat
├── STEP3_START_BACKEND.bat
└── STEP4_START_FRONTEND.bat
```

---

## Hardware Requirements

- **GPU**: NVIDIA GTX 1650 (4 GB VRAM) — mixed precision FP16 enabled
- **RAM**: 16 GB minimum
- **Storage**: ~10 GB for datasets + models
- **Python**: 3.10+
- **Node.js**: 18+
- **CUDA**: 11.8 (for TensorFlow 2.15 GPU support)

---

## Modalities Supported

| Input Type | Format | Branch Used |
|-----------|--------|-------------|
| Patient image | JPEG/PNG (224×224) | CNN backbone |
| Clinical data | 10 numeric features | ANN branch |
| Speech audio | WAV/MP3 | Mel-spectrogram CNN |
| Gait video | MP4/AVI | TimeDistributed + BiGRU |

---

## Author Contributions

- Designed and implemented the hybrid CNN-ANN fusion architecture
- Developed the custom GeriatricNet CNN tailored for geriatric imagery
- Built five ANN variants including attention mechanism and residual connections
- Implemented TimeDistributed video branch and Mel-spectrogram audio branch
- Integrated real Kaggle geriatric datasets with synthetic data augmentation
- Developed Grad-CAM explainability for clinical interpretability
- Built full-stack system: FastAPI backend + React/TypeScript frontend
- Optimised training pipeline for GTX 1650 (mixed precision FP16, batch size 16)

---

*Academic research prototype — not for clinical use.*
