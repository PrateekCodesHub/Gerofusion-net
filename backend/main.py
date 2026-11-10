"""
main.py — FastAPI Application Entry Point
==========================================
Geriatric Syndrome GeroFusion v3

Usage:
    # From project root (neurovision_final/):
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router as predict_router

app = FastAPI(
    title="GeroFusion API v3",
    description=(
        "Multi-modal deep learning system for geriatric syndrome detection. "
        "Supports image, tabular, audio, and video inputs across 5 CNN backbones "
        "and 5 ANN variants."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/")
async def root():
    return {
        "system": "Geriatric Syndrome GeroFusion v3",
        "version": "3.0.0",
        "modalities": ["image", "tabular_clinical", "audio", "video"],
        "endpoints": {
            "docs":          "/docs",
            "health":        "/api/v1/health",
            "predict":       "POST /api/v1/predict",
            "predict_audio": "POST /api/v1/predict/audio",
            "predict_video": "POST /api/v1/predict/video",
            "metrics":       "/api/v1/metrics",
            "comparison":    "/api/v1/metrics/comparison",
            "architecture":  "/api/v1/architecture/{model_type}",
        },
    }
