@echo off
chcp 65001 >nul
title GeroFusion v3 -- Step 2: Train All Models
color 0B

echo.
echo ============================================================
echo   Geriatric Syndrome GeroFusion v3
echo   STEP 2: Train All 10 CNN-ANN Hybrid Configurations
echo ============================================================
echo.
echo   Models to train (CNN x ANN):
echo   [1]  GeriatricNet  + Standard ANN  (Custom baseline)
echo   [2]  GeriatricNet  + Deep ANN
echo   [3]  GeriatricNet  + Attention ANN
echo   [4]  GeriatricNet  + Wide ANN
echo   [5]  GeriatricNet  + Residual ANN
echo   [6]  EfficientNetB0 + Standard ANN
echo   [7]  EfficientNetB4 + Standard ANN
echo   [8]  MobileNetV2   + Standard ANN
echo   [9]  ResNet50      + Standard ANN
echo   [10] ResNet50      + Deep ANN      (Best accuracy)
echo.
echo   Hardware: GTX 1650 (4GB VRAM) + 16GB RAM
echo   Mixed precision FP16 enabled for memory efficiency
echo   Batch size: 16  |  Epochs: 35  |  Est. time: 4-6 hours
echo.
echo   Trained models saved to: trained_models\
echo   Metrics saved to: trained_models\all_metrics.json
echo.
echo   Press any key to start training...
pause >nul

cd /d "%~dp0"

echo.
echo Installing Python dependencies...
pip install -r backend\requirements.txt --quiet

echo.
echo Starting training pipeline...
python backend\train_all.py

echo.
if %errorlevel% equ 0 (
    color 0A
    echo   ============================================================
    echo   TRAINING COMPLETE!
    echo   Models saved to: trained_models\
    echo   Now run STEP3_START_BACKEND.bat
    echo   ============================================================
) else (
    color 0C
    echo   Training encountered errors. Check output above.
    echo   The app will use precomputed metrics as fallback.
)
echo.
pause
