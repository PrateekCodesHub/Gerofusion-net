@echo off
chcp 65001 >nul
title GeroFusion v3 -- Step 1: Download Kaggle Datasets
color 0A

echo.
echo ============================================================
echo   Geriatric Syndrome GeroFusion v3
echo   STEP 1: Download Real Kaggle Datasets
echo ============================================================
echo.
echo   Datasets to download:
echo   [1] Fall Detection Images      ~500MB  (Fall Risk)
echo   [2] Alzheimer MRI 4-class      ~1GB    (Cognitive Impairment)
echo   [3] FER2013 Facial Expressions ~300MB  (Depression/Frailty)
echo   [4] Body Fat Prediction CSV    ~1MB    (Sarcopenia)
echo   [5] NHANES Elderly Health CSV  ~50MB   (Frailty tabular)
echo   [6] Geriatric Depression Scale ~1MB    (Depression tabular)
echo   [7] Dementia Speech Audio      ~200MB  (Audio branch)
echo   [8] SisFall Accelerometer      ~500MB  (Gait/Fall video)
echo.
echo   Kaggle Account: feat1phoenix
echo   Total estimated: ~2.5 GB
echo.
echo   Press any key to begin downloading...
pause >nul

cd /d "%~dp0"

echo.
echo [1/2] Installing Kaggle API if not present...
pip install kaggle --quiet

echo.
echo [2/2] Running dataset downloader...
python backend\download_datasets.py

echo.
if %errorlevel% equ 0 (
    color 0A
    echo   ============================================================
    echo   SUCCESS! Datasets downloaded to dataset\ folder
    echo   Now run STEP2_TRAIN_MODELS.bat
    echo   ============================================================
) else (
    color 0C
    echo   Some datasets may have failed. Check output above.
    echo   You can still run STEP2 - synthetic data will fill gaps.
)
echo.
pause
