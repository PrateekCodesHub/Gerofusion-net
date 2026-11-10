@echo off
chcp 65001 >nul
title GeroFusion v3 -- Step 3: Start Python Backend
color 0E

echo.
echo ============================================================
echo   Geriatric Syndrome GeroFusion v3
echo   STEP 3: Start FastAPI Backend Server
echo ============================================================
echo.
echo   API will be available at:
echo     http://localhost:8000
echo     http://localhost:8000/docs  (Swagger UI)
echo     http://localhost:8000/redoc (ReDoc UI)
echo.
echo   Endpoints:
echo     POST /api/v1/predict        (main inference)
echo     POST /api/v1/predict/audio  (speech analysis)
echo     POST /api/v1/predict/video  (gait analysis)
echo     GET  /api/v1/metrics        (model performance)
echo     GET  /api/v1/architecture/{model}
echo.
echo   Keep this window open while using the app.
echo   Press Ctrl+C to stop the server.
echo.
cd /d "%~dp0"

pip install -r backend\requirements.txt --quiet

echo Starting server...
echo.
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

pause
