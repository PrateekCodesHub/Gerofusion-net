@echo off
chcp 65001 >nul
title GeroFusion v3 -- Step 4: Start React Frontend
color 0D

echo.
echo ============================================================
echo   Geriatric Syndrome GeroFusion v3
echo   STEP 4: Start React Frontend (Vite Dev Server)
echo ============================================================
echo.
echo   Frontend will open at: http://localhost:5173
echo.
echo   Features:
echo     - Landing page with system overview
echo     - Multi-modal input: image + audio + video + clinical data
echo     - Real-time model selection (5 CNNs x 5 ANNs)
echo     - Dashboard with Grad-CAM visualization
echo     - Model comparison charts
echo     - Architecture explorer
echo     - System architecture diagram
echo.
echo   NOTE: Works with mock data even without backend running.
echo   For live inference, start STEP3 backend first.
echo.

cd /d "%~dp0\frontend"

echo Installing Node.js dependencies...
call npm install

echo.
echo Starting frontend...
echo.
call npm run dev

pause
