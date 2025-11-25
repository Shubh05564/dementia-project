@echo off
echo ========================================
echo Dementia Prevention Advisor - Deployment
echo ========================================
echo.
echo Starting services...
echo.

cd /d "C:\Users\H P\dementia_nutrition_ml"

echo [1/2] Starting API Server on http://localhost:8000
start "Dementia API" cmd /k "python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000"

echo Waiting for API to initialize...
timeout /t 5 /nobreak > nul

echo [2/2] Starting Web App on http://localhost:8501
start "Dementia Web App" cmd /k "streamlit run src/api/app.py --server.port 8501"

echo.
echo ========================================
echo Deployment Complete!
echo ========================================
echo.
echo API Server:  http://localhost:8000
echo Web App:     http://localhost:8501
echo API Docs:    http://localhost:8000/docs
echo.
echo The applications will open in new windows.
echo Close those windows to stop the services.
echo.
pause
