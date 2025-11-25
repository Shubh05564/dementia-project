@echo off
echo Starting Dementia Prevention API Server...
cd /d "C:\Users\H P\dementia_nutrition_ml"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
pause
