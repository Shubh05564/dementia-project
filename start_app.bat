@echo off
echo Starting Dementia Prevention Web App...
cd /d "C:\Users\H P\dementia_nutrition_ml"
streamlit run src/api/app.py --server.port 8501 --server.address localhost
pause
