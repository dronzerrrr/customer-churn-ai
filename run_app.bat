@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo Virtual environment not found. Create it first:
  echo   python -m venv .venv
  echo   .venv\Scripts\activate
  echo   pip install -r requirements.txt
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"

if not exist "artifacts\model.joblib" (
  echo No trained model found. Running the training pipeline first...
  call run_pipeline.bat notests
  if errorlevel 1 (
    echo The pipeline failed. See the messages above.
    pause
    exit /b 1
  )
)

echo Starting the dashboard at http://localhost:8501
python -m streamlit run app\streamlit_app.py
endlocal