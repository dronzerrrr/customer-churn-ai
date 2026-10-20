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
  echo No trained model found. Run run_pipeline.bat first.
  pause
  exit /b 1
)

echo Starting the API. Interactive docs: http://127.0.0.1:8000/docs
python -m uvicorn api.main:app --port 8000
endlocal