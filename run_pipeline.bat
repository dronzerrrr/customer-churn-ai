@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo Virtual environment not found. Create it first:
  echo   python -m venv .venv
  echo   .venv\Scripts\activate
  echo   pip install -r requirements.txt
  exit /b 1
)
call ".venv\Scripts\activate.bat"

for %%M in (src.data.download_data src.data.validate src.data.database src.data.split src.models.train src.evaluation.explain) do (
  echo.
  echo === Running %%M ===
  python -m %%M
  if errorlevel 1 (
    echo FAILED at %%M
    exit /b 1
  )
)

if /i not "%~1"=="notests" (
  echo.
  echo === Running tests ===
  python -m pytest -q
  if errorlevel 1 (
    echo Tests failed.
    exit /b 1
  )
)

echo.
echo Pipeline complete. Model saved in the artifacts folder.
endlocal