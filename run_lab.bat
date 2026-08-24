@echo off
setlocal
cd /d "%~dp0"
set "STUDY=%SKOOL_STUDY%"
if "%STUDY%"=="" set "STUDY=capacity"

echo [1/3] Preparing the Python environment...
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul
  if errorlevel 1 (
    python -m venv .venv
  ) else (
    py -3 -m venv .venv
  )
)
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" -u setup_dependencies.py
if errorlevel 1 exit /b %errorlevel%
echo [2/3] Running the CPU scaling experiment...

if "%STUDY%"=="capacity" ".venv\Scripts\python.exe" -u scaling_lab.py %*
if "%STUDY%"=="budget" ".venv\Scripts\python.exe" -u skool_studies.py budget %*
if "%STUDY%"=="data" ".venv\Scripts\python.exe" -u skool_studies.py data %*
if not "%STUDY%"=="capacity" if not "%STUDY%"=="budget" if not "%STUDY%"=="data" (
  echo Unknown study: %STUDY% 1>&2
  exit /b 2
)
if errorlevel 1 exit /b %errorlevel%
echo [3/3] Complete.
