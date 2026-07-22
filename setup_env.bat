@echo off
setlocal
cd /d %~dp0

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found in PATH.
  echo Install Python 3.11 or newer, then run this script again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
  )
)

echo Installing project, test, and build dependencies...
.\.venv\Scripts\python.exe -m pip install -e .[dev,build]
if errorlevel 1 (
  echo Failed to install dependencies.
  pause
  exit /b 1
)

echo Environment is ready.
echo Python: .venv\Scripts\python.exe
