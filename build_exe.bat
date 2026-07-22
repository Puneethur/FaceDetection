@echo off
setlocal
cd /d %~dp0

call setup_env.bat
if errorlevel 1 (
  echo Environment setup failed.
  pause
  exit /b 1
)

echo Building FaceDetectionStudio.exe...
.\.venv\Scripts\python.exe -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onefile ^
  --name FaceDetectionStudio ^
  --collect-data cv2 ^
  run_face_detection_gui.py

if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

echo Build complete.
echo Output: dist\FaceDetectionStudio.exe
pause
