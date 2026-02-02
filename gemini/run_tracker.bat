@echo off
set VENV_DIR=venv

if not exist %VENV_DIR% (
    echo [INFO] Virtual environment not found. Creating one now...
    python -m venv %VENV_DIR%
    
    echo [INFO] Activating virtual environment...
    call %VENV_DIR%\Scripts\activate
    
    echo [INFO] Installing required libraries. This may take a few minutes...
    pip install ultralytics opencv-python numpy
) else (
    echo [INFO] Virtual environment found. Activating...
    call %VENV_DIR%\Scripts\activate
)

echo [INFO] Starting tracker system...
python tracker.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The program exited with an error code.
)

pause