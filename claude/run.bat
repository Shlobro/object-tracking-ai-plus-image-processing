@echo off
echo ============================================================
echo Feature-Based Bottle Tracking System
echo ============================================================
echo.

cd /d "%~dp0"

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting tracker...
echo.
python run_tracker.py

pause
