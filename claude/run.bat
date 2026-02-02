@echo off
echo ============================================================
echo Feature-Based Bottle Tracking System
echo ============================================================
echo.

cd /d "%~dp0"

echo Activating virtual environment...
call "C:\Users\nadav\AppData\Local\Temp\skylyx_drone_venv\Scripts\activate.bat"

echo Starting tracker...
echo.
python run_tracker.py

pause
