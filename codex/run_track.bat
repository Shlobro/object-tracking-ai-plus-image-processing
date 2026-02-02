@echo off
setlocal

set VENV_DIR=.venv
set PY_EXE=%VENV_DIR%\Scripts\python.exe

if not exist %PY_EXE% (
  echo Creating venv in %VENV_DIR% ...
  python -m venv %VENV_DIR%
  if errorlevel 1 (
    echo Failed to create venv.
    exit /b 1
  )
)

call %VENV_DIR%\Scripts\activate.bat

echo Checking dependencies...
%PY_EXE% -m pip show ultralytics >nul 2>&1
if errorlevel 1 (
  echo Installing ultralytics...
  %PY_EXE% -m pip install ultralytics
)
%PY_EXE% -m pip show opencv-python >nul 2>&1
if errorlevel 1 (
  echo Installing opencv-python...
  %PY_EXE% -m pip install opencv-python
)

%PY_EXE% track_points.py
endlocal
