@echo off
REM ============================================================
REM  Comma 4 Drive Visualizer - Windows build script
REM  Double-click this file to build Comma4Visualizer.exe
REM ============================================================
setlocal

echo.
echo  Comma 4 Drive Visualizer - building standalone EXE
echo  ---------------------------------------------------
echo.

REM Find Python
where python >nul 2>nul
if errorlevel 1 (
    echo  ERROR: Python not found on PATH.
    echo  Install Python 3.10+ from https://python.org and tick
    echo  "Add Python to PATH" during setup, then run this again.
    echo.
    pause
    exit /b 1
)

echo  [1/3] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install pygame pycapnp numpy opencv-python-headless zstandard imageio-ffmpeg pyinstaller
if errorlevel 1 (
    echo.
    echo  ERROR: dependency install failed. See messages above.
    pause
    exit /b 1
)

echo.
echo  [2/3] Building EXE with PyInstaller...
python -m PyInstaller --noconfirm comma4_viz.spec
if errorlevel 1 (
    echo.
    echo  ERROR: build failed. See messages above.
    pause
    exit /b 1
)

echo.
echo  [3/3] Done.
echo.
echo  Your program is here:
echo      dist\Comma4Visualizer.exe
echo.
echo  Double-click it to run. No command line needed.
echo.
pause
