@echo off
setlocal
cd /d "%~dp0"
title 10-step-on-people demo

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment was not found.
    echo Follow the installation steps in README.md first.
    pause
    exit /b 1
)

if not exist "data\tokenizer\tokenizer.json" (
    echo [ERROR] The tokenizer was not found.
    echo Run: .venv\Scripts\python.exe -m onebit_llm.prepare
    pause
    exit /b 1
)

if not exist "artifacts\checkpoints\bitnet\best.pt" (
    echo [ERROR] The trained BitNet checkpoint was not found.
    echo Run: .venv\Scripts\python.exe -m onebit_llm.train --model all --config configs/demo.yaml
    pause
    exit /b 1
)

if not exist "artifacts\checkpoints\baseline\best.pt" (
    echo [ERROR] The trained FP baseline checkpoint was not found.
    echo Run: .venv\Scripts\python.exe -m onebit_llm.train --model all --config configs/demo.yaml
    pause
    exit /b 1
)

echo Starting the 10-step-on-people demo...
echo The browser will open automatically. Press Ctrl+C here to stop.
echo.
".venv\Scripts\python.exe" -u app.py
set "DEMO_EXIT=%ERRORLEVEL%"

if not "%DEMO_EXIT%"=="0" (
    echo.
    echo [ERROR] Demo exited with code %DEMO_EXIT%.
    pause
)

endlocal & exit /b %DEMO_EXIT%
