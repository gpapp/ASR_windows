@echo off
echo ============================================
echo  Cohere Transcribe - Environment Setup
echo ============================================
echo.

where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 'uv' is not installed or not in PATH.
    echo Install it from: https://github.com/astral-sh/uv
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
uv venv .venv

echo [2/3] Installing dependencies...
call .venv\Scripts\activate
uv pip install ^
    requests ^
    tqdm ^
    pydub"

if %errorlevel% neq 0 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Setup complete!
echo.
echo Drop audio/video files onto DropToTranscribe.bat to transcribe.
pause
