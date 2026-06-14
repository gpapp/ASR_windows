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
uv venv .venv --clear

echo [2/3] Installing dependencies...
call .venv\Scripts\activate
uv pip install --upgrade -r requirements.txt

if %errorlevel% neq 0 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)
:: ONNX is faster 
:: uv pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/xpu

:: Check for NVIDIA CUDA and upgrade PyTorch if available
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo CUDA detected - installing PyTorch with CUDA support...
    uv pip install --upgrade onnxruntime-gpu
    uv pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu130
    if %errorlevel% neq 0 (
        echo [WARNING] CUDA PyTorch install failed, falling back to CPU version.
    ) else (
        echo CUDA PyTorch installed successfully.
    )
) else (
    echo.
    echo No CUDA detected - using CPU-only PyTorch from requirements.txt.
)

echo.
echo [3/3] Setup complete!
echo.
echo NOTE: The Cohere ONNX model (~2.9 GB) will be downloaded automatically
echo       on first use from: gn64/cohere-transcribe-onnx-int8
echo       Model files will be saved to: models\
echo.
echo Drop audio/video files onto DropToTranscribe.bat to transcribe.
pause
