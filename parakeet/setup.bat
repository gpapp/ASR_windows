@echo off
setlocal

echo Setting up environment for Parakeet ASR with DirectML...

if exist .venv (
    echo Removing existing .venv...
    rd /s /q .venv
)

echo Creating new .venv...
uv venv .venv
if %ERRORLEVEL% neq 0 (
    echo Failed to create .venv
    exit /b %ERRORLEVEL%
)

echo Installing dependencies...
uv pip install "torch-directml>=0.2.5.dev240914" huggingface-hub[hf-xet] ml-dtypes nemo-toolkit[asr] torch torchvision torchaudio accelerate transformers librosa soundfile pip tqdm fastapi uvicorn 
if %ERRORLEVEL% neq 0 (
    echo Failed to install dependencies
    exit /b %ERRORLEVEL%
)

echo Setup complete.
pause
