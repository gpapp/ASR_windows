@echo off
echo Creating Virtual Environment (.venv)...
uv venv .venv
call .venv\Scripts\activate

echo Installing Standard PyTorch (CPU)...
uv pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cpu

echo Installing NeMo and tools...
uv pip install nemo_toolkit[asr] omegaconf tqdm huggingface_hub[hf_xet] librosa soundfile fastapi uvicorn requests
uv pip install sounddevice pynput scipy

echo.
echo Setup Complete! CPU operation is now active.
pause
