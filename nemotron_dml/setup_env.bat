@echo off
setlocal
cd /d "%~dp0"
echo Creating Virtual Environment (.venv) in %cd%...
uv venv .venv
call .venv\Scripts\activate

echo Installing DirectML PyTorch stack...
uv pip install torch-directml==0.2.5.dev240914 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 nemo_toolkit[asr] omegaconf tqdm huggingface_hub[hf_xet] librosa soundfile fastapi uvicorn requests sounddevice pynput scipy

echo.
echo Setup Complete! DirectML environment in %cd% is now active.
pause
