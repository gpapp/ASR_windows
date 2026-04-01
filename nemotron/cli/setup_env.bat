@echo off
echo Creating Virtual Environment (CPU)...
python -m venv venv
call venv\Scripts\activate

echo Installing Standard PyTorch (CPU)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo Installing NeMo and tools...
pip install nemo_toolkit[asr] omegaconf tqdm huggingface_hub[hf_xet] librosa soundfile
echo.
echo Setup Complete! Use DropToTranscribe.bat to start.
pause
