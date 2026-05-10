@echo off
REM This script initializes the Granite Diarization environment

IF NOT EXIST .venv (
    python -m venv .venv
)

echo Activating virtual environment...
call .\.venv\Scripts\activate

echo Installing dependencies from requirements.txt...
uv pip install -r requirements.txt --upgrade

echo Model cache setup (if needed)...
:: Add logic here to ensure model download/caching happens correctly for ibm-granite/granite-speech-4.1-2b
:: For now, assume standard HF download handles it.

echo Installing Torch CUDA packages
uv pip install torch torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu126 


echo Setup complete! Remember to run the script again after updating dependencies.