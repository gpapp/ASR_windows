# AGENTS.md - Cohere Simple Transcription

## Project Overview
ONNX-based Whisper-style transcription using Cohere's models.

## Server Management

### Starting the Server
```powershell
cd C:\Users\Gergely_Papp\source\ASR\cohere-simple
.venv\Scripts\python.exe server.py
```

### Stopping the Server
```powershell
# Ctrl+C or close the terminal window
```

## Server Endpoints
- `GET /health` - Health check
- `POST /transcribe` - Transcribe audio files

## Audio Input Requirements

### Important: Fixed-Length Input
The ONNX model expects **exactly 30 seconds** of audio (480000 samples at 16kHz). The server automatically:
- **Pads** shorter audio with zeros
- **Truncates** longer audio to first 30 seconds

This fixes dimension mismatch errors like:
```
Attempting to broadcast an axis by a dimension other than 1. 1398 by 6398
```

### Client-Side Chunking
When using VAD to split audio, ensure each chunk is padded to 30s before sending, or let the server handle it (recommended).

## Running Transcription

### Via Python API
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/transcribe",
    json={
        "wav_paths": ["audio.wav"],
        "language": "en"
    }
)
print(response.json())
```

### Via CLI
Use `transcribe.py` in the cohere-diarization project (it handles VAD chunking).

## Known Issues

### Dimension Mismatch
If you see ONNX errors about broadcasting axis dimensions, ensure audio is normalized to 30 seconds (handled by server automatically as of latest fix).

## Dependencies
- onnxruntime
- librosa
- fastapi
- uvicorn
- numpy
- huggingface_hub (for model download)