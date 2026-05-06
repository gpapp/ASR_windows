# ASR Windows Diarization System

This repository contains a modular, production-grade Speaker Diarization system built for local execution on Windows. It uses an ONNX-based backend for embedding extraction and clustering.

## System Requirements
*   **Python:** 3.10+
*   **Dependencies:** Installed via `uv` in the project venv
*   **System Tools:** `ffmpeg` must be available on the system PATH

## Quick Start

### 1. Start the Server
```powershell
python server.py
```

Server will be available at `http://127.0.0.1:8000`.

### 2. Process Audio Files
```powershell
python transcribe.py meeting.mp4 --num-speakers 4
```

## Diarization Workflow

```
Audio Input → VAD (Voice Activity Detection) → Feature Extraction → Embedding → Clustering → Speaker Segmentation
```

### Key Options
- `--num-speakers N` - Force exact speaker count
- `--diarization-threshold X` - Distance threshold (default: 0.20)
- `--voiceprints PATH` - Use known voiceprints for identification

## Voiceprint Management

### Creating Voiceprints
```powershell
# Create from a single segment
python voiceprint_mgmt.py create video.mp4 00:05:30 00:06:15 "John"

# Extract segments from diarization
python voiceprint_mgmt.py extract --audio meeting.mp4 --diarize diarization.json --speaker "SPEAKER_00" --output segments/
```

### Refining Voiceprints
```powershell
# Single speaker
python voiceprint_mgmt.py refine --voiceprints voiceprints.json --speaker "John" --segments segments/John/

# Batch refine multiple speakers
python voiceprint_mgmt.py mass_refine segments/ --voiceprints voiceprints.json
```

### Voiceprint Format
```json
{
  "John": {
    "pitch_hz": 129.0,
    "pitch_std": 82.9,
    "energy_rms": 0.0607,
    "total_speech_sec": 1243.6,
    "embedding": [ ... 192-dim vector ... ]
  }
}
```

### Duration-Weighted Averaging
When refining, longer segments have proportionally more influence:
- 60s segment = 10x weight of 6s segment
- Blending weight = existing_duration / total_duration

## Server Endpoints

| Endpoint | Method | Description |
|-----------|-------|-------------|
| `/health` | GET | Health check |
| `/diarize/path` | POST | Diarize audio file by path |
| `/shutdown` | POST | Shutdown server |

## Project Structure

- `server.py` - FastAPI server with ONNX inference
- `transcribe.py` - Client orchestrator
- `voiceprint_mgmt.py` - Voiceprint CLI (create, extract, refine, mass_refine)
- `voiceprint_utils.py` - Shared voiceprint utilities