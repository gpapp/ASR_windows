# FastAPI Transcription & Diarization Server

A production-ready asynchronous Audio Transcription and Speaker Diarization service powered by FastAPI, `onnxruntime`, Cohere transcription models, Pyannote, and Silero VAD.

## Architecture & Features
- **FastAPI Backend:** Fully asynchronous file processing.
- **Cohere Transcription (ONNX):** Uses INT8 quantized Cohere encoder/decoder models for blazing fast transcription on CPUs.
- **High-Performance Diarization:** Uses `pyannote/speaker-diarization-3.1` (ECAPA-TDNN) for speaker clustering.
- **Intelligent VAD Pre-Filtering:** Integrates `silero-vad` natively. It dynamically strips silence out of audio *before* running Pyannote, significantly accelerating the diarization of long files with dead air.
- **KV Cache Pooling:** Memory-efficient token generation pooling.
- **Async Python Client (`transcribe.py`):** Automatically batches, detects silence, groups chunks by speaker, and fetches transcripts seamlessly with progress bars.

## Prerequisites

- **OS:** Cross-platform (tested on Windows 11).
- **Environment:** `uv` (recommended) or `pip` activated environment.
- **Python:** 3.10+
- **System Tools:** `ffmpeg` and `ffprobe` (must be installed and added to PATH).

## Installation

1. **Install Python dependencies:**
```bash
uv pip install -r requirements.txt
```

2. **Configure Environment:**
Copy `.env-template` to `.env` and fill out the details:
```env
TRANSCRIBE_HOST="127.0.0.1"
TRANSCRIBE_PORT=8000
TRANSCRIBE_HF_TOKEN="hf_your_huggingface_token" # Required for Pyannote 3.1
TRANSCRIBE_ENABLE_DIARIZATION=true
TRANSCRIBE_WORKERS=2
```
*Note: Make sure your HuggingFace account has accepted the terms for the `pyannote/speaker-diarization-3.1` model.*

## Usage

### 1. Starting the Server
```powershell
uv run python server.py
```
*The server will download the Cohere ONNX models and Pyannote weights automatically on first run.*

### 2. Transcribing Audio (Client)
The `transcribe.py` script acts as an intelligent client. It uses `ffmpeg` to pre-process the audio, hits the `/diarize/path` endpoint to identify who is speaking and when, and then batches the discrete speaker chunks to the `/transcribe/paths` endpoint.

```powershell
uv run python transcribe.py my_audio_file.wav --format srt
```

**Client Arguments:**
- `--format`: `txt` (default), `srt`, or `json`.
- `--server`: The server URL (default: `http://127.0.0.1:8000`).
- `--language`: ISO language code (e.g. `en`, `es`).
- `--batch-size`: Number of chunks to transcribe concurrently.

## Pipeline Breakdown
This server optimizes diarization compute overhead by using a Concatenate & Remap strategy:

1. **VAD Pass:** Silero VAD (via Torch Hub) scans the audio tensor to detect all active speech segments.
2. **Dense Tensor:** Silence is stripped out, and the speech segments are concatenated into a "dense" timeline.
3. **Diarize:** `pyannote` clusters the dense audio tensor, saving massive amounts of compute time (especially on podcasts or meetings with long pauses).
4. **Remap:** The timestamps from the dense audio are mathematically re-aligned back to their original real-world timestamps before being returned to the client.

## Built With
- `FastAPI`
- `onnxruntime`
- `pyannote.audio`
- `silero-vad`
- `librosa`
- `aiohttp`