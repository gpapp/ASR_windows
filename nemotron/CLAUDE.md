# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local, high-performance ASR (Automatic Speech Recognition) system using NVIDIA Nemotron on Windows 11, optimized for Intel CPUs.

## Environment Setup

Uses `uv` for virtual environment management (no `pyproject.toml` — deps are declared in `setup_env.bat`):

```bat
setup_env.bat        # creates .venv and installs all deps
```

**Activate environment:**
```bash
.venv/Scripts/activate
```

## Running the System

**Start the inference server:**
```bash
python server.py
# FastAPI on http://127.0.0.1:8000
# Loads nvidia/nemotron-speech-streaming-en-0.6b at startup
```

**Transcribe files:**
```bash
python transcribe.py path/to/audio.mp4
# OR drag files onto DropToTranscribe.bat (auto-starts server if needed)
```

**Health check:** `GET http://127.0.0.1:8000/health`

## Architecture

Client-server split for the primary workflow:

1. **`transcribe.py`** (client): Uses ffmpeg to convert audio → 16kHz mono WAV, runs silence detection (`-35dB` threshold, `2s` min), extracts speech chunks (≤30s), sends batches of 4 chunk paths via `POST /transcribe` to the server, assembles timestamped transcript saved as `<filename>.txt`.

2. **`server.py`** (FastAPI server): Loads the NeMo RNNT model once at startup, receives WAV file paths, runs batched inference via `model.forward()` → RNNT decoder → `tokenizer.ids_to_text()`, returns JSON results.

3. **`cli/transcribe.py`**: Older standalone version — loads the model directly (no server), uses fixed 30s chunks with silence-aware speaker detection.

**Device selection:** Defaults to CPU/float32 for stability; auto-upgrades to CUDA/float16 if a GPU is detected. `torch.compile` is disabled by default (enabled only with CUDA or MSVC available).

## Key Constants

- Model: `nvidia/nemotron-speech-streaming-en-0.6b` (downloaded from HuggingFace at first run)
- Server: `127.0.0.1:8000`
- Chunk duration: 30s max, batch size: 4
- Supported formats: `.mp3`, `.mp4`, `.wav`, `.m4a`, `.flac`, `.mov`, `.mkv`
- External dependency: `ffmpeg` must be on PATH

## Notes

- `stream_client.py` (real-time mic + loopback capture) is referenced in AGENTS.md but not currently present on disk
- A DirectML (iGPU) experiment was rolled back due to RNNT decoder instability; remnants in `__pycache__/directml_patch.cpython-312.pyc` and `dml_exp/` folder
