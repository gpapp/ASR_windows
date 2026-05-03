# ASR Windows Diarization System

This repository contains a modular, production-grade Automatic Speech Recognition (ASR) and Speaker Diarization system built for local execution on Windows. It utilizes an external server backend (`cohere-diarization/server.py`) for heavy lifting (transcription, diarization clustering).

## ⚙️ System Requirements
*   **Python:** 3.10+
*   **Dependencies:** `torch`, `torchaudio`, `numpy`, `onnxruntime`, `librosa`, `fastapi`, `uvicorn` etc. (Install via `requirements.txt`)
*   **System Tools:** `ffmpeg` and `ffprobe` must be available on the system PATH.

## 🚀 Workflow Guide

The standard process is now highly modular:

### 1. Server Setup & Initialization
Before running any client code, you *must* start the server to download models and initialize the state.
```bash
python cohere-diarization/server.py
```
*   **Output:** The server logs will indicate successful model loading (Encoder, Decoder, VAD, Embedding) and confirm it is ready on `http://127.0.0.1:8000`.

### 2. Transcribing Files (Client Workflow)
Use the main script to process one or multiple files, which handles conversion, calls the server API, and outputs results in various formats.
```bash
python cohere-diarization/transcribe.py --files <file1.mp3> <file2.wav> -f srt -b 8
# Use -s to specify a custom server URL if needed:
# python cohere-diarization/transcribe.py -s http://localhost:8001 ...
```

### 3. Evaluation & Testing (Validation Workflow)
Use the dedicated evaluator script against the contained test data (`mini-test-data/`) to score performance metrics (WER, Overlap Score).
```bash
python cohere-diarization/evaluator.py
```

## 🧩 Module Breakdown
*   **`server.py`**: The API endpoint and core processing engine. It handles file validation, model loading (`ModelState`), transcription calls, and the complex diarization logic (VAD $\rightarrow$ FBank $\rightarrow$ Clustering).
*   **`transcribe.py`**: Client-side orchestrator. Handles local conversion of audio files to WAV format for API upload.
*   **`evaluator.py`**: Dedicated testing harness. Uses the server client methods (`TranscriptionClient`) to run benchmarks against the known test dataset splits (`train`, `valid`, `test`).

## ⚠️ Critical Notes on ML Placeholder
The embedding generation and scoring sections currently use random data (placeholder). For production, replace the logic in **`generate_embedding_and_score`** with actual model inference calls using the architecture defined in `server.py`.