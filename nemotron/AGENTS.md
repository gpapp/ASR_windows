# Nemotron ASR Project

This project provides a local, high-performance Automatic Speech Recognition (ASR) system using **NVIDIA Nemotron**. It is optimized for **Intel CPUs** on Windows 11.

## Core Components

- **`server.py`**: The FastAPI model host.
  - Loads `nemotron-speech-streaming-en-0.6b` for ASR.
  - Provides REST endpoints for batched transcription.
- **`stream_client.py`**: Real-time dual-stream client.
  - Captures **Microphone** and **System Loopback** simultaneously.
  - Mixes audio and streams chunks to the server for live transcription.
- **`transcribe.py`**: A standard client using `ffmpeg` silence detection for simple chunking.
- **`DropToTranscribe.bat`**: Windows wrapper for drag-and-drop file transcription.

## Key Technologies

- **Models**: Nemotron-0.6B (ASR).
- **Framework**: NVIDIA NeMo, PyTorch, FastAPI.
- **Audio**: Librosa and FFmpeg for processing and segmentation.

## Optimization Notes

- **Backend**: Runs on **CPU (float32)** by default for stability on Windows Intel systems.
- **Efficiency**: Uses `torch.inference_mode()` and `torch.amp.autocast` to minimize overhead.
- **Batching**: The server processes multiple audio chunks in parallel to maximize CPU utilization.
- **Virtual Env**: Use **`uv`** for managing the `.venv` as per global instructions.

## Running the System

1. **Start the Server**: Run `python server.py`.
2. **Transcribe Files**: Drag audio files onto `DropToTranscribe.bat` or run `python transcribe.py <file>`.
3. **Live Stream**: Run `python stream_client.py` and use the interactive commands.

---
*Note: An experiment with DirectML (iGPU) was conducted but rolled back due to stability issues with the RNNT decoder. The progress of that experiment is preserved in the `dml_exp` folder.*
