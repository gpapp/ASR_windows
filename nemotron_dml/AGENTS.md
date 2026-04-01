# Nemotron ASR Project (DirectML Experiment)

This project provides a local, high-performance Automatic Speech Recognition (ASR) system using **NVIDIA Nemotron**. It is optimized for **Intel Iris Xe Graphics** using **DirectML**.

## Core Components

- **`server_dml.py`**: The FastAPI model host.
  - Loads `nemotron-speech-streaming-en-0.6b` for ASR.
  - Uses **`torch_directml`** to accelerate inference on the iGPU.
  - Applies custom patches for **`masked_scatter`** stability on Intel hardware.
- **`stream_client.py`**: Real-time dual-stream client.
  - Captures **Microphone** and **System Loopback** simultaneously.
  - Mixes audio and streams chunks to the server for live transcription.
- **`transcribe.py`**: A standard client using `ffmpeg` silence detection for simple chunking.
- **`DropToTranscribe.bat`**: Windows wrapper for drag-and-drop file transcription.

## Key Technologies

- **Models**: Nemotron-0.6B (ASR).
- **Backend**: PyTorch with **`torch-directml`**.
- **Framework**: NVIDIA NeMo, PyTorch, FastAPI.
- **Audio**: Librosa and FFmpeg for processing and segmentation.

## Optimization Notes

- **Backend**: Runs on **DirectML (float32)** by default for stability on Intel Iris Xe.
- **Stability**: Uses `directml_patch.py` to handle missing operators like `masked_scatter`.
- **Hybrid Inference**: The heavy **Encoder** remains on the **iGPU (DirectML)**, while the lightweight **RNNT Decoder** is moved to the **CPU** to resolve `version_counter` incompatibilities.
- **Efficiency**: Uses `torch.no_grad()` to minimize overhead and avoid versioning issues.
- **Virtual Env**: Uses a dedicated **`.venv`** within the `dml_exp` folder.

## Running the System

1. **Start the Server**: Run `python server_dml.py`.
2. **Transcribe Files**: Drag audio files onto `DropToTranscribe.bat` or run `python transcribe.py <file>`.
3. **Live Stream**: Run `python stream_client.py` and use the interactive commands.
