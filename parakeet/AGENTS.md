# Parakeet ASR Project (DirectML Experiment)

This project provides high-performance Automatic Speech Recognition (ASR) using **NVIDIA Parakeet** and **Canary** models. It is optimized for **Intel Iris Xe Graphics** using **DirectML**.

## Core Components

- **`transcribe.py`**: Local inference script using `nvidia/parakeet-tdt-0.6b-v3`. Supports multi-file processing and progress tracking.
- **`ASR.bat`**: CLI wrapper for transcribing one or more files.
- **`DropToTranscribe.bat`**: Drag-and-drop utility for easy transcription.
- **`setup.bat`**: Sets up the isolated virtual environment and installs dependencies.
- **`directml_patch.py`**: DirectML compatibility patches (includes `PyArrow` and `SIGKILL` fixes).

## Key Technologies

- **Models**: Parakeet TDT 0.6B, Canary 1B Flash.
- **Backend**: PyTorch with **`torch-directml`** for iGPU acceleration.
- **Framework**: NVIDIA NeMo, PyTorch.
- **Environment**: Managed via **`uv`**.

## Optimization & Stability Notes

- **Backend**: Optimized for **DirectML (float32)** on Intel hardware.
- **Selective Offloading**: To prevent "version_counter" and stability errors on DirectML, the model Encoder is moved to the iGPU, while the Decoder/Joint nets are kept on the CPU.
- **Inference Stability**: Uses `.detach().cpu()` when moving tensors from DirectML to CPU to ensure inference tensors don't track gradients or version counts incorrectly.
- **Cleaner Transcripts**: The server now automatically strips NeMo special tokens (e.g., `<|startoftranscript|>`) and collapses repeating punctuation artifacts using regex.
- **Efficiency**: Utilizes `torch.no_grad()` and disabled preprocessor dithering/padding for faster processing.

## Running the System

1. **Initial Setup**: Run `setup.bat`. This will create a `.venv` and install all necessary dependencies.

2. **Transcription**:
   - Use `DropToTranscribe.bat <audio_file>`
   - Example: `DropToTranscribe.bat input.wav`
   - Alternatively, run via `uv`:

     ```bash
     uv run transcribe.py input.wav
     ```
