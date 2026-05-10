# Granite Diarization ASR Tool

This tool implements Automatic Speech Recognition (ASR) and speaker diarization using the ibm-granite/granite-speech-4.1-2b model, structured to follow the repository's standard workflow.

## Prerequisites
* Python 3.10+
* ffmpeg and ffprobe available on system PATH.
* **Dependencies:** `transformers`, `torch`, `soundfile`, and potentially specific diarization libraries must be installed (e.g., using `pip install -r requirements.txt`).

To set up the environment in this folder:
1. Run `./setup_env.bat` to initialize dependencies and cache assets.
2. Ensure all required PyTorch/transformers backend components are available.

## Standard Workflow (Recommended)
The standard workflow for transcribing a file is as follows:

1.  **Setup:** Execute `.\setup_env.bat`.
2.  **Transcribe:** Drag your audio or video file and drop it onto the **`DropToTranscribe.bat`** launcher script inside this directory.

This batch script automatically activates the virtual environment, calls the core processing logic in `server.py`, and provides structured output containing both the ASR transcript and diarization markers.