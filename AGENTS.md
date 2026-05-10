<|think|>

## Internal Reasoning Style
Think caveman. Short. Dense. No fluff.
Pattern: [thing] [action] [reason]. [next step].
Drop: articles, hedging, pleasantries, filler words.
Fragments OK. Technical terms exact.

## Output Style
Final answers: normal clear prose for humans.
Code: unchanged, full, no shortcuts.
Errors: quoted exact.

# ASR Windows Project Conventions

This repository is a collection of local Automatic Speech Recognition (ASR) tools built for Windows, utilizing a specific workflow that may not be obvious from standard file structure.

## ⚙️ Prerequisites & Global Dependencies
- **Python Version:** Requires Python 3.10+ for all sub-tools.
- **System Tools:** `ffmpeg` and `ffprobe` must be available on the system `PATH`.
- **Environment Management:** Each sub-tool contains an isolated virtual environment (`.venv`). Must run the dedicated setup script (`setup_env.bat` or `setup.bat`) *the first time* in a tool folder to initialize dependencies.

## 🚀 Standard Workflow
- **Execution Flow:** The standard process is to `cd <tool-folder>` $\rightarrow$ run `setup_env.bat` (only needed once) $\rightarrow$ execute the transcription via `DropToTranscribe.bat`.
- **Input Handling:** The primary method of running transcription is by dragging audio/video files onto the specialized `DropToTranscribe.bat` launcher script in the respective tool directory.
- **Model Caching:** All models are downloaded automatically on first run and are cached in a system directory outside the tool folders (`../models/`).

## 💻 Hardware & Backend Quirks
The repository supports multiple specialized hardware backends; the choice of tool dictates the target hardware:
- **General:** Use `nemotron/` for CPU/CUDA fallback.
- **iGPU Acceleration:** Use `nemotron_dml/` or `parakeet/` if targeting an Intel iGPU via DirectML for faster, specific performance requirements.

## 🗂️ Directory Ownership
- **`cohere-diarization/`**: The dedicated folder for multi-speaker diarization, which is the most complex feature set.
- **Models Location:** Model data is *always* stored in the repository root's `models/` (gitignored).