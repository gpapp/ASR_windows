# ASR Windows — Local Speech Recognition Toolkit

A collection of local, offline Automatic Speech Recognition (ASR) tools for Windows 11, each using a different model backend. All tools share a common design: an `ffmpeg`-based audio pipeline, a drag-and-drop `.bat` launcher, and a `uv`-managed virtual environment.

## Prerequisites (all tools)

| Tool | Notes |
|---|---|
| **Python 3.10+** | Required by all tools |
| **[uv](https://github.com/astral-sh/uv)** | Used for all virtual environment management |
| **[ffmpeg + ffprobe](https://ffmpeg.org/)** | Must be on `PATH` |
| **HuggingFace account** | Required for first-time model downloads |

---

## Tools Overview

| Folder | Model | Backend | Diarization | Best For |
|---|---|---|---|---|
| [`cohere-diarization`](cohere-diarization/) | Cohere ONNX INT8 | CPU (ONNX Runtime) | Yes — WeSpeaker + VAD | Meetings, multi-speaker recordings |
| [`cohere-simple`](cohere-simple/) | Cohere ONNX INT8 | CPU (ONNX Runtime) | No | Fast single-speaker transcription |
| [`gemma4`](gemma4/) | Gemma 4 multimodal | Ollama (local LLM) | No | Transcription + summarisation |
| [`nemotron`](nemotron/) | Nemotron 0.6B | CPU / CUDA (NeMo) | No | High-accuracy CPU transcription |
| [`nemotron_dml`](nemotron_dml/) | Nemotron 0.6B | DirectML iGPU (NeMo) | No | iGPU-accelerated transcription |
| [`parakeet`](parakeet/) | Parakeet TDT 1.1B | DirectML iGPU (NeMo) | No | Fast iGPU-accelerated transcription |

---

## Shared Conventions

- Each tool has its own isolated `.venv` (gitignored). Run `setup_env.bat` or `setup.bat` once to create it.
- Models are downloaded automatically on first run and stored in `../models/` (gitignored).
- `DropToTranscribe.bat` — drag audio/video files onto it to transcribe without touching a terminal.
- Supported input formats: `.mp3` `.mp4` `.wav` `.m4a` `.flac` `.mov` `.mkv` `.avi` `.webm` `.ogg`

---

## Quick Start

Pick the tool that fits your hardware and use case, then:

```bat
cd <tool-folder>
setup_env.bat           :: first time only
DropToTranscribe.bat my_recording.mp4
```

---

## Repository Layout

```
ASR/
├── cohere-diarization/   # Production tool: Cohere ONNX + speaker diarization
├── cohere-simple/        # Lightweight tool: Cohere ONNX, no diarization
├── gemma4/               # Experimental: Gemma 4 via Ollama, transcribe + summarise
├── nemotron/             # Nemotron 0.6B on CPU/CUDA via NeMo
├── nemotron_dml/         # Nemotron 0.6B on Intel iGPU via DirectML (experimental)
├── parakeet/             # Parakeet TDT 1.1B on Intel iGPU via DirectML
└── models/               # Shared model cache (gitignored)
```
