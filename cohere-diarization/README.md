# cohere-diarization

Production-ready transcription server with **multi-speaker diarization**. Uses the Cohere ONNX INT8 model for transcription and a custom WeSpeaker ONNX + Silero VAD pipeline for speaker identification — no Pyannote, no GPU required.

## Features

- Multi-speaker diarization (4+ speakers) on CPU
- Voice profiling per speaker: pitch, energy, gender hint, total speech time
- Speaker legend written at the top of every transcript
- Stable `SPEAKER1`/`SPEAKER2` labels ordered by pitch across runs
- FastAPI streaming server with health, metrics, and optional API-key auth
- Async client with progress bars, retry logic, and batch transcription
- `DropToTranscribe.bat` for drag-and-drop use
- Output formats: `txt`, `srt`, `json`

## How It Works

```
Audio file
  └─ ffmpeg → 16kHz mono WAV
       └─ Silero VAD → speech timestamps
            └─ WeSpeaker ONNX → 256-dim speaker embeddings (3s sliding windows)
                 └─ Agglomerative Clustering (cosine, threshold=0.28) → speaker labels
                      └─ Island-merge post-pass → clean speaker turns
                           └─ Voice profiler (pitch F0, RMS energy)
                                └─ Cohere ONNX INT8 → transcription per chunk
                                     └─ .txt / .srt / .json output
```

## Setup

```bat
setup_env.bat
```

Copy `.env-template` to `.env` and add your HuggingFace token (needed for the WeSpeaker embedding model download):

```env
TRANSCRIBE_HF_TOKEN=hf_your_token_here
TRANSCRIBE_ENABLE_DIARIZATION=true
```

## Usage

### Drag and drop
Drop audio/video files onto `DropToTranscribe.bat`. The server starts automatically if not running.

### Command line
```powershell
# Start the server (first run downloads ~2.9 GB of models)
.venv\Scripts\python.exe server.py

# Transcribe
.venv\Scripts\python.exe transcribe.py meeting.mp4
.venv\Scripts\python.exe transcribe.py meeting.mp4 --format srt
.venv\Scripts\python.exe transcribe.py meeting.mp4 --format json
```

### Client options
| Flag | Default | Description |
|---|---|---|
| `--format` / `-f` | `txt` | Output format: `txt`, `srt`, `json` |
| `--server` / `-s` | `http://127.0.0.1:8000` | Server URL |
| `--language` / `-l` | `en` | Transcription language |
| `--num-speakers` | `None` | Exact number of speakers (improves diarization if known) |
| `--diarization-threshold` | `None` | Distance threshold for clustering (overrides server default) |
| `--api-key` / `-k` | `None` | API key for authentication |
| `--timeout` / `-t` | `120` | Request timeout in seconds |
| `--language` / `-l` | `en` | ISO 639-1 language code |
| `--batch-size` / `-b` | `4` | Chunks per transcription request |
| `--api-key` / `-k` | — | API key (if server auth enabled) |

## Output Example

```
============================================================
SPEAKER VOICE PROFILES
============================================================
  SPEAKER1: pitch=118Hz (±93Hz)  energy=0.0835  speech=4s  gender=male
  SPEAKER2: pitch=127Hz (±84Hz)  energy=0.0600  speech=112s  gender=male
  SPEAKER3: pitch=129Hz (±84Hz)  energy=0.0668  speech=44s  gender=male
  SPEAKER4: pitch=134Hz (±105Hz)  energy=0.0515  speech=5s  gender=male
============================================================

[00:00:00] SPEAKER3:
I don't know. He's still not feeling so well, but with kids, you never know...

[00:00:38] SPEAKER2:
So, shall we maybe already start? Because I saw that we, anyhow, have this word file.
```

## Server Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Model status and device info |
| `/diarize/path` | POST | Diarize a WAV file, streams NDJSON progress + result with speaker profiles |
| `/transcribe/paths` | POST | Batch transcribe WAV files by path |
| `/transcribe/upload` | POST | Batch transcribe uploaded files |
| `/metrics` | GET | Prometheus metrics |

## Key Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_HF_TOKEN` | — | HuggingFace token for model download |
| `TRANSCRIBE_ENABLE_DIARIZATION` | `true` | Enable VAD + speaker diarization |
| `TRANSCRIBE_DIARIZATION_THRESHOLD` | `0.28` | Agglomerative clustering distance threshold |
| `TRANSCRIBE_HOST` | `127.0.0.1` | Bind address |
| `TRANSCRIBE_PORT` | `8000` | Bind port |
| `TRANSCRIBE_WORKERS` | `2` | Thread pool size |
| `TRANSCRIBE_API_KEYS` | — | Comma-separated API keys (blank = no auth) |

## Testing

A 3-minute test clip and expected transcript are in `tests/diarization_fix/`.

```powershell
# Fast diarization-only probe (no transcription wait)
.venv\Scripts\python.exe tests\diarization_fix\probe_diarize.py tests\diarization_fix\test_3min.mp4
```

## Models

| Model | Source | Size | Purpose |
|---|---|---|---|
| Cohere ONNX INT8 | `gn64/cohere-transcribe-onnx-int8` | ~2.9 GB | Transcription |
| WeSpeaker ResNet34-LM | `onnx-community/wespeaker-voxceleb-resnet34-LM` | ~100 MB | Speaker embeddings |
| Silero VAD | `snakers4/silero-vad` (Torch Hub) | ~2 MB | Voice activity detection |
