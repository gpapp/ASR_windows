# ASR Windows Diarization System

This repository contains a modular, production-grade Speaker Diarization system built for local execution on Windows. It uses an ONNX-based backend for embedding extraction and clustering.

## System Requirements
*   **Python:** 3.10+
*   **Dependencies:** Installed via `uv` in the project venv
*   **System Tools:** `ffmpeg` must be available on the system PATH

## Quick Start

### 1. Start the Server
```powershell
python server.py
```

Server will be available at `http://127.0.0.1:8000`.

### 2. Process Audio Files
```powershell
python transcribe.py meeting.mp4 --num-speakers 4
```

## Diarization Workflow

```
Audio Input → VAD → Energy-dip Splitting → Feature Extraction → Embedding → Clustering → Boundary Refinement → Speaker Identification
```

### Key Options
- `--num-speakers N` - Force exact speaker count
- `--diarization-threshold X` - Distance threshold (default: 0.35, higher = fewer clusters)
- `--voiceprints PATH` - Use known voiceprints for identification
- `--vad-threshold X` - VAD speech probability cutoff (default: 0.5)

### Technical Details
- **Embedding Model:** Wespeaker ecapa-tdnn512 (192-dimensional)
- **Sliding Window:** 2.0s window, 0.75s stride (reduced from 3.0s/1.5s to limit cross-speaker contamination)
- **Energy-dip Splitting:** Long VAD segments (>5s) are split at natural energy minima before embedding extraction, reducing spill-over when speakers alternate with minimal silence
- **Boundary Refinement:** After clustering, each speaker transition is re-examined with 0.5s sub-windows (0.1s stride) to find the precise switch point
- **Max Clusters:** 15 (capped to prevent over-segmentation), then greedy-merged below 0.25 cosine distance
- **Speaker Gap:** 1.0s max gap before a same-speaker segment is split (reduced from 2.0s)
- **Island Absorption:** Isolated segments shorter than 1.0s surrounded by the same speaker are absorbed (reduced from 2.1s)
- **Configuration:** All tunable parameters in `config/thresholds.json`
- **CMN:** Cepstral Mean Normalization applied per window (critical for speaker discrimination)

## Voiceprint Management

### Extracting Training Segments
```powershell
# Identify known speakers and extract their segments:
uv run voiceprint_mgmt.py extract --audio meeting.mp4 --voiceprints voiceprints.json --output segments/

# Also extract unknown speakers for training new voiceprints:
uv run voiceprint_mgmt.py extract --audio meeting.mp4 --voiceprints voiceprints.json --output segments/ --include-unknown

# Extract a specific time range manually:
uv run voiceprint_mgmt.py extract --audio meeting.mp4 --speaker "John" --start 00:05:30 --end 00:06:15 --output segments/
```

### Creating Voiceprints
```powershell
# Create from a single segment
uv run voiceprint_mgmt.py create video.mp4 00:05:30 00:06:15 "John"
```

### Refining Voiceprints
```powershell
# Single speaker
uv run voiceprint_mgmt.py refine --voiceprints voiceprints.json --speaker "John" --segments segments/John/

# Batch refine multiple speakers
uv run voiceprint_mgmt.py mass_refine segments/ --voiceprints voiceprints.json
```

### Voiceprint Format
```json
{
  "John": {
    "pitch_hz": 129.0,
    "pitch_std": 82.9,
    "energy_rms": 0.0607,
    "total_speech_sec": 1243.6,
    "embedding": [ ... 192-dim vector ... ]
  }
}
```

### Duration-Weighted Averaging
When refining, longer segments have proportionally more influence:
- 60s segment = 10x weight of 6s segment
- Blending weight = existing_duration / total_duration

### Quality Guidelines
| Quality | Duration | Notes |
|---------|----------|-------|
| Minimum | 5-10s | Works but may have lower accuracy |
| Good | 30-60s | Reliable matching |
| Excellent | 60s+ | Best accuracy with diverse speech |

## Server Endpoints

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/health` | GET | Health check |
| `/diarize/path` | POST | Diarize audio file by path |
| `/transcribe/paths` | POST | Transcribe audio chunks by path |
| `/shutdown` | POST | Shutdown server |

## Project Structure

```
cohere-diarization/
├── config/
│   ├── thresholds.json        # All tunable parameters
│   └── __init__.py            # Config loader
├── speaker/
│   ├── audio.py               # extract_fbank, generate_sliding_windows, refine_speaker_boundaries
│   ├── embedding.py           # Embedding extraction (ONNX + CMN)
│   ├── matcher.py             # Speaker matching logic
│   ├── profiling.py           # profile_speakers, relabel_by_pitch
│   ├── vad.py                 # run_vad_chunked, run_vad_onnx, split_at_energy_dips
│   └── __init__.py
├── tests/
│   ├── test_cmn_embedding.py  # CMN consistency regression tests
│   ├── test_embedding.py      # Embedding unit tests
│   ├── test_matcher.py        # Matcher unit tests
│   ├── test_offset_indexing.py # ONNX decoder offset regression tests
│   └── test_vad_splitting.py  # VAD splitting tests
├── server.py                  # FastAPI server with ONNX inference
├── transcribe.py              # CLI client
├── voiceprint_mgmt.py         # Voiceprint CLI (create, extract, refine)
├── voiceprint_utils.py        # Shared voiceprint utilities
└── AGENTS.md                  # Detailed technical documentation
```
