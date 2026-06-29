# AGENTS.md - Nemotron Diarization Project

## Project Overview
This is a production-grade ASR + speaker diarization pipeline using NVIDIA Nemotron 3.5 ASR streaming model with ONNX Runtime GenAI and voiceprint recognition.

## Code Structure

### Core Modules
- **`server.py`** — Thin FastAPI app: lifespan, endpoints, middleware wiring
- **`settings.py`** — Application settings (Pydantic BaseSettings) and logging configuration
- **`model_state.py`** — Global model state (Nemotron model, processor, tokenizer, VAD, embeddings)
- **`model_loader.py`** — Model downloading and initialization (HuggingFace Hub + ONNX Runtime GenAI)
- **`transcriber.py`** — ASR inference using Nemotron streaming API, hallucination cleaner

### API Package (`api/`)
- **`schemas.py`** — Pydantic request/response models for all endpoints
- **`security.py`** — API key authentication and path validation
- **`exceptions.py`** — Custom exception hierarchy and FastAPI error handlers
- **`middleware.py`** — Request logging, CORS, rate limiting setup

### Diarization Package (`diarization/`)
- **`pipeline.py`** — `Diarizer` class: full VAD → embed → cluster → match → refine pipeline
- **`clustering.py`** — Greedy merge, cluster capping, known-speaker matching logic
- **`segment_ops.py`** — Segment post-processing: collapse, absorb islands, eliminate ghosts

### Speaker Package (`speaker/`)
Pre-existing utilities:
- **`audio.py`** — fbank extraction, sliding windows, boundary refinement
- **`vad.py`** — Silero VAD wrappers (chunked, ONNX, energy-dip splitting)
- **`profiling.py`** — Pitch/energy extraction, relabeling by pitch
- **`matcher.py`** — Voiceprint distance computation (embedding + pitch + energy)
- **`embedding.py`** — ONNX embedding extraction, batch processing

### Streaming
- **`streaming.py`** — WebSocket endpoint for dual-channel real-time transcription + diarization

## Server Management

### Starting the Server
```powershell
Start-Process -FilePath "cmd.exe" -ArgumentList '/c', 'cd /d "C:\Users\Gergely_Papp\source\ASR\nemotron-diarization" && call .venv\Scripts\activate && python server.py' -WorkingDirectory "C:\Users\Gergely_Papp\source\ASR\nemotron-diarization" -WindowStyle Normal
```

### Stopping the Server
**Always shutdown via API before starting a new instance:**
```powershell
Invoke-WebRequest -Uri 'http://127.0.0.1:8001/shutdown' -Method POST -UseBasicParsing
```

### Server Endpoints
- `GET /health` - Health check
- `POST /diarize/path` - Diarize audio file by local path
- `POST /transcribe/upload` - Transcribe uploaded audio file
- `POST /transcribe/paths` - Transcribe audio chunks by local path list
- `POST /shutdown` - Shutdown server (requires API key)

## Voiceprints (Speaker Recognition)
Same voiceprints.json format and workflow as cohere-diarization.

## Technical Details

### ASR Model
- **Model**: `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`
- **API**: ONNX Runtime GenAI (StreamingProcessor + Generator)
- **Chunk size**: 560ms streaming sub-chunks, 120s batch segments
- **Word timestamps**: ~560ms granularity from streaming chunk positions
- **Language**: 40 locales via lang_id prompt conditioning
- **Model size**: ~800MB (INT4 quantized)

### Transcoding
Audio is processed in two modes:
1. **Batch** (REST endpoints): 120s chunks fed through streaming API, returning sub-chunk timed segments
2. **Per-segment** (WS endpoint): Each diarization segment transcribed independently via streaming API

## Environment Variables
- `TRANSCRIBE_SERVER_URL` - Server URL
- `TRANSCRIBE_API_KEY` - API key
- `TRANSCRIBE_ASR_LANGUAGE` - ASR language code
- `TRANSCRIBE_NUM_SPEAKERS` - Default number of speakers
- `TRANSCRIBE_DIARIZATION_THRESHOLD` - Default threshold
