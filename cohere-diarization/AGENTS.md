# AGENTS.md - Cohere Diarization Project

## Project Overview
This is a custom ONNX-based speaker diarization pipeline with voiceprint recognition.

## Server Management

### Starting the Server
```powershell
# From project root - use visible window so you can kill it manually
Start-Process -FilePath "cmd.exe" -ArgumentList '/c', 'cd /d "C:\Users\Gergely_Papp\source\ASR\cohere-diarization" && call .venv\Scripts\activate && python server.py' -WorkingDirectory "C:\Users\Gergely_Papp\source\ASR\cohere-diarization" -WindowStyle Normal
```

### Stopping the Server
**Always shutdown via API before starting a new instance:**
```powershell
Invoke-WebRequest -Uri 'http://127.0.0.1:8000/shutdown' -Method POST -UseBasicParsing
```

The server will automatically exit after 1 second.

### Server Endpoints
- `GET /health` - Health check
- `GET /metrics` - Prometheus-style metrics
- `POST /diarize/path` - Diarize audio file by local path
- `POST /transcribe/upload` - Transcribe uploaded audio file
- `POST /transcribe/paths` - Transcribe audio chunks by local path list
- `POST /transcribe` - Transcribe audio (generic)
- `POST /shutdown` - Shutdown server (requires API key)

## Voiceprints (Speaker Recognition)

### File Location
Voiceprints are stored in `voiceprints.json` in the project folder (`C:\Users\Gergely_Papp\source\ASR\cohere-diarization\voiceprints.json`).

### Voiceprint Format
```json
{
  "SpeakerName": {
    "pitch_hz": 129.0,
    "pitch_std": 82.9,
    "energy_rms": 0.0607,
    "total_speech_sec": 1243.6,
    "embedding": [ ... 192-dimensional embedding ... ]
  }
}
```

### Using Voiceprints
Pass `--voiceprints` flag to transcribe.py:
```powershell
python transcribe.py "video.mp4" --num-speakers 4 --voiceprints "C:\Users\Gergely_Papp\source\ASR\cohere-diarization\voiceprints.json"
```

### Auto-loading
If `voiceprints.json` exists in the cohere-diarization folder, it's auto-loaded without needing `--voiceprints` flag.

## Command-line Options

### Diarization
- `--num-speakers N` - Force exact number of speakers (improves clustering)
- `--diarization-threshold X` - Distance threshold for clustering (default: 0.35)
- `--voiceprints PATH` - Path to voiceprints.json file
- `--vad-threshold X` - VAD speech probability cutoff (default: 0.5)
- `--vad-min-speech MS` - VAD minimum speech chunk length in ms (default: 250)

### Server
- `--shutdown` - Shutdown the server instead of transcribing

### Other
- `--server URL` - Server URL (default: http://127.0.0.1:8000)
- `--api-key KEY` - API key for authentication
- `--timeout SECONDS` - Request timeout (default: 120)

## Environment Variables
- `TRANSCRIBE_SERVER_URL` - Server URL
- `TRANSCRIBE_API_KEY` - API key
- `TRANSCRIBE_NUM_SPEAKERS` - Default number of speakers
- `TRANSCRIBE_DIARIZATION_THRESHOLD` - Default threshold
- `TRANSCRIBE_VAD_THRESHOLD` - VAD threshold
- `TRANSCRIBE_VAD_MIN_SPEECH_DURATION_MS` - VAD min speech duration

## Debug Mode

Enable debug output by setting `"debug": true` in `config/thresholds.json`:
```json
{
  "debug": true
}
```

When enabled, server logs detailed speaker matching info including:
- Cluster to voiceprint distances
- Best match selection reasoning
- Cluster merging decisions
- Ghost-speaker reassignments (speakers < 2s total)

## Technical Details

### Configuration
All tunable parameters are in `config/thresholds.json`:

```json
{
  "diarization": {
    "default_threshold": 0.35,   // clustering cosine distance
    "max_clusters": 15,
    "merge_threshold": 0.25,     // greedy post-merge threshold
    "close_match_threshold": 0.17,
    "min_segment_duration": 0.5
  },
  "matching": {
    "accept_threshold": 0.35,
    "clear_winner_gap": 0.02,
    "embed_only_threshold": 0.16,
    "confidence_threshold": 0.3,
    "embed_only_accept_threshold": 0.22
  },
  "weights": { "embedding": 0.7, "pitch": 0.2, "energy": 0.1 },
  "normalization": { "pitch_hz_per_unit": 50, "energy_rms_per_unit": 0.05, "confidence_max_distance": 0.5 },
  "vad": { "default_threshold": 0.5, "min_speech_duration_ms": 250, "chunk_duration": 30, "overlap": 5 },
  "debug": false
}
```

### Embedding Model
Uses `Wespeaker/wespeaker-ecapa-tdnn512-LM` for speaker embeddings (192-dimensional).

### Diarization Pipeline
1. **VAD** — Silero VAD (ONNX/DirectML with CPU fallback) detects speech segments, chunked in 30s with 5s overlap.
2. **Energy-dip splitting** (`speaker/vad.py:split_at_energy_dips`) — Long segments (>5s) are split at local RMS energy minima (dip_ratio=0.35, min_dip_dur=0.15s) to reduce cross-speaker window contamination.
3. **Sliding-window embedding** (`speaker/audio.py:generate_sliding_windows`) — 2.0s windows, 0.75s stride. CMN applied per window inside `extract_embedding`.
4. **Agglomerative clustering** — Default cosine threshold 0.35, max 15 clusters.
5. **Greedy merge** — Clusters with centroid cosine distance < 0.25 are merged.
6. **Speaker profiling** (`speaker/profiling.py:profile_speakers`) — Autocorrelation pitch + RMS energy per speaker.
7. **Relabelling** (`speaker/profiling.py:relabel_by_pitch`) — SPEAKER1 = lowest pitch, ascending.
8. **Boundary refinement** (`speaker/audio.py:refine_speaker_boundaries`) — All transition sub-windows across the entire file are collected first, then embedded in a single batched ONNX call. Each boundary is then walked to the precise speaker-switch point (0.5s sub-windows, 0.1s stride).
9. **Island absorption** — Isolated segments < 1.0s surrounded by the same speaker are absorbed.
10. **Voiceprint matching** (`speaker/matcher.py`) — Combined distance (emb 0.7 + pitch 0.2 + energy 0.1); clear winner gap 0.02.
11. **Ghost-speaker elimination** — Speakers with < 2s total speech across the whole file are reassigned: stored voiceprint alternatives are tried first; if none qualify, the segment is merged into the nearest temporal neighbour. Adjacent same-speaker segments are then re-collapsed.

### Clustering
- Default threshold: 0.35 (higher = fewer clusters)
- Max clusters capped at 15 to prevent over-segmentation
- Post-clustering greedy merge at 0.25
- Can be overridden with `--num-speakers N`

### Voiceprint Matching
- Accept threshold: 0.35 (combined distance below this = match)
- Clear winner gap: 0.02 (best must beat second-best by this much)
- Embed-only threshold: 0.16 (when emb_dist < this, lower accept threshold 0.22 applies)
- CMN (Cepstral Mean Normalization) applied per window — critical for speaker discrimination

### VAD Acceleration
Silero VAD runs on ONNX with DirectML for iGPU acceleration. Falls back to PyTorch CPU if unavailable.

## Known Issues

### Over-segmentation
If diarization identifies too many speakers:
1. Use `--num-speakers 4` to force exact speaker count
2. Use `--diarization-threshold 0.35` (default is already higher)

### Similar Voices
When multiple voiceprints have very similar embeddings, matching requires a clear gap (0.02) between best and second-best. Otherwise keeps original SPEAKER label.

### Voiceprint Matching
If voices aren't matching:
- Check that voiceprints were created with ecapa-tdnn512 model
- Recreate voiceprints using `mass_refine` if switching models
- Verify video has enough speech for the known speakers

## Creating Voiceprints

Use the `voiceprint_mgmt.py` CLI to add new speakers:

```powershell
# Create new voiceprint from audio segment:
uv run voiceprint_mgmt.py create meeting.mp4 00:05:30 00:06:15 "John"

# Or add samples to existing voiceprint:
uv run voiceprint_mgmt.py add voiceprints.json "John" meeting1.mp4 00:05:30 00:06:15 meeting2.mp4 00:10:00 00:11:00
```

### Segment Extraction and Refinement Workflow

```powershell
# 1. Extract segments for a speaker from diarization output:
uv run voiceprint_mgmt.py extract --audio meeting.mp4 --diarize diarization.json --speaker "SPEAKER_00" --output segments/

# 2. Review segments in the output folder, remove incorrect ones

# 3. Retrain/refine voiceprint with remaining segments:
uv run voiceprint_mgmt.py refine --voiceprints voiceprints.json --speaker "John" --segments segments/
```

When using `--voiceprints` for automatic speaker identification:
- Speakers with less than 10s total detected speech are skipped
- Segments are written as FLAC directly from the decoded in-memory audio (no intermediate WAV)
- Speaker boundaries are refined with a batched ONNX pass before extraction
- Match threshold defaults to 0.3 (cosine distance); lower = stricter matching

### Mass Refine (Batch Processing)

Process multiple speakers from a folder structure:

```powershell
# Folder structure:
# segments/
#   John/
#     audio1.wav
#     audio2.flac
#   Jane/
#     audio1.flac

# Refine all speakers at once:
uv run voiceprint_mgmt.py mass_refine segments/ --voiceprints voiceprints.json

# Skip speakers who already have voiceprints:
uv run voiceprint_mgmt.py mass_refine segments/ --voiceprints voiceprints.json --skip-existing

# Use larger ONNX batches (default 600s = 10 min per call):
uv run voiceprint_mgmt.py mass_refine segments/ --voiceprints voiceprints.json --block-sec 1200
```

`refine` and `mass_refine` accept `.wav`, `.mp3`, and `.flac` files.
All files for a speaker are loaded first, then embedded in a single batched ONNX
pass (`--block-sec` controls block size). ONNX round-trips scale with total audio
duration rather than file count.

### Voiceprint Quality Guidelines

| Quality | Duration | Notes |
|---------|----------|-------|
| Minimum | 5-10s | Works but may have lower accuracy |
| Good | 30-60s | Reliable matching |
| Excellent | 60s+ | Best accuracy with diverse speech |

- Segment must be at least 1.5 seconds
- Clear audio of the target speaker (less noise = better)
- More diverse speech (different words/phrases) = better embedding
- Use multiple short segments to refine existing voiceprints

### Duration-Weighted Averaging

When refining voiceprints, longer segments have proportionally more influence:
- A 60-second segment contributes 10x more than a 6-second segment
- When blending with existing voiceprint, duration determines blend weight

Example: Existing 600s + new 60s = existing contributes 600/660 = 91%, new contributes 60/660 = 9%