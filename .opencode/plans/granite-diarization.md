# Plan: Granite Speech Diarization (Hybrid Approach)

## Goal
Create `granite-diarization/` as a hybrid of `cohere-diarization/` architecture + Granite SAA capabilities. Granite's speaker-attributed ASR (SAA) replaces the VAD → embedding → clustering diarization stages, while ECAPA-TDNN embeddings are retained solely for **voiceprint matching** (identifying known speakers by name).

## Key Decisions
- **Model**: `ibm-granite/granite-speech-4.1-2b-plus` (speaker-attributed ASR + word timestamps)
- **Device**: Auto-detect CUDA/CPU via `torch.cuda.is_available()`
- **Diarization strategy**: **Hybrid** — Granite SAA handles speaker turn detection, ECAPA-TDNN handles voiceprint matching only
- **SAA Prompt**: `"<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."`
- **Voiceprints**: Fully compatible — same ECAPA-TDNN model, can share `voiceprints.json`

## Architecture Comparison

| Stage | Cohere (old) | Granite Hybrid (new) |
|-------|-------------|---------------------|
| Speaker detection | Silero VAD → energy-dip splitting → sliding windows → ECAPA-TDNN → agglomerative clustering | Granite SAA outputs `[Speaker N]:` tags + word timestamps |
| Text generation | Separate ASR pass per segment (ONNX encoder/decoder) | Already done in the SAA pass — no re-inference needed |
| Voiceprint matching | ECAPA-TDNN embeddings matched against known speakers | Same — ECAPA-TDNN embeddings per speaker turn |
| Boundary precision | Sub-second boundary refinement via embedding re-scoring | Word-level timestamps from Granite Plus model |
| Speaker profiling | Pitch/energy/MFCC per speaker | Same — profiling on speaker turns from SAA output |

### New Pipeline Flow
```
Audio (16kHz mono)
  │
  ├─── [Granite SAA] ──→ Text with [Speaker N]: tags + word timestamps
  │
  ├─── [Parse speaker turns] ──→ List of {speaker, start, end, text}
  │
  ├─── [Extract audio per turn] ──→ Per-turn audio segments
  │
  ├─── [ECAPA-TDNN embedding per turn] ──→ 192-dim embeddings
  │
  ├─── [Voiceprint matching] ──→ Map SPEAKER_N → known names
  │
  ├─── [Speaker profiling] ──→ Pitch/energy for consistent labeling
  │
  └─── Return segments: {start, end, speaker, text, confidence}
```

### What's Removed (vs old pipeline)
- ❌ Silero VAD (for diarization — still used in streaming pre-segmentation)
- ❌ Energy-dip splitting
- ❌ Sliding-window embedding extraction (2.0s windows, 1.2s stride)
- ❌ Agglomerative clustering (`sklearn`)
- ❌ Greedy merge clusters
- ❌ Boundary refinement via embedding re-scoring
- ❌ Island absorption (no longer needed — SAA produces clean turns)
- ❌ Ghost-speaker elimination (can still apply as safety net)
- ❌ Confidence scoring from cluster distances

### What's Kept
- ✅ ECAPA-TDNN ONNX embedding (voiceprint matching only)
- ✅ Voiceprint matching (`speaker/matcher.py` — multi-feature distance)
- ✅ Speaker profiling (`speaker/profiling.py` — pitch, energy, MFCC)
- ✅ Relabel by pitch (SPEAKER1 = lowest pitch)
- ✅ Ghost-speaker elimination (safety net for very short turns)
- ✅ Silero VAD (used in streaming for pre-chunking audio into utterances)
- ✅ Client architecture, API package, config, voiceprint management

## Files to Copy Unchanged (13 files + directories)

- `stream_client.py` (audio capture client - 564 lines)
- `voiceprint_mgmt.py` (voiceprint CLI - 860 lines)
- `voiceprint_utils.py` (voiceprint utilities - 1007 lines)
- `config/` (thresholds.json, __init__.py)
- `api/` (schemas.py, security.py, middleware.py, exceptions.py)
- `diarization/clustering.py` (voiceprint matching functions used by pipeline)
- `diarization/segment_ops.py` (ghost elimination, island absorption)
- `speaker/audio.py` (fbank extraction — needed for embedding)
- `speaker/vad.py` (Silero VAD — needed for streaming)
- `speaker/embedding.py` (ECAPA-TDNN ONNX — needed for voiceprints)
- `speaker/matcher.py` (voiceprint multi-feature matching)
- `speaker/profiling.py` (pitch/energy/MFCC profiling)
- `.env-template`, `.gitignore`, `AGENTS.md`

## Files to Modify (8 files) + New (1 file)

### 1. `settings.py` (~80 lines change)
- **Remove** Cohere-specific: `encoder_model_type`, `decoder_model_type`, `encoder_dtype`, `decoder_dtype`, `n_layers`, `heads`, `head_dim`, `max_ctx`
- **Add**: `granite_model_id = "ibm-granite/granite-speech-4.1-2b-plus"`, `torch_dtype: str = "bfloat16"`
- **Add**: `saa_prompt` with the speaker-attribution prompt text
- **Add**: `asr_chunk_duration_sec: int = 60` (for long audio chunking)
- Keep: `max_new_tokens`, all VAD/embedding/server settings
- Keep most diarization threshold settings (used for voiceprint matching)

### 2. `model_loader.py` (~100 lines change)
- **Replace** `ensure_model()` + ONNX session creation with `load_granite_model()`:
  ```python
  from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
  processor = AutoProcessor.from_pretrained(model_id)
  model = AutoModelForSpeechSeq2Seq.from_pretrained(
      model_id, device_map=device, torch_dtype=dtype
  )
  model.eval()
  ```
- **Keep** `ensure_vad_model()` (for streaming), `ensure_embedding_model()` unchanged
- **Keep** `get_igpu_session_options()` for VAD/embedding DirectML support
- Update `load_models()` orchestrator

### 3. `model_state.py` (~70 lines change)
- **Replace** `encoder`/`decoder` ONNX sessions with `asr_processor`/`asr_model` (transformers)
- **Remove**: `tokens`, `token_to_id`, `pre_computed_prompt_ids`, `pre_computed_eos_id`, `pre_computed_prompt_array`
- **Remove**: `KVCachePool` class and `kv_pool` (transformers handles KV cache internally)
- **Keep**: `vad_session`, `embedding_session`, `embedding_cache`

### 4. `transcriber.py` (~350 lines rewrite)
This is the most changed file. New responsibilities:
- **`transcribe_saa_sync(audio, prefix_text=None)`**: Run Granite SAA on audio
  - Uses `processor(saa_prompt, audio)` → `model.generate(return_timestamps=True)`
  - Returns `(text, word_timestamps)`
- **`parse_speaker_turns(text, word_timestamps)`**: Parse `[Speaker N]:` tags + timestamps into structured turns
  - Returns `[{speaker, start, end, text}]`
- **`transcribe_audio_sync(audio)`**: Plain ASR (no speaker tags) — used by `/transcribe/paths`
  - Uses plain prompt `"can you transcribe the speech into a written format?"`
  - Returns `{text, tokens_generated, audio_duration_sec, inference_time_sec}`
- **Keep**: `clean_transcript()` unchanged
- **Keep**: `transcribe_audio_async()` wrapper (updated args)
- **Remove**: mel spectrogram computation, ONNX encoder/decoder, KV cache, Cohere token decoding

### 5. `diarization/pipeline.py` (~300 lines rewrite)
The `Diarizer` class is simplified significantly:
- **`run()`**: New hybrid pipeline:
  1. Load audio via librosa
  2. Call `transcribe_saa_sync(audio)` → text + word timestamps
  3. Call `parse_speaker_turns(text, word_timestamps)` → structured turns
  4. For each turn: extract audio slice → ECAPA-TDNN embedding → L2-normalize
  5. Voiceprint matching via `match_clusters()` from `speaker/matcher.py`
  6. Speaker profiling via `profile_speakers()` from `speaker/profiling.py`
  7. Relabel by pitch
  8. Ghost-speaker elimination (safety net)
  9. Return segments with `{start, end, speaker, text, confidence}`
- **Remove**: `_run_vad()`, `_extract_features()`, `_extract_embeddings()`, `_cluster_embeddings()`, `_assign_labels_to_segments()`, `_map_to_speakers()`, `_refine_boundaries()`
- **Keep**: `_compute_confidence()` (simplified — based on voiceprint match distance)
- **Keep**: `_format_response()`

### 6. `server.py` (~60 lines change)
- **Update** `/diarize/path`:
  - Check `state.asr_model` instead of VAD/embedding for readiness
  - Create `Diarizer(state, settings)` as before
  - Endpoint still streams NDJSON progress → final result
- **Update** `/transcribe/paths`:
  - Replace mel spectrogram preprocessing + ONNX transcriber with Granite ASR
  - Chunk audio at 60s boundaries, pass raw audio to `transcribe_audio_sync()`
- **Update** `/transcribe/upload`: same simplification
- **Remove**: `_compute_mel_spectrogram_fast`, `TARGET_SAMPLES` imports
- **Keep**: health, shutdown, streaming endpoint wiring

### 7. `streaming.py` (~150 lines change)
- **Replace** VAD + embedding + clustering per window with:
  1. Run Granite SAA on the utterance audio
  2. Parse speaker turns from SAA output
  3. Per-turn: embed → match against voiceprints + session speakers
- **Keep**: Session speaker tracking, voiceprint save/load, `_estimate_embedding_from_audio()`
- **Remove**: `run_vad_onnx_direct()`, `split_at_energy_dips()`, `generate_sliding_windows()`, `AgglomerativeClustering` per window

### 8. `transcribe.py` (client — ~30 lines change)
- In `transcribe_file()`: after `diarize_path()`, check if segments have `text` field
  - If yes: use pre-computed text directly, skip `/transcribe/paths` calls
  - If no: fall back to existing chunk-extract-transcribe flow (backward compatible)
- This is the key client optimization — avoids re-running ASR on segments

### 9. `requirements.txt` (~3 lines change)
- **Add**: `transformers>=4.52.1`
- **Keep**: `onnxruntime-directml` (still needed for VAD + ECAPA-TDNN embeddings)
- **Keep**: `torch`, `torchaudio`, `scikit-learn` (used by voiceprint matching)

### 10. `setup_env.bat` (~5 lines change)
- Update messaging (mention ~4GB Granite model)
- Keep CUDA torch install

### 11. `DropToTranscribe.bat` (~10 lines change)
- Copy from cohere-diarization, update server name messages
- Fix: existing granite version references nonexistent `transcribe.py`

## New Files
- None — all changes are modifications of existing files

## Execution Order
1. Copy all unchanged files from `cohere-diarization/` to `granite-diarization/`
2. Modify `settings.py` (model config, remove ONNX ASR settings)
3. Modify `model_state.py` (replace ONNX sessions with transformers objects)
4. Modify `model_loader.py` (load Granite via transformers, keep VAD/embedding)
5. Rewrite `transcriber.py` (Granite SAA + plain ASR + speaker turn parsing)
6. Rewrite `diarization/pipeline.py` (hybrid SAA → voiceprint matching)
7. Modify `server.py` (update endpoints for hybrid flow)
8. Modify `streaming.py` (Granite SAA per utterance)
9. Modify `transcribe.py` (use pre-computed text when available)
10. Update `requirements.txt`, `setup_env.bat`, `DropToTranscribe.bat`
11. Test: `setup_env.bat` → start server → transcribe test audio

## Risks
- Granite SAA quality: speaker turn detection may be less accurate than ECAPA-TDNN clustering for overlapping speech or very similar voices
- Word timestamp accuracy: boundary precision depends on Granite Plus model's timestamp quality
- CUDA memory: ~4GB model + ONNX embeddings (~1GB), needs ~6GB VRAM total
- CPU fallback will be significantly slower (~10-20x)
- `return_timestamps=True` API must be verified — fallback: use VAD for boundaries if timestamps unavailable
- Streaming latency: Granite SAA on short utterances (~3-10s) may be slower than the lightweight ONNX pipeline
- `voiceprint_mgmt.py` and `voiceprint_utils.py` assume ONNX embedding interface — verify compatibility
