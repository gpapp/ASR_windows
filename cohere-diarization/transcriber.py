"""
ASR transcription inference and transcript cleaning.
"""

import sys
import gc
import time
import signal
import re
from contextlib import contextmanager
from typing import Optional

import numpy as np
import librosa
import asyncio
import structlog
from scipy import signal as scipy_signal
import onnxruntime as ort

from settings import get_settings
from model_state import state, executor
from model_loader import reload_encoder_session, reload_embedding_session
from api.exceptions import TranscriptionError, AudioValidationError, TimeoutError

log = structlog.get_logger()

# Target samples for chunking (60 seconds at 16kHz)
TARGET_SAMPLES = 960000

# Pre-compute mel filterbank once (reuse across all calls)
_MEL_FILTERBANK = None

def _get_mel_filterbank():
    """Lazily compute and cache mel filterbank for fast reuse."""
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        # Create mel filterbank using librosa (one-time cost)
        mel_fb = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128)
        _MEL_FILTERBANK = mel_fb.astype(np.float32)
    return _MEL_FILTERBANK

def _compute_mel_spectrogram_fast(audio: np.ndarray) -> np.ndarray:
    """
    Fast mel-spectrogram computation using pre-computed filterbank.
    ~5-10x faster than librosa.feature.melspectrogram().
    
    Args:
        audio: Raw PCM audio waveform (float32, 16kHz, mono)
    
    Returns:
        Mel-spectrogram with shape [n_mels, seq_len] in dB scale, normalized
    """
    # STFT using scipy (faster than librosa for single call)
    _, _, Sxx = scipy_signal.spectrogram(
        audio,
        fs=16000,
        window='hann',
        nperseg=512,
        noverlap=512 - 160,  # hop_length = 160
        return_onesided=True,
        mode='magnitude'
    )
    
    # Apply mel filterbank: [n_mels, n_freqs] @ [n_freqs, n_frames] = [n_mels, n_frames]
    mel_fb = _get_mel_filterbank()
    mel_spec = np.dot(mel_fb, Sxx)
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize per-feature
    mel_spec_db = (mel_spec_db - mel_spec_db.mean(axis=1, keepdims=True)) / (mel_spec_db.std(axis=1, keepdims=True) + 1e-8)
    
    return mel_spec_db


# ============================================================================
# Inference Timeout
# ============================================================================

@contextmanager
def inference_timeout(seconds: int):
    """Context manager for inference timeout (Unix only)."""
    if sys.platform == "win32":
        yield  # Windows doesn't support SIGALRM
        return
    
    def handler(signum, frame):
        raise TimeoutError(f"Inference timed out after {seconds}s")
    
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================================
# Transcript Hallucination Cleaner
# ============================================================================
# Whisper-family models produce looping repetitions when fed laughter, noise,
# or silence: e.g. "The police are the ones who are the ones who are ..."
# repeated hundreds of times.
#
# The tricky part: the text before the detected repeat is often a *partial*
# instance of the same phrase (the loop started mid-sentence), and the text
# after may be a partial trailing instance.  We trim both.
#
# Example:
#   "Right. The police are the ones who are the ones who are the ones."
#   repeated unit  = "are the ones who"
#   prefix before  = "Right. The police "  <- ends with suffix of unit
#   suffix after   = " are the ones."      <- starts with prefix of unit
#   result         = "Right. [inaudible]."

_LOOP_RE = re.compile(
    r'(.{4,120}?)(?:\s+\1){2,}',   # phrase repeated ≥3 times total
    re.IGNORECASE,
)

def _trim_partial_prefix(before: str, unit: str) -> str:
    """Remove a trailing suffix-of-unit bleed from `before`."""
    words = unit.lower().split()
    b = before.rstrip()
    b_lower = b.lower()
    for start in range(len(words)):           # try longest suffix first
        suffix = " ".join(words[start:])
        if b_lower.endswith(suffix):
            return b[: len(b) - len(suffix)].rstrip()
    return b

def _trim_partial_suffix(after: str, unit: str) -> str:
    """Remove a leading prefix-of-unit bleed from `after`."""
    words = unit.lower().split()
    a = after.lstrip()
    a_lower = a.lower()
    for end in range(len(words), 0, -1):      # try longest prefix first
        prefix = " ".join(words[:end])
        if a_lower.startswith(prefix):
            return a[len(prefix):].lstrip()
    return a

def clean_transcript(text: str) -> str:
    """Replace hallucinated looping repetitions with [inaudible]."""
    # First pass: collapse adjacent 2× word-level stutters (e.g. "Hello Hello",
    # "usual usual").  This catches short repeats that _LOOP_RE misses.
    text = re.sub(r'\b(\w{3,})\s+\1\b', r'\1', text)
    # Also handle 2-char repeats like "Ah Ah"
    text = re.sub(r'\b(\w{2})\s+\1\b', r'\1', text)
    prev = None
    while prev != text:
        prev = text
        m = _LOOP_RE.search(text)
        if not m:
            break
        unit   = m.group(1)
        before = _trim_partial_prefix(text[:m.start()], unit)
        after  = _trim_partial_suffix(text[m.end():], unit)
        parts  = [p for p in (before, '[inaudible]', after) if p]
        text   = ' '.join(parts)
    # Collapse consecutive tags
    text = re.sub(r'(\[inaudible\]\s*){2,}', '[inaudible] ', text)
    # Drop short fragments sandwiched between [inaudible] tags — these are
    # mutation-zone remnants where the loop phrase slowly morphed and the
    # exact-repeat regex could not catch the transitional words (≤12 words).
    text = re.sub(
        r'\[inaudible\]\s+(?:\w[\w\s,\']{0,80}?)\s+\[inaudible\]',
        '[inaudible]',
        text,
    )
    return text.strip()


# ============================================================================
# ASR Inference
# ============================================================================

def transcribe_audio_sync(
    audio: Optional[np.ndarray] = None,
    language: str = "en",
    timeout_sec: int = 120,
    mel_spectrogram: Optional[np.ndarray] = None,
    past_kv_cache_ort: Optional[dict] = None,
    prefix_ids: Optional[list[int]] = None
) -> dict:
    """
    Synchronous ASR transcription using Cohere Transcribe ONNX model.
    
    Args:
        audio: Raw PCM audio waveform (16kHz, mono), shape [T]
        language: ISO language code (default: "en")
        timeout_sec: Inference timeout in seconds
        mel_spectrogram: Precomputed mel-spectrogram features with shape [1, seq_len, 128]
                         If provided, audio parameter is ignored
        past_kv_cache_ort: Previous KV cache for incremental decoding
        prefix_ids: Last few token IDs from the previous segment to use as a prefix

    Returns:
        Dict with transcribed text, tokens generated, timing, and audio duration.
    """
    settings = get_settings()
    start_time = time.perf_counter()
    stage_timings = {}
    
    if not state.is_ready:
        raise TranscriptionError("Model not ready")
    
    with inference_timeout(timeout_sec):
        encoder = state.encoder
        decoder = state.decoder
        tokens_dict = state.tokens
        token_to_id = state.token_to_id
        
        # ====================================================================
        # Step 1: Extract mel-spectrogram features from raw waveform
        # ====================================================================
        # Cohere uses mel-spec with: sample_rate=16kHz, n_fft=512, 
        # hop_length=160 (10ms), n_mels=128
        
        if mel_spectrogram is not None:
            # Use precomputed mel-spectrogram
            input_features = mel_spectrogram.astype(np.float32)
            # Calculate duration from mel spectrogram shape
            # Shape is [1, seq_len, 128], hop_length=160, sr=16000
            # Each frame = 160 samples = 0.01 seconds
            seq_len = input_features.shape[1]
            audio_duration = seq_len * 0.01
            stage_timings["mel_extraction_sec"] = 0.0
        else:
            # Validate audio
            if audio is None:
                raise ValueError("Either audio or mel_spectrogram must be provided")
                
            audio_duration = len(audio) / 16000
            if audio_duration > settings.max_audio_duration_sec:
                raise AudioValidationError(
                    f"Audio too long: {audio_duration:.1f}s > {settings.max_audio_duration_sec}s max"
                )
            
            # Apply pre-emphasis (matches CohereAsrFeatureExtractor)
            audio_float = audio.astype(np.float32)
            
            # Step 1.1: Volume Normalization (Improved to reduce hallucinations)
            rms = np.sqrt(np.mean(audio_float**2))
            if rms > 0.005:  # Noise floor check
                audio_float = audio_float * (0.06 / rms)

            audio_float[1:] -= 0.97 * audio_float[:-1]
            
            # Fast mel-spectrogram computation (5-10x faster than librosa)
            mel_start = time.perf_counter()
            mel_spec_db = _compute_mel_spectrogram_fast(audio_float)
            stage_timings["mel_extraction_sec"] = time.perf_counter() - mel_start
            
            # Transpose to [sequence_length, n_mels] and add batch dimension
            # Shape: [1, seq_len, 128]
            input_features = mel_spec_db.T[np.newaxis, :, :].astype(np.float32)
        
        log.debug(
            "audio_features_extracted",
            shape=str(input_features.shape),
            duration_sec=audio_duration
        )
        
        # ====================================================================
        # Step 2: Run encoder to get context vectors
        # ====================================================================
        enc_inputs = {
            "input_features": input_features
        }
        
        try:
            enc_start = time.perf_counter()
            enc_outputs = encoder.run(None, enc_inputs)
            stage_timings["encoder_sec"] = time.perf_counter() - enc_start
            raw_encoder_hidden_state = enc_outputs[0]  # Shape: [1, T', 1024]
            log.debug("encoder_inference_complete", output_shape=str(raw_encoder_hidden_state.shape))
        except Exception as e:
            log.warning("encoder_inference_failed_reloading", error=str(e))
            reload_embedding_session(settings, force_cpu=True)
            reload_encoder_session(settings, force_cpu=True)
            encoder = state.encoder
            try:
                enc_start = time.perf_counter()
                enc_outputs = encoder.run(None, enc_inputs)
                stage_timings["encoder_sec"] = time.perf_counter() - enc_start
                raw_encoder_hidden_state = enc_outputs[0]
                log.info("encoder_inference_succeeded_after_reload")
            except Exception as e2:
                log.error("encoder_inference_failed_after_reload", error=str(e2))
                raise TranscriptionError(f"Encoder inference failed: {str(e2)}")
        
        # Both _q4 and _fp16 models expect encoder_hidden_states as float32
        encoder_hidden_state = raw_encoder_hidden_state.astype(np.float32)
        
        # Free encoder output to release GPU memory held by DirectML
        enc_outputs = None
        raw_encoder_hidden_state = None
        gc.collect()
        
        # ====================================================================
        # Step 3: Prepare decoder inputs with prompt
        # ====================================================================
        # Use the provided language to select the correct prompt tokens
        if language == "en":
            prompt_ids = state.pre_computed_prompt_ids
        else:
            lang_token = f"<|{language}|>"
            prompt_tokens = [
                "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
                lang_token, lang_token, "<|pnc|>", "<|noitn|>", "<|timestamp|>", "<|nodiarize|>",
            ]
            prompt_ids = [token_to_id[t] for t in prompt_tokens if t in token_to_id]
        eos_id = state.pre_computed_eos_id
        
        # Initialize with batch size 1
        batch_size = 1
        
        # ====================================================================
        # Step 4: Autoregressive generation with KV cache
        # ====================================================================
        max_new_tokens = settings.max_new_tokens
        
        # Initialize KV cache (empty at start)
        # Shape per layer: [batch_size=1, num_heads=8, seq_len=0, head_dim=128]
        num_layers = 8
        num_heads = 8
        head_dim = 128
        
        # Determine current sequence length in the cache
        past_seq_len = 0
        if past_kv_cache_ort is not None:
            # Extract length from the first decoder key in the cache
            # OrtValue.shape() returns a list: [batch, heads, seq_len, head_dim]
            first_key_name = f"past_key_values.0.decoder.key"
            if first_key_name in past_kv_cache_ort:
                past_seq_len = past_kv_cache_ort[first_key_name].shape()[2]

        if past_kv_cache_ort is None:
            # Initialize new KV cache with OrtValue objects
            new_kv_cache = {}
            for layer_idx in range(num_layers):
                new_kv_cache[f"past_key_values.{layer_idx}.decoder.key"] = ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((batch_size, num_heads, 0, head_dim), dtype=settings.decoder_dtype), "cpu", 0
                )
                new_kv_cache[f"past_key_values.{layer_idx}.decoder.value"] = ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((batch_size, num_heads, 0, head_dim), dtype=settings.decoder_dtype), "cpu", 0
                )
                new_kv_cache[f"past_key_values.{layer_idx}.encoder.key"] = ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((batch_size, num_heads, 0, head_dim), dtype=settings.decoder_dtype), "cpu", 0
                )
                new_kv_cache[f"past_key_values.{layer_idx}.encoder.value"] = ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((batch_size, num_heads, 0, head_dim), dtype=settings.decoder_dtype), "cpu", 0
                )
            past_kv_cache_ort = new_kv_cache
            past_seq_len = 0

        else:
            # IMPORTANT: We must reset the encoder KV cache entries every call 
            # because the encoder_hidden_states are fresh for this specific audio chunk.
            for layer_idx in range(num_layers):
                past_kv_cache_ort[f"past_key_values.{layer_idx}.encoder.key"] = ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((batch_size, num_heads, 0, head_dim), dtype=settings.decoder_dtype), "cpu", 0
                )
                past_kv_cache_ort[f"past_key_values.{layer_idx}.encoder.value"] = ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((batch_size, num_heads, 0, head_dim), dtype=settings.decoder_dtype), "cpu", 0
                )

        # generated_ids tracks all tokens for output stripping (prompt + prefix
        # context + newly generated).
        generated_ids = list(prompt_ids)
        if prefix_ids:
            generated_ids.extend(prefix_ids[-20:])
        num_context = len(generated_ids)

        decoder_start = time.perf_counter()
        tokens_this_call = 0  # input tokens processed in this call so far (excl. cached)
        for step in range(max_new_tokens):
            # Get the last token ID(s) to feed
            if step == 0:
                if past_kv_cache_ort is not None and prefix_ids:
                    # Bridge: feed just the last prefix token — the KV cache
                    # already holds the rest of the previous utterance's context.
                    current_ids = np.array([[prefix_ids[-1]]], dtype=np.int64)
                else:
                    current_ids = np.array([prompt_ids], dtype=np.int64)
            else:
                current_ids = np.array([[generated_ids[-1]]], dtype=np.int64)
            
            # Build attention masks (all 1s - attend to everything)
            seq_length = current_ids.shape[1]
            total_seq_len = past_seq_len + tokens_this_call + seq_length
            attention_mask = np.ones((batch_size, total_seq_len), dtype=np.int64)
            
            # Position IDs
            if step == 0:
                position_ids = (np.arange(seq_length, dtype=np.int64) + past_seq_len).reshape(batch_size, -1)
            else:
                position_ids = np.array([[total_seq_len - 1]], dtype=np.int64)
            
            tokens_this_call += seq_length
            
            # num_logits_to_keep: how many output logits to keep (1 for greedy decoding)
            num_logits_to_keep = np.array(1, dtype=np.int64)
            
            # Prepare decoder inputs
            decoder_inputs = {
                "input_ids": current_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "num_logits_to_keep": num_logits_to_keep,
                "encoder_hidden_states": encoder_hidden_state,
            }
            
            # Add KV cache inputs (OrtValue objects)
            decoder_inputs.update(past_kv_cache_ort)
            
            try:
                dec_outputs = decoder.run(None, decoder_inputs)
                logits = dec_outputs[0]  # Shape: [batch, num_logits_to_keep, vocab_size]
                
                # Extract and update KV cache from outputs
                for layer_idx in range(num_layers):
                    # Update past_kv_cache_ort with new OrtValue objects
                    past_kv_cache_ort[f"past_key_values.{layer_idx}.decoder.key"] = dec_outputs[1 + layer_idx * 4]
                    past_kv_cache_ort[f"past_key_values.{layer_idx}.decoder.value"] = dec_outputs[2 + layer_idx * 4]
                    past_kv_cache_ort[f"past_key_values.{layer_idx}.encoder.key"] = dec_outputs[3 + layer_idx * 4]
                    past_kv_cache_ort[f"past_key_values.{layer_idx}.encoder.value"] = dec_outputs[4 + layer_idx * 4]
                    
            except Exception as e:
                log.error("decoder_inference_failed", step=step, error=str(e))
                raise TranscriptionError(f"Decoder inference failed at step {step}: {str(e)}")
            
            # Get the last token logits and argmax for greedy decoding
            # logits shape: [1, 1, 16384]
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            next_token_id = int(np.argmax(next_token_logits))
            
            # Check for EOS
            if next_token_id == eos_id:
                log.debug("generation_finished_eos", step=step)
                break
            
            generated_ids.append(next_token_id)
            
            if (step + 1) % 10 == 0:
                log.debug("generation_progress", step=step, tokens_generated=len(generated_ids) - num_context)
        
        decoder_time = time.perf_counter() - decoder_start
        stage_timings["decoder_sec"] = decoder_time
        
        # ====================================================================
        # Step 5: Decode tokens to timed segments
        # ====================================================================
        # Skip context tokens when decoding (prompt or prefix context)
        generated_tokens = generated_ids[num_context:]
        
        # Split token constants
        SPLIT_TOKEN_BASE = token_to_id.get("<|spltoken0|>", -1)
        NUM_SPLIT_BINS = 34
        
        # Decode tokens into text parts and track split-token boundaries
        segments = []
        seg_text_parts = []
        current_seg_start = 0.0
        
        for token_id in generated_tokens:
            token_str = tokens_dict.get(token_id, "")
            if token_str.startswith("<|"):
                if SPLIT_TOKEN_BASE != -1 and SPLIT_TOKEN_BASE <= token_id < SPLIT_TOKEN_BASE + NUM_SPLIT_BINS:
                    # Flush current text as a segment ending at this split point
                    seg_text = "".join(seg_text_parts).strip()
                    if seg_text:
                        seg_end = audio_duration * (token_id - SPLIT_TOKEN_BASE) / NUM_SPLIT_BINS
                        segments.append({
                            "start": round(current_seg_start, 3),
                            "end": round(seg_end, 3),
                            "text": seg_text
                        })
                        current_seg_start = seg_end
                    seg_text_parts = []
                continue
            token_str = token_str.replace("▁", " ")
            seg_text_parts.append(token_str)
        
        # Flush remaining text after the last split token
        seg_text = "".join(seg_text_parts).strip()
        if seg_text:
            segments.append({
                "start": round(current_seg_start, 3),
                "end": round(audio_duration, 3),
                "text": seg_text
            })
        
        # Clean each segment text individually, then build flat text
        for seg in segments:
            seg["text"] = clean_transcript(seg["text"])
        text_parts = [s["text"] for s in segments]
        text = " ".join(text_parts).strip()
        
        tokens_generated = len(generated_ids) - num_context
        stage_timings["decoder_avg_token_sec"] = stage_timings["decoder_sec"] / max(1, tokens_generated)
        inference_time = time.perf_counter() - start_time
        
        log.info(
            "transcription_complete",
            audio_duration=f"{audio_duration:.2f}s",
            tokens=tokens_generated,
            inference_time=f"{inference_time:.2f}s",
            text_preview=text[:100],
            stage_timings=stage_timings
        )
        
        # Clear large encoder memory before returning
        input_features = None
        encoder_hidden_state = None
        decoder_inputs = None
        logits = None
        
        return {
            "text": text,
            "segments": segments,
            "tokens_generated": tokens_generated,
            "audio_duration_sec": audio_duration,
            "inference_time_sec": inference_time,
            "stage_timings": stage_timings,
            "past_kv_cache_ort": past_kv_cache_ort, # Return the updated OrtValue KV cache
            "last_token_ids": generated_ids[num_context:] # Return IDs for future prefixing
        }


async def transcribe_audio_async(
    audio: Optional[np.ndarray] = None,
    language: str = "en",
    timeout_sec: int = 120, # This is the request timeout, not inference timeout
    mel_spectrogram: Optional[np.ndarray] = None,
    past_kv_cache_ort: Optional[dict] = None,
    prefix_ids: Optional[list[int]] = None
) -> dict:
    """Async wrapper for transcription."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, 
        transcribe_audio_sync, 
        audio, 
        language,
        timeout_sec,
        mel_spectrogram,
        past_kv_cache_ort,
        prefix_ids
    )
