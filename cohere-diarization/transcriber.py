"""
ASR transcription inference and transcript cleaning.
"""

import sys
import time
import signal
import re
from contextlib import contextmanager

import numpy as np
import librosa
import asyncio
import structlog

from settings import get_settings
from model_state import state, executor
from api.exceptions import TranscriptionError, AudioValidationError, TimeoutError

log = structlog.get_logger()

# Target samples for chunking (30 seconds at 16kHz)
TARGET_SAMPLES = 480000


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
    audio: np.ndarray, 
    language: str = "en",
    timeout_sec: int = 120
) -> dict:
    """
    Synchronous ASR transcription using Cohere Transcribe ONNX model.
    
    Args:
        audio: Raw PCM audio waveform (16kHz, mono), shape [T]
        language: ISO language code (default: "en")
        timeout_sec: Inference timeout in seconds
    
    Returns:
        Dict with transcribed text, tokens generated, timing, and audio duration.
    """
    settings = get_settings()
    start_time = time.perf_counter()
    
    if not state.is_ready:
        raise TranscriptionError("Model not ready")
    
    # Validate audio
    audio_duration = len(audio) / 16000
    if audio_duration > settings.max_audio_duration_sec:
        raise AudioValidationError(
            f"Audio too long: {audio_duration:.1f}s > {settings.max_audio_duration_sec}s max"
        )
    
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
        
        # Normalize audio to [-1, 1]
        audio_float = audio.astype(np.float32) / 32768.0
        
        # Compute mel-spectrogram using librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio_float,
            sr=16000,
            n_fft=512,
            hop_length=160,  # 10ms stride
            n_mels=128,
            window='hann'
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize per-feature (per-frequency bin)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean(axis=1, keepdims=True)) / (mel_spec_db.std(axis=1, keepdims=True) + 1e-8)
        
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
            enc_outputs = encoder.run(None, enc_inputs)
            raw_encoder_hidden_state = enc_outputs[0]  # Shape: [1, T', 1024]
            log.debug("encoder_inference_complete", output_shape=str(raw_encoder_hidden_state.shape))
        except Exception as e:
            log.error("encoder_inference_failed", error=str(e))
            raise TranscriptionError(f"Encoder inference failed: {str(e)}")
        
        if settings.encoder_dtype!=settings.decoder_dtype:
            encoder_hidden_state = raw_encoder_hidden_state.astype(settings.decoder_dtype)
        else:
            encoder_hidden_state = raw_encoder_hidden_state
        
        # ====================================================================
        # Step 3: Prepare decoder inputs with prompt
        # ====================================================================
        # Pre-computed prompt tokens (from state)
        prompt_ids = state.pre_computed_prompt_ids
        eos_id = state.pre_computed_eos_id
        
        # Initialize with batch size 1
        batch_size = 1
        
        generated_ids = list(prompt_ids)
        
        # ====================================================================
        # Step 4: Autoregressive generation with KV cache
        # ====================================================================
        max_new_tokens = settings.max_new_tokens
        
        # Initialize KV cache (empty at start)
        # Shape per layer: [batch_size=1, num_heads=8, seq_len=0, head_dim=128]
        num_layers = 8
        num_heads = 8
        head_dim = 128
        
        past_kv_cache = {}
        for layer_idx in range(num_layers):
            past_kv_cache[f"past_key_values.{layer_idx}.decoder.key"] = np.zeros(
                (batch_size, num_heads, 0, head_dim), dtype=settings.decoder_dtype
            )
            past_kv_cache[f"past_key_values.{layer_idx}.decoder.value"] = np.zeros(
                (batch_size, num_heads, 0, head_dim), dtype=settings.decoder_dtype
            )
            past_kv_cache[f"past_key_values.{layer_idx}.encoder.key"] = np.zeros(
                (batch_size, num_heads, 0, head_dim), dtype=settings.decoder_dtype
            )
            past_kv_cache[f"past_key_values.{layer_idx}.encoder.value"] = np.zeros(
                (batch_size, num_heads, 0, head_dim), dtype=settings.decoder_dtype
            )
        
        for step in range(max_new_tokens):
            # Get the last token ID(s) to feed
            if step == 0:
                # First step: use full prompt
                current_ids = np.array([prompt_ids], dtype=np.int64)
            else:
                # Subsequent steps: use just the last generated token
                current_ids = np.array([[generated_ids[-1]]], dtype=np.int64)
            
            # Build attention masks (all 1s - attend to everything)
            seq_length = current_ids.shape[1]
            total_seq_len = len(generated_ids)  # All tokens generated so far
            attention_mask = np.ones((batch_size, total_seq_len), dtype=np.int64)
            
            # Position IDs
            if step == 0:
                position_ids = np.arange(seq_length, dtype=np.int64).reshape(batch_size, -1)
            else:
                # Only position for the last token
                position_ids = np.array([[total_seq_len - 1]], dtype=np.int64)
            
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
            
            # Add KV cache inputs
            decoder_inputs.update(past_kv_cache)
            
            try:
                dec_outputs = decoder.run(None, decoder_inputs)
                logits = dec_outputs[0]  # Shape: [batch, num_logits_to_keep, vocab_size]
                
                # Extract and update KV cache from outputs
                # Outputs: [logits, present.0.decoder.key, present.0.decoder.value, ...]
                for layer_idx in range(num_layers):
                    past_kv_cache[f"past_key_values.{layer_idx}.decoder.key"] = dec_outputs[1 + layer_idx * 4]
                    past_kv_cache[f"past_key_values.{layer_idx}.decoder.value"] = dec_outputs[2 + layer_idx * 4]
                    past_kv_cache[f"past_key_values.{layer_idx}.encoder.key"] = dec_outputs[3 + layer_idx * 4]
                    past_kv_cache[f"past_key_values.{layer_idx}.encoder.value"] = dec_outputs[4 + layer_idx * 4]
                    
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
                log.debug("generation_progress", step=step, tokens_generated=len(generated_ids) - len(prompt_ids))
        
        # ====================================================================
        # Step 5: Decode tokens to text
        # ====================================================================
        # Skip prompt tokens when decoding
        generated_tokens = generated_ids[len(prompt_ids):]
        
        # Convert token IDs to text
        text_parts = []
        for token_id in generated_tokens:
            token_str = tokens_dict.get(token_id, "")
            if token_str.startswith("<|"):
                # Skip special tokens
                continue
            # Replace sentence piece marker with space
            token_str = token_str.replace("▁", " ")
            text_parts.append(token_str)
        
        text = "".join(text_parts).strip()
        
        # Clean up repetition artifacts (hallucinations)
        text = clean_transcript(text)
        
        tokens_generated = len(generated_ids) - len(prompt_ids)
        inference_time = time.perf_counter() - start_time
        
        log.info(
            "transcription_complete",
            audio_duration=f"{audio_duration:.2f}s",
            tokens=tokens_generated,
            inference_time=f"{inference_time:.2f}s",
            text_preview=text[:100]
        )
        
        return {
            "text": text,
            "tokens_generated": tokens_generated,
            "audio_duration_sec": audio_duration,
            "inference_time_sec": inference_time,
        }


async def transcribe_audio_async(
    audio: np.ndarray, 
    language: str = "en",
    timeout_sec: int = 120
) -> dict:
    """Async wrapper for transcription."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, 
        transcribe_audio_sync, 
        audio, 
        language,
        timeout_sec
    )
