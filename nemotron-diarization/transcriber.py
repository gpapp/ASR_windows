"""
ASR transcription inference using Nemotron streaming model.
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
import onnxruntime_genai as og

from settings import get_settings
from model_state import state, executor
from model_loader import LANG_TO_ID
from api.exceptions import TranscriptionError, AudioValidationError, TimeoutError

log = structlog.get_logger()

MEL_FILTERBANK = None

def _get_mel_filterbank():
    global MEL_FILTERBANK
    if MEL_FILTERBANK is None:
        mel_fb = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128)
        MEL_FILTERBANK = mel_fb.astype(np.float32)
    return MEL_FILTERBANK

def _compute_mel_spectrogram_fast(audio: np.ndarray) -> np.ndarray:
    _, _, Sxx = scipy_signal.spectrogram(
        audio, fs=16000, window='hann',
        nperseg=512, noverlap=512 - 160,
        return_onesided=True, mode='magnitude'
    )
    mel_fb = _get_mel_filterbank()
    mel_spec = np.dot(mel_fb, Sxx)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean(axis=1, keepdims=True)) / (
        mel_spec_db.std(axis=1, keepdims=True) + 1e-8)
    return mel_spec_db


@contextmanager
def inference_timeout(seconds: int):
    if sys.platform == "win32":
        yield
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


_LOOP_RE = re.compile(r'(.{4,120}?)(?:\s+\1){2,}', re.IGNORECASE)

def _trim_partial_prefix(before: str, unit: str) -> str:
    words = unit.lower().split()
    b = before.rstrip()
    b_lower = b.lower()
    for start in range(len(words)):
        suffix = " ".join(words[start:])
        if b_lower.endswith(suffix):
            return b[: len(b) - len(suffix)].rstrip()
    return b

def _trim_partial_suffix(after: str, unit: str) -> str:
    words = unit.lower().split()
    a = after.lstrip()
    a_lower = a.lower()
    for end in range(len(words), 0, -1):
        prefix = " ".join(words[:end])
        if a_lower.startswith(prefix):
            return a[len(prefix):].lstrip()
    return a

def clean_transcript(text: str) -> str:
    text = re.sub(r'\b(\w{3,})\s+\1\b', r'\1', text)
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
    text = re.sub(r'(\[inaudible\]\s*){2,}', '[inaudible] ', text)
    text = re.sub(
        r'\[inaudible\]\s+(?:\w[\w\s,\']{0,80}?)\s+\[inaudible\]',
        '[inaudible]',
        text,
    )
    return text.strip()


def _get_lang_id(language: str) -> int:
    """Map language code to Nemotron lang_id. Defaults to 0 (en-US)."""
    lang = language or "en"
    return LANG_TO_ID.get(lang, 0)


def transcribe_streaming(
    audio: np.ndarray,
    language: str = "en",
) -> dict:
    """
    Transcribe audio using Nemotron streaming API.
    Returns full text + per-chunk timestamp info.

    Args:
        audio: Raw PCM float32 audio at 16000Hz
        language: Language code

    Returns:
        dict with "text", "audio_duration_sec", "inference_time_sec",
               "chunk_texts" - list of {"start", "end", "text"} per streaming chunk
    """
    settings = get_settings()
    start_time = time.perf_counter()

    if not state.is_ready:
        raise TranscriptionError("Model not ready")

    sample_rate = settings.nemotron_sample_rate
    chunk_samples = settings.nemotron_chunk_samples
    audio_duration = len(audio) / sample_rate

    if audio_duration > settings.max_audio_duration_sec:
        raise AudioValidationError(
            f"Audio too long: {audio_duration:.1f}s > {settings.max_audio_duration_sec}s max"
        )

    processor = state.processor
    tokenizer = state.tokenizer

    lang_id = _get_lang_id(language)

    tokenizer_stream = tokenizer.create_stream()
    params = og.GeneratorParams(state.model)
    generator = og.Generator(state.model, params)
    generator.set_runtime_option("lang_id", str(lang_id))

    chunk_texts = []
    chunk_start_times = []
    chunk_count = 0

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples].astype(np.float32)
        inputs = processor.process(chunk)
        if inputs is not None:
            generator.set_inputs(inputs)
            chunk_text = ""
            while not generator.is_done():
                generator.generate_next_token()
                tokens = generator.get_next_tokens()
                if len(tokens) > 0:
                    token_text = tokenizer_stream.decode(tokens[0])
                    if token_text:
                        chunk_text += token_text
            if chunk_text.strip():
                text = clean_transcript(chunk_text.strip())
                if text:
                    chunk_start = i / sample_rate
                    chunk_end = min((i + chunk_samples) / sample_rate, audio_duration)
                    chunk_texts.append({
                        "start": round(chunk_start, 3),
                        "end": round(chunk_end, 3),
                        "text": text
                    })

    inputs = processor.flush()
    if inputs is not None:
        generator.set_inputs(inputs)
        flush_text = ""
        while not generator.is_done():
            generator.generate_next_token()
            tokens = generator.get_next_tokens()
            if len(tokens) > 0:
                token_text = tokenizer_stream.decode(tokens[0])
                if token_text:
                    flush_text += token_text
        if flush_text.strip():
            text = clean_transcript(flush_text.strip())
            if text:
                chunk_texts.append({
                    "start": round(audio_duration - 0.05, 3) if chunk_texts else 0.0,
                    "end": round(audio_duration, 3),
                    "text": text
                })

    full_text = " ".join(c["text"] for c in chunk_texts)
    inference_time = time.perf_counter() - start_time

    log.info(
        "transcription_complete",
        audio_duration=f"{audio_duration:.2f}s",
        inference_time=f"{inference_time:.2f}s",
        chunks=len(chunk_texts),
        text_preview=full_text[:100],
    )

    return {
        "text": full_text,
        "chunk_texts": chunk_texts,
        "audio_duration_sec": audio_duration,
        "inference_time_sec": inference_time,
    }


def transcribe_segment(
    audio: np.ndarray,
    language: str = "en",
) -> dict:
    """
    Transcribe a single audio segment (typically one diarization segment).
    Creates fresh generator and processor flow per segment.
    """
    settings = get_settings()
    start_time = time.perf_counter()

    if not state.is_ready:
        raise TranscriptionError("Model not ready")

    sample_rate = settings.nemotron_sample_rate
    chunk_samples = settings.nemotron_chunk_samples
    audio_duration = len(audio) / sample_rate

    processor = state.processor
    tokenizer = state.tokenizer
    lang_id = _get_lang_id(language)

    tokenizer_stream = tokenizer.create_stream()
    params = og.GeneratorParams(state.model)
    generator = og.Generator(state.model, params)
    generator.set_runtime_option("lang_id", str(lang_id))

    full_text = ""

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples].astype(np.float32)
        inputs = processor.process(chunk)
        if inputs is not None:
            generator.set_inputs(inputs)
            while not generator.is_done():
                generator.generate_next_token()
                tokens = generator.get_next_tokens()
                if len(tokens) > 0:
                    token_text = tokenizer_stream.decode(tokens[0])
                    if token_text:
                        full_text += token_text

    inputs = processor.flush()
    if inputs is not None:
        generator.set_inputs(inputs)
        while not generator.is_done():
            generator.generate_next_token()
            tokens = generator.get_next_tokens()
            if len(tokens) > 0:
                token_text = tokenizer_stream.decode(tokens[0])
                if token_text:
                    full_text += token_text

    full_text = clean_transcript(full_text.strip())
    inference_time = time.perf_counter() - start_time

    return {
        "text": full_text,
        "audio_duration_sec": audio_duration,
        "inference_time_sec": inference_time,
    }


async def transcribe_audio_async(
    audio: Optional[np.ndarray] = None,
    language: str = "en",
    timeout_sec: int = 120,
    mel_spectrogram: Optional[np.ndarray] = None,
    use_chunked: bool = True,
) -> dict:
    """
    Async wrapper for Nemotron streaming transcription.

    For full-file transcription (use_chunked=True), uses transcribe_streaming.
    For per-segment transcription (use_chunked=False), uses transcribe_segment.

    mel_spectrogram is kept for API compatibility but not used by Nemotron.
    """
    loop = asyncio.get_event_loop()

    if use_chunked:
        return await loop.run_in_executor(
            executor, transcribe_streaming, audio, language
        )
    else:
        return await loop.run_in_executor(
            executor, transcribe_segment, audio, language
        )
