"""
Granite ASR transcription: SAA (speaker-attributed) and plain ASR modes.
ONNX int8 path when available, fallback to PyTorch.
"""

import sys
import time
import signal
import re
import asyncio
from contextlib import contextmanager
from typing import Optional

import numpy as np
import structlog

from settings import get_settings
from model_state import state, executor
from api.exceptions import TranscriptionError, AudioValidationError, TimeoutError

log = structlog.get_logger()

PLAIN_ASR_PROMPT = (
    "<|audio|> can you transcribe the speech into a written format?"
)

_LOOP_RE = re.compile(
    r'(.{4,120}?)(?:\s+\1){2,}',
    re.IGNORECASE,
)

_SAA_TAG_RE = re.compile(r'\[Speaker (\d+)\]\s*:\s*')


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
    """Replace hallucinated looping repetitions with [inaudible]."""
    if not text:
        return text
    text = re.sub(r'\b(\w{3,})\s+\1\b', r'\1', text)
    text = re.sub(r'\b(\w{2})\s+\1\b', r'\1', text)
    prev = None
    while prev != text:
        prev = text
        m = _LOOP_RE.search(text)
        if not m:
            break
        unit = m.group(1)
        before = _trim_partial_prefix(text[:m.start()], unit)
        after = _trim_partial_suffix(text[m.end():], unit)
        parts = [p for p in (before, '[inaudible]', after) if p]
        text = ' '.join(parts)
    text = re.sub(r'(\[inaudible\]\s*){2,}', '[inaudible] ', text)
    text = re.sub(
        r'\[inaudible\]\s+(?:\w[\w\s,\']{0,80}?)\s+\[inaudible\]',
        '[inaudible]',
        text,
    )
    return text.strip()


def parse_speaker_turns(text: str) -> list[dict]:
    """Parse [Speaker N]: tags from SAA output into structured segments.

    Returns list of {speaker, text} dicts in temporal order.
    """
    segments = []
    if not text.strip():
        return segments

    parts = _SAA_TAG_RE.split(text)

    for i in range(1, len(parts), 2):
        speaker_num = parts[i]
        spoken = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if spoken:
            segments.append({
                "speaker": f"SPEAKER_{speaker_num}",
                "text": spoken,
            })

    if not segments:
        stripped = parts[0].strip() if parts else text.strip()
        if stripped:
            segments.append({"speaker": "SPEAKER_1", "text": stripped})

    return segments


def _trim_prefix(transcript: str, max_words: int = 10) -> str:
    """Build prefix from last complete speaker turns for next-chunk context."""
    if not transcript.strip():
        return ""
    parts = re.split(r'(\[Speaker \d+\]:)', transcript)

    turns = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if re.match(r'\[Speaker \d+\]:', part):
            tag = part
            text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if text:
                trimmed = " ".join(text.split()[-max_words:])
                turns.append(f"{tag} {trimmed}")
            i += 2
        else:
            i += 1

    return " ".join(turns).strip()


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


def _transcribe_onnx(
    audio: np.ndarray,
    prompt_text: str,
    max_new_tokens: int,
) -> dict:
    from granite_onnx import transcribe_audio
    text = transcribe_audio(audio, prompt_text, max_new_tokens)
    return {"text": text}


def transcribe_saa_sync(
    audio: np.ndarray,
    prefix_text: Optional[str] = None,
    timeout_sec: int = 300,
) -> dict:
    """
    Run Granite SAA (Speaker-Attributed ASR) on an audio chunk.

    Returns:
        {text, segments, inference_time_sec, audio_duration_sec}
    """
    settings = get_settings()
    start_time = time.perf_counter()

    if not state.is_ready:
        raise TranscriptionError("Model not ready")

    if audio is None or len(audio) == 0:
        raise AudioValidationError("Empty audio")

    audio_duration = len(audio) / 16000

    with inference_timeout(timeout_sec):
        tokenizer = state.tokenizer
        if tokenizer is None:
            raise TranscriptionError("No tokenizer available")

        if state.onnx_ready:
            chat = [
                {"role": "system", "content": settings.system_prompt},
                {"role": "user", "content": settings.saa_prompt},
            ]
            extra = {"prefix_text": prefix_text} if prefix_text else {}
            prompt_text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, **extra
            )
            raw = _transcribe_onnx(audio, prompt_text, settings.max_new_tokens)
            text = raw.get("text", "")
        else:
            import torch
            processor = state.processor
            model = state.model
            device = state.device

            chat = [
                {"role": "system", "content": settings.system_prompt},
                {"role": "user", "content": settings.saa_prompt},
            ]
            extra = {"prefix_text": prefix_text} if prefix_text else {}
            prompt_text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, **extra
            )

            inputs = processor(prompt_text, audio, device=device, return_tensors="pt").to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=settings.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.3,
                )

            new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)

        text = clean_transcript(text)
        segments = parse_speaker_turns(text)

    inference_time = time.perf_counter() - start_time

    log.info(
        "saa_complete",
        audio_duration=f"{audio_duration:.2f}s",
        inference_time=f"{inference_time:.2f}s",
        segments=len(segments),
        text_preview=text[:100],
    )

    return {
        "text": text,
        "segments": segments,
        "inference_time_sec": inference_time,
        "audio_duration_sec": audio_duration,
    }


def transcribe_audio_sync(
    audio: np.ndarray,
    timeout_sec: int = 120,
) -> dict:
    """
    Plain ASR transcription (no speaker attribution). Returns dict with text.

    Args:
        audio: Raw PCM audio waveform (16kHz, mono), shape [T]
        timeout_sec: Inference timeout

    Returns:
        Dict with transcribed text, inference time, audio duration.
    """
    settings = get_settings()
    start_time = time.perf_counter()

    if not state.is_ready:
        raise TranscriptionError("Model not ready")

    if audio is None or len(audio) == 0:
        raise AudioValidationError("Empty audio")

    audio_duration = len(audio) / 16000
    if audio_duration > settings.max_audio_duration_sec:
        raise AudioValidationError(
            f"Audio too long: {audio_duration:.1f}s > {settings.max_audio_duration_sec}s max"
        )

    with inference_timeout(timeout_sec):
        tokenizer = state.tokenizer
        if tokenizer is None:
            raise TranscriptionError("No tokenizer available")

        if state.onnx_ready:
            chat = [
                {"role": "system", "content": settings.system_prompt},
                {"role": "user", "content": PLAIN_ASR_PROMPT},
            ]
            prompt_text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            raw = _transcribe_onnx(audio, prompt_text, settings.max_new_tokens)
            text = raw.get("text", "")
        else:
            import torch
            processor = state.processor
            model = state.model
            device = state.device

            chat = [
                {"role": "system", "content": settings.system_prompt},
                {"role": "user", "content": PLAIN_ASR_PROMPT},
            ]
            prompt_text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(prompt_text, audio, device=device, return_tensors="pt").to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=settings.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.3,
                )

            new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
            text = clean_transcript(text)

    inference_time = time.perf_counter() - start_time

    log.info(
        "transcription_complete",
        audio_duration=f"{audio_duration:.2f}s",
        inference_time=f"{inference_time:.2f}s",
        text_preview=text[:100],
    )

    return {
        "text": text,
        "inference_time_sec": inference_time,
        "audio_duration_sec": audio_duration,
    }


async def transcribe_audio_async(
    audio: Optional[np.ndarray] = None,
    timeout_sec: int = 120,
) -> dict:
    """Async wrapper for plain ASR transcription."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        transcribe_audio_sync,
        audio,
        timeout_sec,
    )


async def transcribe_saa_async(
    audio: Optional[np.ndarray] = None,
    prefix_text: Optional[str] = None,
    timeout_sec: int = 300,
) -> dict:
    """Async wrapper for SAA transcription."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        transcribe_saa_sync,
        audio,
        prefix_text,
        timeout_sec,
    )
