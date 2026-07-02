"""
Async audio transcription client with robust error handling.

Features:
- Async HTTP requests with aiohttp
- Automatic retry with exponential backoff  
- Progress reporting with tqdm
- Parallel file processing
- Configurable via environment variables
"""

import sys
import os
import re
import wave
import asyncio
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json
from contextlib import asynccontextmanager

import numpy as np
import aiohttp
from tqdm import tqdm

# ============================================================================
# Constants
# ============================================================================

MIN_ISLAND_DUR = 0.3  # Minimum duration for a segment to be considered an "island"

# ---------------------------------------------------------------------------
# Transcript hallucination cleaner (mirrors server.py clean_transcript)
# ---------------------------------------------------------------------------
_LOOP_RE = re.compile(r"(.{4,120}?)(?:\s+\1){2,}", re.IGNORECASE)

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
    prev = None
    while prev != text:
        prev = text
        m = _LOOP_RE.search(text)
        if not m:
            break
        unit   = m.group(1)
        before = _trim_partial_prefix(text[:m.start()], unit)
        after  = _trim_partial_suffix(text[m.end():], unit)
        parts  = [p for p in (before, "[inaudible]", after) if p]
        text   = " ".join(parts)
    text = re.sub(r"(\[inaudible\]\s*){2,}", "[inaudible] ", text)
    # Drop short fragments sandwiched between [inaudible] tags
    text = re.sub(
        r"\[inaudible\]\s+(?:\w[\w\s,\']{0,80}?)\s+\[inaudible\]",
        "[inaudible]",
        text,
    )
    return text.strip()

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Transcription client configuration."""
    
    # Server settings
    server_url: str = "http://127.0.0.1:8000"
    api_key: Optional[str] = None
    request_timeout: int = 600
    
    # Audio processing
    sample_rate: int = 16000
    silence_noise: str = "-35dB"
    silence_duration: float = 2.0
    min_segment_duration: float = 0.5
    merge_gap: float = 1.5
    max_chunk_duration: float = 120.0
    rms_silence_threshold: float = 0.005
    
    # Speaker detection
    speaker_turn_gap: float = 1.5
    num_speakers: Optional[int] = None
    diarization_threshold: Optional[float] = None
    window_sec: Optional[float] = None
    stride_sec: Optional[float] = None
    min_embed_duration: Optional[float] = None
    vad_threshold: Optional[float] = None
    vad_min_speech_duration_ms: Optional[int] = None
    known_speakers_file: Optional[str] = None
    
    # Batch settings
    batch_size: int = 4
    max_concurrent_requests: int = 2
    
    # Retry settings
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    
    # Output
    output_format: str = "txt"  # txt, srt, json
    
    # Supported formats
    supported_formats: set = field(default_factory=lambda: {
        ".mp3", ".mp4", ".wav", ".m4a", ".flac", 
        ".mov", ".mkv", ".avi", ".webm", ".ogg"
    })
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        num_speakers_env = os.getenv("TRANSCRIBE_NUM_SPEAKERS")
        threshold_env = os.getenv("TRANSCRIBE_DIARIZATION_THRESHOLD")
        window_env = os.getenv("TRANSCRIBE_WINDOW_SEC")
        stride_env = os.getenv("TRANSCRIBE_STRIDE_SEC")
        min_embed_env = os.getenv("TRANSCRIBE_MIN_EMBED_DURATION")
        vad_thresh_env = os.getenv("TRANSCRIBE_VAD_THRESHOLD")
        vad_min_speech_env = os.getenv("TRANSCRIBE_VAD_MIN_SPEECH_DURATION_MS")
        known_speakers_env = os.getenv("TRANSCRIBE_KNOWN_SPEAKERS_FILE")

        return cls(
            server_url=os.getenv("TRANSCRIBE_SERVER_URL", cls.server_url),
            api_key=os.getenv("TRANSCRIBE_API_KEY"),
            request_timeout=int(os.getenv("TRANSCRIBE_TIMEOUT", cls.request_timeout)),
            batch_size=int(os.getenv("TRANSCRIBE_BATCH_SIZE", cls.batch_size)),
            max_concurrent_requests=int(os.getenv("TRANSCRIBE_MAX_CONCURRENT", cls.max_concurrent_requests)),
            num_speakers=int(num_speakers_env) if num_speakers_env else None,
            diarization_threshold=float(threshold_env) if threshold_env else None,
            window_sec=float(window_env) if window_env else None,
            stride_sec=float(stride_env) if stride_env else None,
            min_embed_duration=float(min_embed_env) if min_embed_env else None,
            vad_threshold=float(vad_thresh_env) if vad_thresh_env else None,
            vad_min_speech_duration_ms=int(vad_min_speech_env) if vad_min_speech_env else None,
            known_speakers_file=known_speakers_env if known_speakers_env else None,
        )


# ============================================================================
# Logging
# ============================================================================

class Logger:
    """Simple colored logger."""
    
    COLORS = {
        "INFO": "\033[94m",   # Blue
        "WARN": "\033[93m",   # Yellow
        "ERROR": "\033[91m",  # Red
        "SUCCESS": "\033[92m", # Green
        "RESET": "\033[0m"
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def _log(self, level: str, msg: str, **kwargs):
        if not self.verbose and level == "INFO":
            return
        color = self.COLORS.get(level, "")
        reset = self.COLORS["RESET"]
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"{color}[{level}]{reset} {msg} {extra}".strip())
    
    def info(self, msg: str, **kwargs):
        self._log("INFO", msg, **kwargs)
    
    def warn(self, msg: str, **kwargs):
        self._log("WARN", msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self._log("ERROR", msg, **kwargs)
    
    def success(self, msg: str, **kwargs):
        self._log("SUCCESS", msg, **kwargs)


log = Logger()


# ============================================================================
# Audio Processing Utilities
# ============================================================================

def ffmpeg_convert(input_path: str, output_wav: str, sample_rate: int = 16000) -> bool:
    """Convert any audio/video to 16kHz mono WAV."""
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(sample_rate), "-ac", "1", "-vn",
            "-loglevel", "error", output_wav,
        ], capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"FFmpeg conversion failed: {e.stderr}")
        return False


def get_total_duration(wav_path: str) -> float:
    """Get audio duration using ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            wav_path,
        ], capture_output=True, text=True)
        return float(result.stdout.strip())
    except (ValueError, subprocess.CalledProcessError):
        return 0.0


def group_diarization_segments(
    segments: list[dict],
    min_island_duration: float,
    max_duration: float,
    merge_gap: float
) -> list[tuple[float, float, str]]:
    """Group tiny segments belonging to the same speaker into larger core segments."""
    if not segments:
        return []

    grouped = []
    current_start = segments[0]["start"]
    current_end = segments[0]["end"]
    current_speaker = segments[0]["speaker"]

    for seg in segments[1:]:
        # If same speaker, small gap, and total duration under max
        if (
            seg["speaker"] == current_speaker and
            (seg["start"] - current_end) <= merge_gap and
            (seg["end"] - current_start) <= max_duration
        ):
            current_end = seg["end"]
        else:
            grouped.append((current_start, current_end, current_speaker))
            current_start = seg["start"]
            current_end = seg["end"]
            current_speaker = seg["speaker"]

    # Append the final group
    grouped.append((current_start, current_end, current_speaker))

    # This function now handles the initial coarse merging of continuous speech turns.
    # Further refinement (like overlap handling) will occur in the advanced pipeline.
    return [dict(g) for g in grouped]


def extract_chunk(wav_path: str, start: float, duration: float, output: str, sample_rate: int) -> bool:
    """Extract a chunk from WAV file."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-ss", str(start), "-t", str(duration),
            "-ar", str(sample_rate), "-ac", "1",
            "-acodec", "pcm_s16le", "-loglevel", "error", output,
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def rms_check(wav_path: str, threshold: float) -> bool:
    """Check if audio has sufficient energy (not silence)."""
    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            if not frames:
                return False
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            if len(samples) == 0:
                return False
            rms = np.sqrt(np.mean(samples ** 2)) / 32768.0
            return rms >= threshold
    except Exception:
        return False


def extract_speaker_audio(wav_path: str, segments: list, speaker_name: str, output_path: str) -> bool:
    """Extract concatenated audio for a specific speaker from segments."""
    try:
        with wave.open(wav_path, 'rb') as orig:
            n_channels = orig.getnchannels()
            sample_width = orig.getsampwidth()
            frame_rate = orig.getframerate()
            orig_frames = orig.readframes(orig.getnframes())
        
        # Filter segments for this speaker
        speaker_segments = [s for s in segments if s["speaker"] == speaker_name]
        if not speaker_segments:
            return False
        
        # Extract audio chunks for this speaker
        audio_chunks = []
        for seg in speaker_segments:
            start_frame = int(seg["start"] * frame_rate)
            end_frame = int(seg["end"] * frame_rate)
            chunk_size = sample_width * n_channels
            start_byte = start_frame * chunk_size
            end_byte = end_frame * chunk_size
            audio_chunks.append(orig_frames[start_byte:end_byte])
        
        # Write concatenated audio
        with wave.open(output_path, 'wb') as out:
            out.setnchannels(n_channels)
            out.setsampwidth(sample_width)
            out.setframerate(frame_rate)
            out.writeframes(b''.join(audio_chunks))
        
        return True
    except Exception as e:
        log.warn(f"Failed to extract speaker audio: {e}")
        return False


def format_timestamp(seconds: float, fmt: str = "hms") -> str:
    """Format seconds to timestamp string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    
    if fmt == "hms":
        return f"{h:02d}:{m:02d}:{s:02d}"
    elif fmt == "srt":
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    else:
        return str(seconds)


def format_duration_readable(seconds: float) -> str:
    """Format seconds to human-readable duration (e.g., '1h 23m 45s' or '23m 45s')."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


# ============================================================================
# HTTP Client
# ============================================================================

class TranscriptionClient:
    """Async HTTP client for transcription server."""
    
    def __init__(self, config: Config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    @asynccontextmanager
    async def _get_session(self, timeout_override: int = None):
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            headers = {}
            if self.config.api_key:
                headers["X-API-Key"] = self.config.api_key

            timeout = aiohttp.ClientTimeout(total=timeout_override or self.config.request_timeout)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        elif timeout_override:
            # If the session is already created but we need a specific timeout for this request,
            # we should just create a new temporary session just for this request
            pass # We'll handle this in the calling function instead

        try:
            yield self.session
        except Exception:
            if self.session:
                await self.session.close()
                self.session = None
            raise

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            async with self._get_session() as session:
                async with session.get(f"{self.config.server_url}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("model_status") == "ready"
            return False
        except Exception as e:
            log.error(f"Health check failed: {e}")
            return False

    async def diarize_path(self, wav_path: str) -> tuple[list[dict], dict]:
        """Get diarization segments and speaker profiles from server."""
        try:
            headers = {}
            if self.config.api_key:
                headers["X-API-Key"] = self.config.api_key

            extended_timeout = aiohttp.ClientTimeout(total=3600)

            async with aiohttp.ClientSession(headers=headers, timeout=extended_timeout) as session:
                payload = {"wav_path": str(Path(wav_path).resolve())}
                if self.config.num_speakers is not None:
                    payload["num_speakers"] = self.config.num_speakers
                if self.config.diarization_threshold is not None:
                    payload["diarization_threshold"] = self.config.diarization_threshold
                if self.config.vad_threshold is not None:
                    payload["vad_threshold"] = self.config.vad_threshold
                if self.config.vad_min_speech_duration_ms is not None:
                    payload["vad_min_speech_duration_ms"] = self.config.vad_min_speech_duration_ms

                if not self.config.known_speakers_file and os.path.exists("voiceprints.json"):
                    self.config.known_speakers_file = "voiceprints.json"

                if self.config.known_speakers_file:
                    try:
                        with open(self.config.known_speakers_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            known_spk = {}
                            for name, profile in data.items():
                                if "embedding" in profile:
                                    known_spk[name] = profile
                            if known_spk:
                                payload["known_speakers"] = known_spk
                                print(f"[INFO] Loaded {len(known_spk)} voiceprints from {self.config.known_speakers_file}")
                    except Exception as e:
                        log.warn(f"Failed to load voiceprints from {self.config.known_speakers_file}: {e}")

                async with session.post(
                    f"{self.config.server_url}/diarize/path",
                    json=payload
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        try:
                            err_data = json.loads(body)
                            detail = err_data.get("error") or err_data.get("detail") or body
                        except Exception:
                            detail = body or f"HTTP {resp.status}"
                        log.warn(f"Failed to get diarization: {resp.status} {resp.reason} - {detail[:300]}")
                        return [], {}

                    pbar = None
                    current_step = None

                    try:
                        async for line in resp.content:
                            line = line.strip()
                            if not line:
                                continue

                            data = json.loads(line)
                            msg_type = data.get("type")

                            if msg_type == "progress":
                                step = data.get("step", "processing")
                                completed = data.get("completed", 0)
                                total = data.get("total", 1)

                                if current_step != step:
                                    if pbar:
                                        pbar.close()
                                    pbar = tqdm(total=total, desc=f"Diarizing ({step})", leave=False, dynamic_ncols=True)
                                    current_step = step

                                if pbar:
                                    pbar.update(completed - pbar.n)

                            elif msg_type == "result":
                                if pbar:
                                    pbar.close()
                                return data.get("segments", []), data.get("profiles", {})

                            elif msg_type == "error":
                                if pbar:
                                    pbar.close()
                                log.warn(f"Server diarization error: {data.get('error', 'Unknown error')}")
                                return [], {}
                    finally:
                        if pbar:
                            pbar.close()

                    # Fallback if stream ends without a result message
                    return [], {}

        except asyncio.TimeoutError:
            log.warn("Failed to get diarization: Connection timed out after 30 minutes")
            return [], {}
        except Exception as e:
            log.warn(f"Failed to get diarization: {e}")
            return [], {}
    
    async def transcribe_ws(
        self,
        wav_path: str,
        language: str = "en",
        diarize_segments: Optional[list] = None,
        progress_callback=None
    ) -> dict:
        """Transcribe a full audio file via WebSocket with per-chunk progress.

        Server aligns timed transcript with diarization segments and streams
        speaker-attributed segments back via the callback.

        Args:
            wav_path: Path to the audio file on the server
            language: Language code
            diarize_segments: List of {"start": ..., "end": ..., "speaker": ...}
            progress_callback: Called with progress and segment dicts

        Returns:
            dict with "inference_time_sec", "tokens_generated"
        """
        ws_url = self.config.server_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url += "/ws/transcribe"

        headers = {}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key

        result = {"inference_time_sec": 0.0, "tokens_generated": 0}

        async with aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=3600)
        ) as session:
            async with session.ws_connect(ws_url) as ws:
                await ws.send_json({
                    "wav_path": wav_path,
                    "language": language,
                    "api_key": self.config.api_key or "",
                    "diarize_segments": diarize_segments or []
                })

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type == "progress" or msg_type == "segment":
                            if progress_callback:
                                if asyncio.iscoroutinefunction(progress_callback):
                                    await progress_callback(data)
                                else:
                                    progress_callback(data)

                        elif msg_type == "done":
                            break

                        elif msg_type == "error":
                            raise Exception(data.get("message", "Unknown error"))

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise Exception(f"WebSocket error: {ws.exception()}")

        return result
    
    async def transcribe_batch(
        self, 
        wav_paths: list[str],
        language: str = "en"
    ) -> list[dict]:
        """
        Transcribe a batch of audio files with retry logic.
        Returns list of result dicts.
        """
        async with self.semaphore:
            return await self._transcribe_with_retry(wav_paths, language)
    
    async def _transcribe_with_retry(
        self, 
        wav_paths: list[str],
        language: str,
    ) -> list[dict]:
        """Transcribe with exponential backoff retry."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                async with self._get_session() as session:
                    async with session.post(
                        f"{self.config.server_url}/transcribe/paths",
                        json={"wav_paths": wav_paths, "language": language}
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        
                        # Handle both old and new response formats
                        if "results" in data:
                            results = data["results"]
                            if isinstance(results[0], str):
                                # Legacy format
                                return [{"text": r, "error": None} for r in results]
                            else:
                                # New format
                                return results
                        return data
                        
            except aiohttp.ClientError as e:
                last_error = e
                delay = min(
                    self.config.retry_base_delay * (2 ** attempt),
                    self.config.retry_max_delay
                )
                log.warn(f"Request failed, retrying in {delay:.1f}s", attempt=attempt+1, error=str(e))
                await asyncio.sleep(delay)
            
            except asyncio.TimeoutError:
                last_error = TimeoutError("Request timed out")
                delay = min(
                    self.config.retry_base_delay * (2 ** attempt),
                    self.config.retry_max_delay
                )
                log.warn(f"Request timed out, retrying in {delay:.1f}s", attempt=attempt+1)
                await asyncio.sleep(delay)
        
        # All retries failed
        log.error(f"All retries failed: {last_error}")
        return [{"text": "", "error": str(last_error)} for _ in wav_paths]


# ============================================================================
# Transcript Writer
# ============================================================================

class TranscriptWriter:
    """Handles writing transcripts in various formats."""
    
    def __init__(self, output_path: str, fmt: str = "txt"):
        self.output_path = output_path
        self.format = fmt
        self.segments: list[dict] = []
        self.last_end = 0.0
    
    def add_segment(
        self,
        text: str,
        start: float,
        end: float,
        speaker_turn_gap: float = None,  # Kept for compatibility, not used
        speaker: str = "SPEAKER",
        confidence: float = None,
        alternatives: list = None
    ):
        """Add a transcribed segment."""
        text = clean_transcript(text)
        if not text.strip():
            return

        is_new_speaker = (
            not self.segments or
            self.segments[-1].get("speaker") != speaker
        )

        segment_data = {
            "text": text.strip(),
            "start": start,
            "end": end,
            "new_speaker": is_new_speaker,
            "speaker": speaker,
            "confidence": confidence,
            "alternatives": alternatives or []
        }
        self.segments.append(segment_data)
        self.last_end = end

        # For text format, we can write out the delta immediately
        if self.format == "txt":
            self._append_txt(segment_data)

    def _format_speaker_line(self, seg: dict) -> str:
        """Format speaker line with confidence and alternatives."""
        speaker_name = seg.get("speaker", "SPEAKER")
        conf = seg.get("confidence")
        alternatives = seg.get("alternatives", [])

        if conf is not None:
            conf_str = f" ({conf:.0%})"
        else:
            conf_str = ""

        # Add alternatives if any are >= 50%
        alt_str = ""
        if alternatives:
            high_conf_alts = [a for a in alternatives if a.get("confidence", 0) >= 0.5]
            if high_conf_alts:
                alt_names = ", ".join(f"{a['speaker']} ({a['confidence']:.0%})" for a in high_conf_alts)
                alt_str = f" [also: {alt_names}]"

        return f"\n[{format_timestamp(seg['start'])}] {speaker_name}{conf_str}{alt_str}:\n"

    def _append_txt(self, seg: dict):
        """Append a single segment directly to the file."""
        with open(self.output_path, "a", encoding="utf-8") as f:
            if seg["new_speaker"]:
                f.write(self._format_speaker_line(seg))
            f.write(seg["text"] + " ")
    
    def write(self):
        """Write final cleanup for transcript."""
        if self.format == "txt":
            # Just do final whitespace cleanup
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    content = f.read()
                content = re.sub(r"\[\d{2}:\d{2}:\d{2}\] SPEAKER\w*:\s*$", "", content.rstrip()).rstrip()
                with open(self.output_path, "w", encoding="utf-8") as f:
                    f.write(content + "\n")
            except FileNotFoundError:
                pass
        elif self.format == "srt":
            self._write_srt()
        elif self.format == "json":
            self._write_json()
        else:
            self._write_txt()

    def _write_txt(self):
        """Write plain text format."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            for seg in self.segments:
                if seg["new_speaker"]:
                    f.write(self._format_speaker_line(seg))
                f.write(seg["text"] + " ")

        # Clean up trailing whitespace
        with open(self.output_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = re.sub(r"\[\d{2}:\d{2}:\d{2}\] SPEAKER\w*:\s*$", "", content.rstrip()).rstrip()

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
    
    def _write_srt(self):
        """Write SRT subtitle format."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(self.segments, 1):
                start_ts = format_timestamp(seg["start"], "srt")
                end_ts = format_timestamp(seg["end"], "srt")
                f.write(f"{i}\n")
                f.write(f"{start_ts} --> {end_ts}\n")
                f.write(f"{seg['text']}\n\n")
    
    def _write_json(self):
        """Write JSON format."""
        import json
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump({
                "segments": self.segments,
                "full_text": " ".join(s["text"] for s in self.segments)
            }, f, indent=2, ensure_ascii=False)


# ============================================================================
# Segment Refinement and Overlap Detection
# ============================================================================

def refactor_and_detect_overlaps(diarize_segments: list[dict], min_duration: float,
                                  max_chunk_duration: float, merge_gap: float) -> tuple[list[tuple], list[dict]]:
    """
    Refine diarization segments and detect overlaps between speakers.

    Args:
        diarize_segments: List of dicts with 'start', 'end', 'speaker' keys
        min_duration: Minimum segment duration to keep
        max_chunk_duration: Maximum duration for a single chunk
        merge_gap: Maximum gap between segments to merge

    Returns:
        Tuple of (refined_segments, overlaps)
        - refined_segments: List of tuples (start, end, speaker, confidence)
        - overlaps: List of overlapping regions with speaker info
    """
    if not diarize_segments:
        return [], []

    # Convert start and end to floats, preserve confidence and alternatives
    processed = []
    for seg in diarize_segments:
        processed.append({
            'start': float(seg['start']),
            'end': float(seg['end']),
            'speaker': seg['speaker'],
            'confidence': float(seg.get('confidence', 0.5)),
            'alternatives': seg.get('alternatives', [])
        })

    # Sort segments by start time
    sorted_segments = sorted(processed, key=lambda x: x['start'])

    # Merge segments that are close together (same speaker)
    merged = []
    current = dict(sorted_segments[0])

    for seg in sorted_segments[1:]:
        if seg['speaker'] == current['speaker'] and (seg['start'] - current['end']) <= merge_gap:
            current['end'] = max(current['end'], seg['end'])
            current['confidence'] = (current.get('confidence', 0.5) + seg.get('confidence', 0.5)) / 2
        else:
            if current['end'] - current['start'] >= min_duration:
                merged.append(current)
            current = dict(seg)

    if current['end'] - current['start'] >= min_duration:
        merged.append(current)

# Note: Truncation logic disabled - confidence metric not reliable enough
    # Re-enable if confidence improves in the future
    # Disabled to prevent cutting sentences incorrectly

    # Detect overlaps between segments of different speakers
    # The truncation was causing sentences to be cut incorrectly
    # merged = truncated

    # Detect overlaps between segments of different speakers
    overlaps = []
    refined = []

    for i, seg in enumerate(merged):
        # Return as tuple (start, end, speaker, confidence, alternatives)
        refined.append((
            seg['start'],
            seg['end'],
            seg['speaker'],
            seg.get('confidence', 0.5),
            seg.get('alternatives', [])
        ))
        # Check for overlaps with subsequent segments
        for j in range(i + 1, len(merged)):
            other = merged[j]
            if other['start'] >= seg['end']:
                break
            if other['speaker'] != seg['speaker'] and other['start'] < seg['end']:
                overlap_start = max(seg['start'], other['start'])
                overlap_end = min(seg['end'], other['end'])
                if overlap_end > overlap_start:
                    overlaps.append({
                        'start': overlap_start,
                        'end': overlap_end,
                        'speakers': [seg['speaker'], other['speaker']]
                    })

    return refined, overlaps


# ============================================================================
# Main Transcription Logic
# ============================================================================

async def transcribe_file(
    input_path: str, 
    config: Config,
    client: TranscriptionClient,
    progress_bar: Optional[tqdm] = None,
    args = None
) -> Optional[dict]:
    """
    Transcribe a single audio/video file.
    Returns dict with output_path, total_duration, and speaker_stats, or None on failure.
    """
    p = Path(input_path).resolve()
    
    if p.suffix.lower() not in config.supported_formats:
        log.warn(f"Unsupported format: {p.suffix}", file=p.name)
        return None  # Return None for errors
    
    pid = os.getpid()
    temp_dir = tempfile.mkdtemp(prefix="transcribe_")
    temp_wav = os.path.join(temp_dir, "audio.wav")
    
    # Determine output path
    output_ext = {"txt": ".txt", "srt": ".srt", "json": ".json"}.get(config.output_format, ".txt")
    output_path = str(p.parent / (p.stem + output_ext))
    
    log.info(f"Processing: {p.name}")
    
    try:
        # Convert to WAV
        log.info("Converting to 16kHz mono WAV...")
        if not ffmpeg_convert(str(p), temp_wav, config.sample_rate):
            log.error("Conversion failed", file=p.name)
            return None
        
        # Detect speech segments via Diarization endpoint
        log.info("Running diarization on the full audio...")
        diarize_segments, speaker_profiles = await client.diarize_path(temp_wav)

        # Update voiceprints.json with new speakers (those with embeddings)
        if speaker_profiles and diarize_segments:
            voiceprints_path = Path("voiceprints.json")
            existing_vp = {}
            if voiceprints_path.exists():
                try:
                    with open(voiceprints_path, 'r', encoding='utf-8') as f:
                        existing_vp = json.load(f)
                except:
                    pass
            
            # Calculate total duration and speaker confidence from diarize_segments
            speaker_durations = {}
            speaker_confidence_sum = {}
            speaker_counts = {}
            for seg in diarize_segments:
                spk = seg["speaker"]
                dur = seg["end"] - seg["start"]
                conf = float(seg.get("confidence", 0.5))
                speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur
                speaker_confidence_sum[spk] = speaker_confidence_sum.get(spk, 0.0) + conf
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

            updated = False
            name_remap = {}
            filename_prefix = re.sub(r'[^a-zA-Z0-9_-]', '_', p.stem)

            for name, profile in speaker_profiles.items():
                if "embedding" not in profile:
                    continue
                # Skip if already a known/existing voiceprint (or if we matched it)
                if name in existing_vp:
                    continue

                total_dur = speaker_durations.get(name, 0.0)
                if total_dur < 10.0:
                    continue

                avg_conf = speaker_confidence_sum.get(name, 0.0) / max(1, speaker_counts.get(name, 1))
                if avg_conf < 0.65:
                    continue

                # Generate persistent speaker name using processed filename stem as prefix
                existing_names = [n for n in existing_vp.keys() if n.startswith(f"{filename_prefix}_SPEAKER")]
                next_n = len(existing_names) + 1
                persistent_name = f"{filename_prefix}_SPEAKER_{next_n}"

                existing_vp[persistent_name] = {
                    "pitch_hz": profile.get("pitch_hz", 0.0),
                    "total_speech_sec": total_dur,
                    "embedding": profile.get("embedding", []),
                }
                name_remap[name] = persistent_name
                updated = True
                print(f"[INFO] Added new voiceprint: {persistent_name} (dur={total_dur:.1f}s, conf={avg_conf:.2%})")
            
            if updated:
                try:
                    with open(voiceprints_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_vp, f, indent=2, ensure_ascii=False)
                    print(f"[INFO] Updated voiceprints.json with {len(existing_vp)} speakers")
                except Exception as e:
                    log.warn(f"Failed to update voiceprints.json: {e}")
                
                # Rename segments and profiles to match the persistent voiceprint names
                for seg in diarize_segments:
                    old = seg["speaker"]
                    if old in name_remap:
                        seg["speaker"] = name_remap[old]
                for old, new in list(name_remap.items()):
                    if old in speaker_profiles:
                        speaker_profiles[new] = speaker_profiles.pop(old)
        
        # Post-process unknown speakers if requested
        # Allow running even when `speaker_profiles` is empty so we can attempt
        # to extract voiceprints for unknowns that the server didn't embed.
        if args and getattr(args, 'post_process_unknowns', False) and diarize_segments:
            voiceprints_path = Path("voiceprints.json")
            existing_vp = {}
            if voiceprints_path.exists():
                try:
                    with open(voiceprints_path, 'r', encoding='utf-8') as f:
                        existing_vp = json.load(f)
                except:
                    pass
            
            # Identify unknown speakers (those without embeddings in current profiles)
            known_speakers = set(speaker_profiles.keys()) if speaker_profiles else set()
            unknown_speakers = set(s["speaker"] for s in diarize_segments) - known_speakers
            
            if unknown_speakers:
                # Calculate stats for unknowns from diarize_segments
                speaker_durations = {}
                speaker_confidence_sum = {}
                speaker_counts = {}
                for seg in diarize_segments:
                    if seg["speaker"] in unknown_speakers:
                        spk = seg["speaker"]
                        dur = seg["end"] - seg["start"]
                        conf = float(seg.get("confidence", 0.5))
                        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur
                        speaker_confidence_sum[spk] = speaker_confidence_sum.get(spk, 0.0) + conf
                        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
                
                # Filter and extract clean voiceprints for unknown speakers
                filename_prefix = re.sub(r'[^a-zA-Z0-9_-]', '_', p.stem)
                post_name_remap = {}
                
                for spk in unknown_speakers:
                    total_dur = speaker_durations.get(spk, 0.0)
                    if total_dur < 30.0:
                        continue

                    avg_conf = speaker_confidence_sum.get(spk, 0.0) / max(1, speaker_counts.get(spk, 1))
                    if avg_conf < 0.65:
                        continue

                    # Extract clean audio for this speaker
                    speaker_temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    speaker_temp_path = speaker_temp_wav.name
                    speaker_temp_wav.close()
                    
                    if not extract_speaker_audio(temp_wav, diarize_segments, spk, speaker_temp_path):
                        try:
                            os.unlink(speaker_temp_path)
                        except:
                            pass
                        continue
                    
                    try:
                        # Compute embedding locally for the extracted speaker audio
                        print(f"[INFO] Computing local embedding for unknown speaker {spk}...")
                        from settings import Settings
                        from voiceprint_utils import load_audio_segment, get_embedding_session
                        from speaker.embedding import extract_embedding, compute_pitch

                        # Load audio and get embedding session
                        waveform, sr = load_audio_segment(speaker_temp_path, 0, None)
                        embedding_session = get_embedding_session(Settings())

                        emb = extract_embedding(waveform, sr, embedding_session)
                        if emb is not None:
                            pitch = compute_pitch(waveform, sr)
                            if isinstance(pitch, tuple):
                                pitch = pitch[0]

                            existing_names = [n for n in existing_vp.keys() if n.startswith(f"{filename_prefix}_post_SPEAKER")]
                            next_n = len(existing_names) + 1
                            persistent_name = f"{filename_prefix}_post_SPEAKER_{next_n}"

                            voiceprint_entry = {
                                "pitch_hz": round(pitch, 1) if pitch > 0 else 0.0,
                                "total_speech_sec": total_dur,
                                "embedding": emb,
                            }

                            existing_vp[persistent_name] = voiceprint_entry
                            post_name_remap[spk] = persistent_name
                            print(f"[INFO] Added post-processed voiceprint: {persistent_name} (dur={total_dur:.1f}s, conf={avg_conf:.2%})")

                    except Exception as e:
                        log.warn(f"Failed to extract voiceprint for {spk}: {e}")
                    finally:
                        try:
                            os.unlink(speaker_temp_path)
                        except:
                            pass
                
                if post_name_remap:
                    for seg in diarize_segments:
                        old = seg["speaker"]
                        if old in post_name_remap:
                            seg["speaker"] = post_name_remap[old]
                
                # Save updated voiceprints if any were added
                if existing_vp: 
                    try:
                        with open(voiceprints_path, 'w', encoding='utf-8') as f:
                            json.dump(existing_vp, f, indent=2, ensure_ascii=False)
                        print(f"[INFO] Updated voiceprints.json with post-processed speakers")
                    except Exception as e:
                        log.warn(f"Failed to save post-processed voiceprints: {e}")
        

        if args and getattr(args, 'segments_trained', None) is not None and diarize_segments:
            import soundfile as sf
            trained_dir = Path(args.segments_trained)
            max_seg = args.max_trained_segments or 20
            audio_data, sr = sf.read(temp_wav)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            from collections import defaultdict
            spk_segs = defaultdict(list)
            for seg in diarize_segments:
                spk_segs[seg["speaker"]].append(seg)
            trained_count = 0
            for spk, segs in spk_segs.items():
                sorted_segs = sorted(segs, key=lambda s: float(s.get("confidence", 0.5)), reverse=True)
                top_segs = sorted_segs[:max_seg]
                spk_dir = trained_dir / spk
                spk_dir.mkdir(parents=True, exist_ok=True)
                for rank, seg in enumerate(top_segs, 1):
                    start_f = seg["start"]
                    end_f = seg["end"]
                    dur = end_f - start_f
                    if dur < 1.5:
                        continue
                    conf = int(float(seg.get("confidence", 0.5)) * 100)
                    out_file = spk_dir / f"{rank:02d}_{int(start_f)}_{conf:02d}_{dur:02.0f}.flac"
                    start_s = int(start_f * sr)
                    end_s = int(end_f * sr)
                    sf.write(str(out_file), audio_data[start_s:end_s], sr)
                    trained_count += 1
                print(f"  - {spk}: {len(top_segs)} top-trained to {spk_dir}")
            if trained_count:
                print(f"[INFO] Extracted {trained_count} top-trained segments to {trained_dir}")
        
        if not diarize_segments:
            log.warn("Diarization returned no segments. Assuming single speaker for fallback.", file=p.name)
            total_dur = get_total_duration(temp_wav)
            # Fallback: treating the whole file as one segment for embedding/analysis
            diarize_segments = [{"start": 0.0, "end": total_dur, "speaker": "SPEAKER"}]
            speaker_profiles = {}
        
        # 1. Refine segments and detect overlaps.
        # Overlaps are detected but not yet used for embedding/audio extraction in this iteration.
        refined_segments, overlaps = refactor_and_detect_overlaps(
            diarize_segments,
            MIN_ISLAND_DUR, # Use old threshold for compatibility
            config.max_chunk_duration,
            config.merge_gap
        )
        
        # 2. Proceed with the refined segments for subsequent steps (embedding/profiling)
        segments = refined_segments
        
        log.info(f"Grouped into {len(segments)} speaker-homogeneous chunks")

        if not segments:
            log.warn("No valid segments left after grouping", file=p.name)
            return None

        # Initialize writer
        writer = TranscriptWriter(output_path, config.output_format)

        # Write speaker legend header if profiles available
        with open(output_path, "w", encoding="utf-8") as f:
            if speaker_profiles and config.output_format == "txt":
                f.write("=" * 60 + "\n")
                f.write("SPEAKER VOICE PROFILES\n")
                f.write("=" * 60 + "\n")
                for spk in sorted(speaker_profiles.keys()):
                    p_info = speaker_profiles[spk]
                    f.write(
                        f"  {spk}: pitch={p_info.get('pitch_hz', 0):.0f}Hz "
                        f"(±{p_info.get('pitch_std', 0):.0f}Hz)  "
                        f"energy={p_info.get('energy_rms', 0):.4f}  "
                        f"speech={p_info.get('total_speech_sec', 0):.0f}s\n"
                    )
                f.write("=" * 60 + "\n\n")

        # Convert diarization segments to dicts for the server
        diarize_segments_for_ws = []
        for seg in segments:
            if len(seg) >= 5:
                start, end, speaker, confidence, alternatives = seg[0], seg[1], seg[2], seg[3], seg[4] if len(seg) > 4 else []
            elif len(seg) >= 4:
                start, end, speaker, confidence = seg[0], seg[1], seg[2], seg[3]
                alternatives = []
            else:
                start, end, speaker = seg[0], seg[1], seg[2]
                confidence = None
                alternatives = []
            diarize_segments_for_ws.append({
                "start": start, "end": end, "speaker": speaker,
                "confidence": confidence, "alternatives": alternatives
            })

        # Transcribe the full audio via WebSocket — server aligns and streams
        # speaker-attributed segments back, which we write immediately.
        if progress_bar is not None:
            total_audio_dur = get_total_duration(temp_wav)
            progress_bar.total = int(round(total_audio_dur))
            progress_bar.set_description(f"{p.name[:20]}")
            progress_bar.unit = ""
            progress_bar.bar_format = "{desc}: {percentage:.0f}% |{bar}| {postfix} ({elapsed}<{remaining})"
            progress_bar.set_postfix_str("Waiting...")
            progress_bar.update(0)

        def on_ws_message(data):
            msg_type = data.get("type")
            if msg_type == "progress":
                if progress_bar is not None:
                    chunk_num = data.get("chunk", 0)
                    total = data.get("total_chunks", 1)
                    progress_bar.set_postfix_str(f"Chunk {chunk_num}/{total}")
                    if total > 0:
                        target = int(round(progress_bar.total * chunk_num / total))
                        step = target - progress_bar.n
                        if step > 0:
                            progress_bar.update(step)
            elif msg_type == "segment":
                text = data.get("text", "")
                if not text.strip():
                    return
                speaker = data.get("speaker", "SPEAKER")
                start = data.get("start", 0.0)
                end = data.get("end", 0.0)
                ds_idx = data.get("ds_idx", -1)
                conf = None
                alts = []
                if 0 <= ds_idx < len(diarize_segments_for_ws):
                    ds = diarize_segments_for_ws[ds_idx]
                    conf = ds.get("confidence")
                    alts = ds.get("alternatives", [])
                writer.add_segment(text, start, end, speaker=speaker,
                                   confidence=conf, alternatives=alts)
            elif msg_type == "done":
                if progress_bar is not None:
                    progress_bar.set_postfix_str("Done")
                    progress_bar.update(progress_bar.total - progress_bar.n)

        await client.transcribe_ws(
            temp_wav, language="en",
            diarize_segments=diarize_segments_for_ws,
            progress_callback=on_ws_message
        )
        
        # Collect speaker statistics
        speaker_stats = {}
        if diarize_segments:
            for seg in diarize_segments:
                spk = seg.get("speaker", "UNKNOWN")
                if spk not in speaker_stats:
                    speaker_stats[spk] = {
                        "total_duration": 0.0,
                        "segment_count": 0,
                        "times_active": 0,
                        "last_end": -1.0
                    }
                
                duration = seg.get("end", 0) - seg.get("start", 0)
                speaker_stats[spk]["total_duration"] += duration
                speaker_stats[spk]["segment_count"] += 1
                
                # Count "times active" as number of contiguous speech blocks
                current_start = seg.get("start", 0)
                if current_start > speaker_stats[spk]["last_end"] + 0.5:  # 500ms gap threshold
                    speaker_stats[spk]["times_active"] += 1
                speaker_stats[spk]["last_end"] = seg.get("end", 0)
            
            # Calculate percentages
            total_audio_duration = get_total_duration(temp_wav)
            for spk in speaker_stats:
                duration = speaker_stats[spk]["total_duration"]
                speaker_stats[spk]["percentage"] = (duration / total_audio_duration * 100) if total_audio_duration > 0 else 0
                del speaker_stats[spk]["last_end"]  # Remove temporary field
        
        # Write output
        writer.write()
        log.success(f"Saved: {output_path}")
        
        return {
            "output_path": output_path,
            "total_duration": get_total_duration(temp_wav),
            "speaker_stats": speaker_stats
        }
        
    except Exception as e:
        log.error(f"Failed to process {p.name}: {e}")
        return None
    
    finally:
        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


async def transcribe_files(input_paths: list[str], config: Config, args):
    """Transcribe multiple files."""
    client = TranscriptionClient(config)
    
    try:
        import time
        overall_start_time = time.perf_counter()

        # Health check
        log.info("Checking server health...")
        if not await client.health_check():
            log.error(f"Server not ready at {config.server_url}")
            log.info("Start the server with: python server.py")
            return
        
        log.success("Server is ready")
        
        # Process files
        results = []
        file_times = []  # Track processing time per file
        
        for input_path in input_paths:
            file_start_time = time.perf_counter()
            with tqdm(unit="", leave=True, dynamic_ncols=True) as pbar:
                result = await transcribe_file(input_path, config, client, pbar, args)
                results.append((input_path, result))
                file_times.append(time.perf_counter() - file_start_time)
        
        # Summary
        print("\n" + "=" * 70)
        print("TRANSCRIPTION SUMMARY")
        print("=" * 70)
        
        success = sum(1 for _, r in results if r)
        failed = len(results) - success
        
        total_audio_duration = 0.0
        total_file_size = 0.0
        
        for (input_path, result), proc_time in zip(results, file_times):
            p = Path(input_path)
            status = "✓" if result else "✗"
            print(f"\n  {status} {p.name}")
            
            if result:
                output_path = result.get("output_path") if isinstance(result, dict) else result
                file_duration = result.get("total_duration", 0) if isinstance(result, dict) else 0
                speaker_stats = result.get("speaker_stats", {}) if isinstance(result, dict) else {}
                
                print(f"    → {output_path}")
                if file_duration > 0:
                    total_audio_duration += file_duration
                    print(f"    Duration: {format_duration_readable(file_duration)}")
                    if proc_time > 0:
                        speedup = file_duration / proc_time
                        print(f"    Processing time: {format_duration_readable(proc_time)} (speedup: {speedup:.1f}x)")
                
                # Display speaker statistics
                if speaker_stats:
                    print(f"    Speakers ({len(speaker_stats)}):")
                    for spk in sorted(speaker_stats.keys()):
                        stats = speaker_stats[spk]
                        duration = stats.get("total_duration", 0)
                        percentage = stats.get("percentage", 0)
                        times_active = stats.get("times_active", 0)
                        seg_count = stats.get("segment_count", 0)
                        print(f"      {spk}: {format_duration_readable(duration)} ({percentage:.1f}%) | "
                              f"{times_active} active period(s) | {seg_count} segment(s)")
                
                # Calculate file size in MB
                if p.exists():
                    file_size_mb = p.stat().st_size / (1024 * 1024)
                    total_file_size += file_size_mb
        
        print("\n" + "=" * 70)
        print(f"Completed: {success}/{len(results)} files")
        if failed:
            print(f"Failed: {failed} files")

        overall_time = time.perf_counter() - overall_start_time
        print(f"Total processing time: {format_duration_readable(overall_time)}")
        
        if total_audio_duration > 0:
            overall_speedup = total_audio_duration / overall_time
            print(f"Overall speedup: {overall_speedup:.1f}x")
        
        if total_file_size > 0:
            print(f"Total media size: {total_file_size:.1f} MB")
        
        print("=" * 70)

    finally:
        await client.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using ASR server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s recording.mp3
  %(prog)s *.wav --format srt
  %(prog)s meeting.mp4 --server http://localhost:8000
  
  Environment variables:
    TRANSCRIBE_SERVER_URL  - Server URL (default: http://127.0.0.1:8000)
    TRANSCRIBE_API_KEY     - API key for authentication
    TRANSCRIBE_TIMEOUT     - Request timeout in seconds
    TRANSCRIBE_BATCH_SIZE  - Batch size for requests
    TRANSCRIBE_NUM_SPEAKERS - Exact number of speakers
    TRANSCRIBE_DIARIZATION_THRESHOLD - Threshold for clustering
          """
    )
    
    parser.add_argument(
        "files", 
        nargs="*", 
        help="Audio/video files to transcribe (optional if --shutdown is used)"
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Shutdown the server instead of transcribing files"
    )
    parser.add_argument(
        "--server", "-s",
        default=None,
        help="Server URL (default: http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["txt", "srt", "json"],
        default="txt",
        help="Output format (default: txt)"
    )
    parser.add_argument(
        "--language", "-l",
        default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Batch size for requests"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="API key for authentication"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=None,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers (improves diarization if known)"
    )
    parser.add_argument(
        "--diarization-threshold",
        type=float,
        default=None,
        help="Distance threshold for clustering (overrides server default)"
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=None,
        help="Size of sliding window in seconds for embeddings (e.g. 1.5, 3.0)"
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=None,
        help="Stride step between windows in seconds (e.g. 0.375, 1.5)"
    )
    parser.add_argument(
        "--min-embed-duration",
        type=float,
        default=None,
        help="Minimum segment duration (sec) to include in clustering (e.g. 0.5)"
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="VAD speech probability cutoff (0.0-1.0)"
    )
    parser.add_argument(
        "--vad-min-speech",
        type=int,
        default=None,
        help="VAD minimum speech chunk length (ms)"
    )
    parser.add_argument(
        "--voiceprints",
        type=str,
        default=None,
        help="Path to JSON file with known voice profiles (embedding-based identification)"
    )
    parser.add_argument(
        "--post-process-unknowns",
        action="store_true",
        help="Extract clean voiceprints for unknown speakers with substantial talk (>= 30s, >= 0.8 confidence)"
    )
    parser.add_argument(
        "--segments-trained",
        nargs='?', const='segments_trained', default=None,
        help="Directory to save top N best-matching segments per speaker for retraining (default: segments_trained/)"
    )
    parser.add_argument(
        "--max-trained-segments",
        type=int, default=20,
        help="Maximum best-matching segments per speaker for retraining (default: 20)"
    )

    args = parser.parse_args()
    
    # Build config
    config = Config.from_env()
    
    if args.server:
        config.server_url = args.server
    if args.format:
        config.output_format = args.format
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.api_key:
        config.api_key = args.api_key
    if args.timeout:
        config.request_timeout = args.timeout
    if args.num_speakers is not None:
        config.num_speakers = args.num_speakers
    if args.diarization_threshold is not None:
        config.diarization_threshold = args.diarization_threshold
    if args.window_sec is not None:
        config.window_sec = args.window_sec
    if args.stride_sec is not None:
        config.stride_sec = args.stride_sec
    if args.min_embed_duration is not None:
        config.min_embed_duration = args.min_embed_duration
    if args.vad_threshold is not None:
        config.vad_threshold = args.vad_threshold
    if args.vad_min_speech is not None:
        config.vad_min_speech_duration_ms = args.vad_min_speech
    if args.voiceprints is not None:
        config.known_speakers_file = args.voiceprints

    if args.shutdown:
        async def do_shutdown():
            headers = {}
            if config.api_key:
                headers["X-API-Key"] = config.api_key
            try:
                log.info(f"Sending shutdown request to {config.server_url}")
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.post(f"{config.server_url}/shutdown") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            log.info(f"Server responded: {data.get('status')}")
                        else:
                            log.error(f"Failed to shutdown server: HTTP {resp.status}")
            except Exception as e:
                log.error(f"Error shutting down server: {e}")
        
        asyncio.run(do_shutdown())
        sys.exit(0)

    # Expand globs and validate files
    input_files = []
    for pattern in args.files:
        path = Path(pattern)
        if path.exists():
            input_files.append(str(path))
        else:
            # Try glob
            matches = list(Path(".").glob(pattern))
            if matches:
                input_files.extend(str(m) for m in matches)
            else:
                log.warn(f"File not found: {pattern}")
    
    if not input_files:
        log.error("No valid input files")
        sys.exit(1)
    
    log.info(f"Processing {len(input_files)} file(s)")
    
    # Run async
    asyncio.run(transcribe_files(input_files, config, args))


if __name__ == "__main__":
    main()