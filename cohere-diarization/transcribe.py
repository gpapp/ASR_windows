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
# Configuration
# ============================================================================

@dataclass
class Config:
    """Transcription client configuration."""
    
    # Server settings
    server_url: str = "http://127.0.0.1:8000"
    api_key: Optional[str] = None
    request_timeout: int = 120
    
    # Audio processing
    sample_rate: int = 16000
    silence_noise: str = "-35dB"
    silence_duration: float = 2.0
    min_segment_duration: float = 0.5
    merge_gap: float = 2.5
    max_chunk_duration: float = 30.0
    rms_silence_threshold: float = 0.005
    
    # Speaker detection
    speaker_turn_gap: float = 1.5
    
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
        return cls(
            server_url=os.getenv("TRANSCRIBE_SERVER_URL", cls.server_url),
            api_key=os.getenv("TRANSCRIBE_API_KEY"),
            request_timeout=int(os.getenv("TRANSCRIBE_TIMEOUT", cls.request_timeout)),
            batch_size=int(os.getenv("TRANSCRIBE_BATCH_SIZE", cls.batch_size)),
            max_concurrent_requests=int(os.getenv("TRANSCRIBE_MAX_CONCURRENT", cls.max_concurrent_requests)),
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
    max_duration: float,
    merge_gap: float
) -> list[tuple[float, float, str]]:
    """Group tiny Pyannote segments belonging to the same speaker into larger chunks."""
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

    # Handle segments that are still somehow larger than max_duration
    final_chunks = []
    for start, end, speaker in grouped:
        dur = end - start
        if dur > max_duration:
            # Split it evenly
            n = int(dur / max_duration) + 1
            step = dur / n
            for i in range(n):
                final_chunks.append((start + i * step, start + (i + 1) * step, speaker))
        else:
            final_chunks.append((start, end, speaker))

    return final_chunks


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

    async def diarize_path(self, wav_path: str) -> list[dict]:
        """Get diarization segments from server with real-time progress."""
        try:
            headers = {}
            if self.config.api_key:
                headers["X-API-Key"] = self.config.api_key

            # Diarization can take a long time even with VAD, especially on CPU.
            # We bypass the shared session here to use a massively extended timeout
            extended_timeout = aiohttp.ClientTimeout(total=1800)  # 30 minutes for diarization

            async with aiohttp.ClientSession(headers=headers, timeout=extended_timeout) as session:
                async with session.post(
                    f"{self.config.server_url}/diarize/path",
                    json={"wav_path": str(Path(wav_path).resolve())}
                ) as resp:
                    resp.raise_for_status()

                    # Handle streaming NDJSON response
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

                                # Create or update progress bar based on step
                                if current_step != step:
                                    if pbar:
                                        pbar.close()
                                    pbar = tqdm(total=total, desc=f"Diarizing ({step})", leave=False, dynamic_ncols=True)
                                    current_step = step

                                if pbar:
                                    # Update by delta
                                    pbar.update(completed - pbar.n)

                            elif msg_type == "result":
                                if pbar:
                                    pbar.close()
                                return data.get("segments", [])

                            elif msg_type == "error":
                                if pbar:
                                    pbar.close()
                                log.warn(f"Server diarization error: {data.get('error', 'Unknown error')}")
                                return []
                    finally:
                        if pbar:
                            pbar.close()

                    # Fallback if stream ends without a result message
                    return []

        except asyncio.TimeoutError:
            log.warn("Failed to get diarization: Connection timed out after 30 minutes")
            return []
        except aiohttp.ClientResponseError as e:
            text = "Unknown error"
            if 'resp' in locals():
                try:
                    text = await resp.text()
                except Exception:
                    pass
            log.warn(f"Failed to get diarization: {e} - Response: {text[:200]}")
            return []
        except Exception as e:
            log.warn(f"Failed to get diarization: {e}")
            return []
    
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
        speaker: str = "SPEAKER"
    ):
        """Add a transcribed segment."""
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
            "speaker": speaker
        }
        self.segments.append(segment_data)
        self.last_end = end

        # For text format, we can write out the delta immediately
        if self.format == "txt":
            self._append_txt(segment_data)

    def _append_txt(self, seg: dict):
        """Append a single segment directly to the file."""
        with open(self.output_path, "a", encoding="utf-8") as f:
            if seg["new_speaker"]:
                f.write(f"\n[{format_timestamp(seg['start'])}] SPEAKER:\n")
            f.write(seg["text"] + " ")
    
    def write(self):
        """Write final cleanup for transcript."""
        if self.format == "txt":
            # Just do final whitespace cleanup
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    content = f.read()
                content = re.sub(r"\[\d{2}:\d{2}:\d{2}\] SPEAKER:\s*$", "", content.rstrip()).rstrip()
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
                    f.write(f"\n[{format_timestamp(seg['start'])}] SPEAKER:\n")
                f.write(seg["text"] + " ")
        
        # Clean up trailing whitespace
        with open(self.output_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        content = re.sub(r"\[\d{2}:\d{2}:\d{2}\] SPEAKER:\s*$", "", content.rstrip()).rstrip()
        
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
# Main Transcription Logic
# ============================================================================

async def transcribe_file(
    input_path: str, 
    config: Config,
    client: TranscriptionClient,
    progress_bar: Optional[tqdm] = None
) -> Optional[str]:
    """
    Transcribe a single audio/video file.
    Returns path to output transcript file, or None on failure.
    """
    p = Path(input_path).resolve()
    
    if p.suffix.lower() not in config.supported_formats:
        log.warn(f"Unsupported format: {p.suffix}", file=p.name)
        return None
    
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
        diarize_segments = await client.diarize_path(temp_wav)

        if not diarize_segments:
            log.warn("Diarization returned no segments. Falling back to entire file as one speaker.", file=p.name)
            total_dur = get_total_duration(temp_wav)
            diarize_segments = [{"start": 0.0, "end": total_dur, "speaker": "SPEAKER"}]

        # Group segments to avoid tiny chunks
        segments = group_diarization_segments(diarize_segments, config.max_chunk_duration, config.merge_gap)
        log.info(f"Grouped into {len(segments)} speaker-homogeneous chunks")

        if not segments:
            log.warn("No valid segments left after grouping", file=p.name)
            return None
        
        # Initialize writer
        writer = TranscriptWriter(output_path, config.output_format)
        
        # Process in batches
        total_batches = (len(segments) + config.batch_size - 1) // config.batch_size

        if progress_bar is not None:
            progress_bar.total = len(segments)
            progress_bar.set_description(p.name[:30])

        # Clear output file if it exists and write format header if necessary
        with open(output_path, "w", encoding="utf-8") as f:
            pass

        for batch_idx in range(0, len(segments), config.batch_size):
            batch = segments[batch_idx:batch_idx + config.batch_size]
            chunk_files = []
            chunk_info = []  # (path, start, end)
            
            # Extract chunks
            for j, (start, end, speaker) in enumerate(batch):
                chunk_path = os.path.join(temp_dir, f"chunk_{batch_idx + j}.wav")
                duration = end - start

                if extract_chunk(temp_wav, start, duration, chunk_path, config.sample_rate):
                    if rms_check(chunk_path, config.rms_silence_threshold):
                        chunk_files.append(chunk_path)
                        chunk_info.append((chunk_path, start, end, speaker))
                    else:
                        # Silent chunk, skip
                        try:
                            os.remove(chunk_path)
                        except OSError:
                            pass

            if not chunk_files:
                if progress_bar is not None:
                    progress_bar.update(len(batch))
                continue

            # Transcribe batch
            try:
                results = await client.transcribe_batch(
                    [info[0] for info in chunk_info],
                    language="en"
                )

                for (_, start, end, speaker), result in zip(chunk_info, results):
                    if isinstance(result, dict):
                        text = result.get("text", "")
                        error = result.get("error")
                    else:
                        text = result
                        error = None

                    if error:
                        log.warn(f"Chunk error at {format_timestamp(start)}: {error}")
                        continue

                    if text.strip():
                        writer.add_segment(text, start, end, speaker=speaker)
                        if progress_bar is not None:
                            progress_bar.set_postfix_str(text[:40] + "...")

            except Exception as e:
                log.error(f"Batch transcription failed: {e}")
            
            finally:
                # Cleanup chunk files
                for cf in chunk_files:
                    try:
                        os.remove(cf)
                    except OSError:
                        pass
            
            if progress_bar is not None:
                progress_bar.update(len(batch))
        
        # Write output
        writer.write()
        log.success(f"Saved: {output_path}")
        return output_path
        
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


async def transcribe_files(input_paths: list[str], config: Config):
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
        
        for input_path in input_paths:
            with tqdm(unit="seg", leave=True, dynamic_ncols=True) as pbar:
                result = await transcribe_file(input_path, config, client, pbar)
                results.append((input_path, result))
        
        # Summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        
        success = sum(1 for _, r in results if r)
        failed = len(results) - success
        
        for input_path, output_path in results:
            status = "✓" if output_path else "✗"
            print(f"  {status} {Path(input_path).name}")
            if output_path:
                print(f"    → {output_path}")
        
        print(f"\nCompleted: {success}/{len(results)} files")
        if failed:
            print(f"Failed: {failed} files")

        overall_time = time.perf_counter() - overall_start_time
        print(f"Total transcription time: {overall_time:.2f} seconds")

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
        """
    )
    
    parser.add_argument(
        "files", 
        nargs="+", 
        help="Audio/video files to transcribe"
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
    asyncio.run(transcribe_files(input_files, config))


if __name__ == "__main__":
    main()