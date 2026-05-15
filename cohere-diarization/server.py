"""
Production-ready ASR transcription server using ONNX Runtime.

Features:
- Secure file upload and validated path access
- Async request handling with thread pool for inference
- Request validation and size limits
- Timeout protection
- KV cache pooling for memory efficiency
- Prometheus metrics
- Structured logging
- Optional API key authentication
- Rate limiting
"""

import os
import sys
import time
import threading
import tempfile
import signal
import re
from pathlib import Path
from typing import Optional
from functools import lru_cache
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio

from dotenv import load_dotenv
load_dotenv()

import torch
import torchaudio
import numpy as np
import json
import librosa
import onnxruntime as ort
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine

TARGET_SAMPLES = 480000  # 30 seconds at 16kHz

# Config and speaker modules
from config import get, is_debug
from speaker.matcher import match_clusters, merge_matched_clusters
from speaker.audio import extract_fbank, generate_sliding_windows, refine_speaker_boundaries
from speaker.vad import run_vad_chunked, run_vad_onnx, split_at_energy_dips
from speaker.profiling import profile_speakers, relabel_by_pitch

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

def ensure_embedding_model(repo_id: str, filename: str, token: str) -> str:
    """Downloads the ONNX embedding model from HF Hub if not present locally."""
    log.info("downloading_embedding_model", repo=repo_id, file=filename)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        return path
    except Exception as e:
        log.error("embedding_download_failed", error=str(e))
        raise


# ============================================================================
# Configuration
# ============================================================================

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model settings
    model_repo: str = "gn64/cohere-transcribe-onnx-int8"
    model_dir: Path = Path(__file__).parent.parent / "models"
    
    # Model architecture constants
    n_layers: int = 8
    heads: int = 8
    head_dim: int = 128
    max_ctx: int = 1024
    max_new_tokens: int = 448
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 2
    request_timeout: int = 120
    max_request_size_mb: int = 200
    
    # Batch settings
    max_batch_size: int = 10
    max_audio_duration_sec: int = 600  # 10 minutes max per file
    
    # Security settings
    allowed_audio_dir: Optional[Path] = None  # If set, only allow paths under this dir
    api_keys: Optional[str] = None  # Comma-separated API keys, None = no auth
    enable_cors: bool = True
    cors_origins: str = "*"
    
    # Feature flags
    enable_dml: bool = True
    enable_metrics: bool = True
    enable_rate_limit: bool = True
    enable_diarization: bool = True
    hf_token: Optional[str] = None
    rate_limit: str = "30/minute"
    
    # Diarization settings
    diarization_threshold: float = Field(default=0.35, description="Distance threshold for AgglomerativeClustering - higher = fewer clusters")
    vad_threshold: float = Field(default=0.5, description="Speech probability cutoff (0.0 to 1.0) for Silero VAD")
    vad_min_speech_duration_ms: int = Field(default=250, description="Minimum speech chunk length (ms) for Silero VAD")
    
    # Embedding model settings
    embedding_model_repo: str = Field(default="Wespeaker/wespeaker-ecapa-tdnn512-LM", description="HuggingFace repo for the embedding ONNX model")
    embedding_model_filename: str = Field(default="voxceleb_ECAPA512_LM.onnx", description="Filename of the ONNX embedding model")
    
    # Cache settings
    kv_cache_pool_size: int = 4

    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIBE_",
        env_file=".env",
        extra="ignore"
    )

    @property
    def api_key_set(self) -> set[str]:
        if not self.api_keys:
            return set()
        return set(k.strip() for k in self.api_keys.split(",") if k.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging():
    """Configure structured JSON logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if os.getenv("LOG_JSON") else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

setup_logging()
log = structlog.get_logger()


# ============================================================================
# Metrics (Prometheus)
# ============================================================================

class Metrics:
    """Prometheus metrics container."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return
            
        from prometheus_client import Counter, Histogram, Gauge
        
        self.requests_total = Counter(
            "transcribe_requests_total",
            "Total transcription requests",
            ["endpoint", "status"]
        )
        self.audio_duration = Histogram(
            "transcribe_audio_duration_seconds",
            "Duration of audio processed",
            buckets=[1, 5, 10, 30, 60, 120, 300, 600]
        )
        self.inference_time = Histogram(
            "transcribe_inference_seconds",
            "Time spent in inference",
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
        )
        self.tokens_generated = Histogram(
            "transcribe_tokens_generated",
            "Number of tokens generated",
            buckets=[10, 50, 100, 200, 300, 448]
        )
        self.active_requests = Gauge(
            "transcribe_active_requests",
            "Currently processing requests"
        )
        self.model_loaded = Gauge(
            "transcribe_model_loaded",
            "Whether the model is loaded (1) or not (0)"
        )
    
    def inc_request(self, endpoint: str, status: str):
        if self.enabled:
            self.requests_total.labels(endpoint=endpoint, status=status).inc()
    
    def observe_audio(self, duration: float):
        if self.enabled:
            self.audio_duration.observe(duration)
    
    def observe_inference(self, duration: float):
        if self.enabled:
            self.inference_time.observe(duration)
    
    def observe_tokens(self, count: int):
        if self.enabled:
            self.tokens_generated.observe(count)
    
    @contextmanager
    def track_request(self):
        if self.enabled:
            self.active_requests.inc()
        try:
            yield
        finally:
            if self.enabled:
                self.active_requests.dec()
    
    def set_model_loaded(self, loaded: bool):
        if self.enabled:
            self.model_loaded.set(1 if loaded else 0)
    
    def generate(self) -> bytes:
        if not self.enabled:
            return b""
        from prometheus_client import generate_latest
        return generate_latest()


# ============================================================================
# Custom Exceptions
# ============================================================================

class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class TimeoutError(TranscriptionError):
    """Inference timeout error."""
    pass


class AudioValidationError(TranscriptionError):
    """Invalid audio input error."""
    pass


class PathSecurityError(TranscriptionError):
    """Path access security error."""
    pass


# ============================================================================
# KV Cache Pool
# ============================================================================

class KVCachePool:
    """
    Pool of reusable KV caches to avoid repeated GPU memory allocations.
    Thread-safe implementation with automatic growth.
    """
    
    def __init__(self, settings: Settings, device: str = "cpu"):
        self.settings = settings
        self.device = device
        self.pool: list[dict] = []
        self.lock = threading.Lock()
        self.created_count = 0
        
        # Pre-allocate initial pool
        for _ in range(settings.kv_cache_pool_size):
            self.pool.append(self._create_cache())
    
    def _create_cache(self) -> dict:
        """Create a new KV cache pair."""
        self.created_count += 1
        s = self.settings
        return {
            "self_k": ort.OrtValue.ortvalue_from_numpy(
                np.zeros((s.n_layers, 1, s.heads, s.max_ctx, s.head_dim), dtype=np.float32),
                self.device, 0
            ),
            "self_v": ort.OrtValue.ortvalue_from_numpy(
                np.zeros((s.n_layers, 1, s.heads, s.max_ctx, s.head_dim), dtype=np.float32),
                self.device, 0
            ),
        }
    
    @contextmanager
    def acquire(self):
        """
        Acquire a KV cache from the pool.
        Creates a new one if pool is empty.
        Returns cache to pool when done.
        """
        with self.lock:
            if self.pool:
                cache = self.pool.pop()
            else:
                log.warning("kv_cache_pool_empty", created_total=self.created_count)
                cache = self._create_cache()
        
        try:
            yield cache
        finally:
            with self.lock:
                # Return to pool if under limit
                if len(self.pool) < self.settings.kv_cache_pool_size * 2:
                    self.pool.append(cache)


# ============================================================================
# Model State
# ============================================================================

class ModelState:
    """Thread-safe container for model state."""

    def __init__(self):
        self.encoder: Optional[ort.InferenceSession] = None
        self.decoder: Optional[ort.InferenceSession] = None
        self.embedding_session: Optional[ort.InferenceSession] = None
        self.vad_model = None
        self.get_speech_timestamps = None
        self.tokens: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self.pre_computed_prompt_ids: list[int] = []
        self.pre_computed_eos_id: int = -1
        self.pre_computed_prompt_array: Optional[np.ndarray] = None
        self.use_dml: bool = False
        self.device: str = "cpu"
        self.status: str = "initializing"
        self.kv_pool: Optional[KVCachePool] = None
        self.lock = threading.Lock()
    
    @property
    def is_ready(self) -> bool:
        return self.status == "ready"


state = ModelState()
metrics: Optional[Metrics] = None
executor: Optional[ThreadPoolExecutor] = None


# ============================================================================
# Model Loading
# ============================================================================

def ensure_model(settings: Settings) -> Path:
    """Download model files if not present."""
    from huggingface_hub import snapshot_download
    
    needed = [
        "cohere-encoder.int8.onnx",
        "cohere-encoder.int8.onnx.data",
        "cohere-decoder.int8.onnx",
        "tokens.txt",
    ]
    
    if all((settings.model_dir / f).exists() for f in needed):
        log.info("model_files_present", path=str(settings.model_dir))
        return settings.model_dir
    
    log.info("downloading_model", repo=settings.model_repo, size_gb=2.9)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id=settings.model_repo,
        allow_patterns=["*.onnx", "*.onnx.data", "tokens.txt"],
        local_dir=str(settings.model_dir),
    )
    
    log.info("model_download_complete")
    return settings.model_dir


def load_models(settings: Settings):
    """Load encoder and decoder models."""
    global metrics
    
    model_dir = ensure_model(settings)
    
    # Load vocabulary
    tokens: dict[int, str] = {}
    with open(model_dir / "tokens.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().rsplit(" ", 1)
            if len(parts) == 2:
                tokens[int(parts[1])] = parts[0]
    
    token_to_id = {v: k for k, v in tokens.items()}
    log.info("vocabulary_loaded", token_count=len(tokens))
    
    # Pre-compute prompt tokens for transcription
    prompt_tokens = [
        "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
        "<|en|>", "<|en|>", "<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>",
    ]
    pre_computed_prompt_ids = [token_to_id[t] for t in prompt_tokens if t in token_to_id]
    pre_computed_eos_id = token_to_id.get("<|endoftext|>", -1)
    pre_computed_prompt_array = np.array([pre_computed_prompt_ids], dtype=np.int64)
    
    # Determine execution providers
    # Per user request: Cohere ONNX runs best on CPU, not DirectML
    use_dml = False
    providers = ["CPUExecutionProvider"]
    log.info("execution_providers", providers=providers, dml_enabled=use_dml)
    
    # Session options
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = min(4, max(1, os.cpu_count() or 4))
    opts.intra_op_num_threads = min(4, max(1, os.cpu_count() or 4))
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Load models
    log.info("loading_encoder")
    encoder = ort.InferenceSession(
        str(model_dir / "cohere-encoder.int8.onnx"), 
        opts, 
        providers=providers
    )
    
    log.info("loading_decoder")
    decoder = ort.InferenceSession(
        str(model_dir / "cohere-decoder.int8.onnx"), 
        opts, 
        providers=providers
    )
    
    # Update state
    device = "dml" if use_dml else "cpu"
    
    state.encoder = encoder
    state.decoder = decoder
    state.tokens = tokens
    state.token_to_id = token_to_id
    state.pre_computed_prompt_ids = pre_computed_prompt_ids
    state.pre_computed_eos_id = pre_computed_eos_id
    state.pre_computed_prompt_array = pre_computed_prompt_array
    state.use_dml = use_dml
    state.device = device
    state.kv_pool = KVCachePool(settings, device)

    if settings.enable_diarization:
        log.info("loading_vad_model")
        
        # First try: ONNX with DirectML
        vad_onnx_path = str(Path(torch.hub.get_dir()) / "silero_vad" / "silero_vad.onnx")
        
        if Path(vad_onnx_path).exists():
            try:
                vad_providers = ["DMLExecutionProvider", "CPUExecutionProvider"]
                state.vad_session = ort.InferenceSession(vad_onnx_path, opts, providers=vad_providers)
                state.use_vad_onnx = True
                state.vad_model = None
                
                # Load utils for timestamps (they work with ONNX model too)
                _, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=True
                )
                state.get_speech_timestamps = utils[0]
                log.info("vad_model_loaded_onnx_dml")
            except Exception as e:
                log.warning("vad_onnx_dml_failed", error=str(e))
                state.vad_session = None
                state.use_vad_onnx = False
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                state.vad_model = model
                state.get_speech_timestamps = utils[0]
                log.info("vad_model_loaded_pytorch")
        else:
            # Fallback to PyTorch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            model = model.to(torch.device("cpu"))
            state.vad_model = model
            state.vad_session = None
            state.use_vad_onnx = False
            state.get_speech_timestamps = utils[0]
            log.info("vad_model_loaded_pytorch")

        log.info("loading_embedding_model")
        try:
            emb_path = ensure_embedding_model(
                settings.embedding_model_repo,
                settings.embedding_model_filename,
                settings.hf_token
            )
            state.embedding_session = ort.InferenceSession(emb_path, opts, providers=providers)
            log.info("embedding_model_loaded", path=emb_path)
        except Exception as e:
            log.error("embedding_model_failed", error=str(e))

    state.status = "ready"
    
    if metrics:
        metrics.set_model_loaded(True)
    
    log.info("model_ready", device=device)


# ============================================================================
# Inference
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


# ---------------------------------------------------------------------------
# Transcript hallucination cleaner
# ---------------------------------------------------------------------------
# Whisper-family models produce looping repetitions when fed laughter, noise,
# or silence: e.g. "the ones who are the ones who are ..." repeated hundreds
# of times.  Detect any phrase of 1-10 words that repeats 4+ times
# consecutively and replace the entire run with [inaudible].
_LOOP_RE = re.compile(
    r'\b(.{4,80}?)(?:\s+\1){3,}',   # phrase repeated ≥4 times
    re.IGNORECASE,
)

def clean_transcript(text: str) -> str:
    """Replace hallucinated looping repetitions with [inaudible]."""
    # Iterate: one pass may expose a shorter inner loop after the outer is removed
    prev = None
    while prev != text:
        prev = text
        text = _LOOP_RE.sub('[inaudible]', text)
    # Collapse multiple consecutive [inaudible] tags into one
    text = re.sub(r'(\[inaudible\]\s*){2,}', '[inaudible] ', text)
    return text.strip()


def transcribe_audio_sync(
    audio: np.ndarray, 
    language: str = "en",
    timeout_sec: int = 120
) -> dict:
    """
    Synchronous transcription function.
    Returns dict with text, token count, and timing info.
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
    
    if metrics:
        metrics.observe_audio(audio_duration)
    
    with inference_timeout(timeout_sec):
        encoder = state.encoder
        decoder = state.decoder
        tokens = state.tokens
        token_to_id = state.token_to_id
        device = state.device
        
        # Use pre-computed prompt for default language (en)
        prompt_ids = state.pre_computed_prompt_ids
        eos_id = state.pre_computed_eos_id
        
        # Run encoder
        enc_io = encoder.io_binding()
        enc_io.bind_cpu_input("audio", audio.reshape(1, -1).astype(np.float32))
        enc_io.bind_output("n_layer_cross_k", device)
        enc_io.bind_output("n_layer_cross_v", device)
        encoder.run_with_iobinding(enc_io)
        
        enc_out = enc_io.get_outputs()
        cross_k_ov = enc_out[0]
        cross_v_ov = enc_out[1]
        
        # Use pre-computed prompt and eos_id
        prompt_ids = state.pre_computed_prompt_ids
        eos_id = state.pre_computed_eos_id
        
        # Get KV cache from pool
        with state.kv_pool.acquire() as kv_cache:
            self_k_ov = kv_cache["self_k"]
            self_v_ov = kv_cache["self_v"]
            
            generated = list(prompt_ids)
            current = np.array([prompt_ids], dtype=np.int64)
            offset = np.array(0, dtype=np.int64)
            
            # Pre-allocate reusable arrays to avoid repeated allocation in loop
            current_buffer = np.zeros((1, 1), dtype=np.int64)
            
            dec_io = decoder.io_binding()
            
            for _ in range(settings.max_new_tokens):
                dec_io.bind_cpu_input("tokens", current)
                dec_io.bind_ortvalue_input("in_n_layer_self_k_cache", self_k_ov)
                dec_io.bind_ortvalue_input("in_n_layer_self_v_cache", self_v_ov)
                dec_io.bind_ortvalue_input("n_layer_cross_k", cross_k_ov)
                dec_io.bind_ortvalue_input("n_layer_cross_v", cross_v_ov)
                dec_io.bind_cpu_input("offset", offset)
                dec_io.bind_output("logits", device)
                dec_io.bind_output("out_n_layer_self_k_cache", device)
                dec_io.bind_output("out_n_layer_self_v_cache", device)
                
                decoder.run_with_iobinding(dec_io)
                
                dec_out = dec_io.get_outputs()
                logits = dec_out[0].numpy()
                
                # Handle potential scalar output from new torch version
                if logits.ndim == 0:
                    logits = logits.reshape(1, -1)
                elif logits.ndim == 1:
                    logits = logits.reshape(1, -1)
                
                self_k_ov = dec_out[1]
                self_v_ov = dec_out[2]
                
                next_id = int(np.argmax(logits[0, -1, :]))
                if next_id == eos_id:
                    break
                    
                generated.append(next_id)
                # offset is 0-d scalar for ONNX; use int() to extract value
                offset = np.array(int(offset) + current.shape[1], dtype=np.int64)
                current_buffer[0, 0] = next_id
                current = current_buffer
        
        # Decode text and clean hallucinated repetition loops
        text = "".join(
            tokens.get(t, "").replace("\u2581", " ")
            for t in generated[len(prompt_ids):]
            if not tokens.get(t, "").startswith("<|")
        ).strip()
        text = clean_transcript(text)

        tokens_generated = len(generated) - len(prompt_ids)
        inference_time = time.perf_counter() - start_time

        if metrics:
            metrics.observe_inference(inference_time)
            metrics.observe_tokens(tokens_generated)

        log.debug(
            "transcription_complete",
            audio_duration=f"{audio_duration:.2f}s",
            tokens=tokens_generated,
            inference_time=f"{inference_time:.2f}s"
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


# ============================================================================
# Request/Response Models
# ============================================================================


class DiarizeResult(BaseModel):
    start: float
    end: float
    speaker: str

class DiarizeResponse(BaseModel):
    segments: list[DiarizeResult]
    total_time_sec: float
    error: Optional[str] = None

class DiarizePathsRequest(BaseModel):
    wav_path: str = Field(..., description="Audio file path")
    num_speakers: Optional[int] = Field(None, description="Exact number of speakers (if known)")
    diarization_threshold: Optional[float] = Field(None, description="Distance threshold for clustering (overrides server default)")
    vad_threshold: Optional[float] = Field(None, description="VAD speech probability cutoff (0.0-1.0)")
    vad_min_speech_duration_ms: Optional[int] = Field(None, description="VAD minimum speech chunk length (ms)")
    known_speakers: Optional[dict[str, dict]] = Field(None, description="Map of known speaker names to their profiles")

class TranscribePathsRequest(BaseModel):
    """Request model for path-based transcription."""
    wav_paths: list[str] = Field(..., max_length=10, description="List of audio file paths")
    language: str = Field(default="en", pattern=r"^[a-z]{2}$", description="ISO 639-1 language code")
    
    @field_validator("wav_paths")
    @classmethod
    def validate_paths(cls, v):
        if not v:
            raise ValueError("At least one path required")
        return v


class TranscribeResult(BaseModel):
    """Single transcription result."""
    text: str
    audio_duration_sec: float
    inference_time_sec: float
    tokens_generated: int
    error: Optional[str] = None


class TranscribeResponse(BaseModel):
    """Transcription response model."""
    results: list[TranscribeResult]
    total_time_sec: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_status: str
    device: str
    version: str = "1.0.0"


# ============================================================================
# Security
# ============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    settings: Settings = Depends(get_settings)
) -> Optional[str]:
    """Verify API key if authentication is enabled."""
    if not settings.api_key_set:
        return None  # No auth required
    
    if not api_key or api_key not in settings.api_key_set:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key"
        )
    return api_key


def validate_path_security(path: str, settings: Settings) -> Path:
    """Validate that a path is allowed to be accessed."""
    resolved = Path(path).resolve()
    
    if not resolved.exists():
        raise PathSecurityError(f"File not found: {path}")
    
    if settings.allowed_audio_dir:
        allowed = settings.allowed_audio_dir.resolve()
        if not str(resolved).startswith(str(allowed)):
            raise PathSecurityError(
                f"Path not allowed. Must be under: {allowed}",
                details={"path": path}
            )
    
    # Check file extension
    allowed_extensions = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".webm"}
    if resolved.suffix.lower() not in allowed_extensions:
        raise PathSecurityError(
            f"File type not allowed: {resolved.suffix}",
            details={"path": path, "allowed": list(allowed_extensions)}
        )
    
    return resolved


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global metrics, executor
    
    settings = get_settings()
    
    # Initialize metrics
    metrics = Metrics(enabled=settings.enable_metrics)
    
    # Initialize thread pool
    executor = ThreadPoolExecutor(max_workers=settings.workers)
    
    # Load models
    log.info("starting_server", host=settings.host, port=settings.port)
    load_models(settings)
    
    yield
    
    # Cleanup
    log.info("shutting_down")
    if executor:
        executor.shutdown(wait=True)
    if metrics:
        metrics.set_model_loaded(False)
    state.status = "shutdown"


app = FastAPI(
    title="Transcription Server",
    description="Production ASR transcription service using Cohere model",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Global request middleware for logging and size limits."""
    settings = get_settings()
    request_id = request.headers.get("X-Request-ID", str(time.time_ns()))
    
    # Check request size
    content_length = request.headers.get("content-length")
    max_size = settings.max_request_size_mb * 1024 * 1024
    if content_length and int(content_length) > max_size:
        return JSONResponse(
            status_code=413,
            content={"error": f"Request too large. Max: {settings.max_request_size_mb}MB"}
        )
    
    # Log request
    start = time.perf_counter()
    
    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        
        log.info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=f"{elapsed*1000:.1f}"
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed*1000:.1f}ms"
        return response
        
    except Exception as e:
        log.exception("request_error", request_id=request_id, error=str(e))
        raise


# CORS middleware
settings = get_settings()
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Rate limiting (optional)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    RATE_LIMIT_AVAILABLE = False
    limiter = None
    log.warning("rate_limiting_unavailable", reason="slowapi not installed")


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(TranscriptionError)
async def transcription_error_handler(request: Request, exc: TranscriptionError):
    """Handle transcription-specific errors."""
    if metrics:
        metrics.inc_request(request.url.path, "error")
    
    return JSONResponse(
        status_code=400 if isinstance(exc, (AudioValidationError, PathSecurityError)) else 500,
        content={
            "error": exc.message,
            "type": type(exc).__name__,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    if metrics:
        metrics.inc_request(request.url.path, "error")
    
    log.exception("unhandled_error", path=request.url.path)
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="online" if state.is_ready else "degraded",
        model_status=state.status,
        device=state.device
    )


async def delayed_shutdown():
    await asyncio.sleep(1.0)
    os._exit(0)


@app.post("/shutdown")
async def shutdown(_: str = Depends(verify_api_key)):
    """Shutdown the server."""
    log.info("shutdown_requested")
    asyncio.create_task(delayed_shutdown())
    return {"status": "shutting down"}


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if not metrics or not metrics.enabled:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    return PlainTextResponse(metrics.generate(), media_type="text/plain")


def get_rate_limit_decorator():
    """Get rate limit decorator if available."""
    settings = get_settings()
    if RATE_LIMIT_AVAILABLE and settings.enable_rate_limit and limiter:
        return limiter.limit(settings.rate_limit)
    return lambda f: f  # No-op decorator








@app.post("/diarize/path")
async def diarize_path_endpoint(
    req: DiarizePathsRequest,
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_api_key)
):
    """
    Streams Pyannote diarization progress as NDJSON, then yields the final segments.
    """
    start_time = time.perf_counter()
    if not state.vad_model or not state.embedding_session:
        return JSONResponse(
            status_code=400,
            content={"error": "Diarization not enabled or pipeline not loaded"}
        )

    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    try:
        resolved = validate_path_security(req.wav_path, settings)
    except PathSecurityError as e:
        return JSONResponse(status_code=403, content={"error": f"Access denied: {e.message}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    def run_diarization_thread():
        try:
            waveform, sr = librosa.load(str(resolved), sr=16000, mono=True)
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()

            if not state.vad_model or not state.get_speech_timestamps or not state.embedding_session:
                log.error("diarization_models_missing")
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "error": "Diarization models not fully loaded"
                }))
                return

            # 1. Run VAD
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "progress", "step": "Voice Activity Detection", "completed": 0, "total": 1
            }))

            vad_thresh_val = req.vad_threshold if req.vad_threshold is not None else settings.vad_threshold
            vad_min_dur_val = req.vad_min_speech_duration_ms if req.vad_min_speech_duration_ms is not None else settings.vad_min_speech_duration_ms

            audio_duration = waveform_tensor.shape[-1] / 16000
            
            # Use ONNX VAD with DirectML if available, otherwise PyTorch
            if getattr(state, 'use_vad_onnx', False) and getattr(state, 'vad_session', None):
                log.info("vad_using_onnx_dml", duration=audio_duration)
                speech_ts = run_vad_onnx(
                    waveform_tensor,
                    state.vad_session,
                    sample_rate=16000,
                    chunk_duration=30,
                    overlap=5,
                    threshold=vad_thresh_val,
                    min_speech_duration_ms=vad_min_dur_val
                )
            elif audio_duration > 60:
                log.info("vad_using_chunked", duration=audio_duration)
                speech_ts = run_vad_chunked(
                    waveform_tensor,
                    state.vad_model,
                    state.get_speech_timestamps,
                    sample_rate=16000,
                    chunk_duration=30,
                    overlap=5,
                    threshold=vad_thresh_val,
                    min_speech_duration_ms=vad_min_dur_val
                )
            else:
                speech_ts = state.get_speech_timestamps(
                    waveform_tensor,
                    state.vad_model,
                    sampling_rate=16000,
                    return_seconds=True,
                    threshold=vad_thresh_val,
                    min_speech_duration_ms=vad_min_dur_val
                )

            if not speech_ts:
                log.info("vad_found_no_speech", path=req.wav_path)
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "type": "result", "segments": [], "profiles": {}
                }))
                loop.call_soon_threadsafe(queue.put_nowait, None)
                return

            # 1b. Split long VAD segments at energy dips to reduce cross-speaker
            #     window contamination. When speakers alternate with minimal silence,
            #     VAD treats the whole exchange as one segment. Sliding embedding
            #     windows that straddle the boundary produce blended embeddings that
            #     confuse clustering and cause the last words of one speaker to spill
            #     into the next. Splitting at energy dips before window extraction
            #     keeps each window within a single speaker turn.
            waveform_np = waveform_tensor.squeeze(0).numpy()
            speech_ts = split_at_energy_dips(
                speech_ts,
                waveform_np,
                sample_rate=16000,
            )
            log.debug("vad_after_energy_split", n_segments=len(speech_ts))

            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "progress", "step": "Voice Activity Detection", "completed": 1, "total": 1
            }))

            # 2. Extract chunks and features
            # Windows shorter than MIN_EMBED_DURATION produce unreliable embeddings
            # (backchannels: "Mm-hmm", "Okay", etc.). We cluster only long windows
            # and assign short ones to their nearest long neighbour by time.
            MIN_EMBED_DURATION = 1.5  # seconds

            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "progress", "step": "Feature Extraction", "completed": 0, "total": len(speech_ts)
            }))

            all_fbanks = []
            all_segments_meta = []   # metadata for ALL windows
            embeddable_indices = []  # indices into all_segments_meta that have an embedding

            for i, ts in enumerate(speech_ts):
                start_sample = int(ts['start'] * 16000)
                end_sample = int(ts['end'] * 16000)
                segment_wav = waveform_tensor[:, start_sample:end_sample]

                windows, start_times = generate_sliding_windows(segment_wav, 16000, window_sec=2.0, stride_sec=0.75)

                for w, rel_start in zip(windows, start_times):
                    chunk_duration = w.shape[-1] / 16000
                    global_start = ts['start'] + rel_start
                    global_end = global_start + chunk_duration
                    meta_idx = len(all_segments_meta)
                    all_segments_meta.append({"start": global_start, "end": global_end})

                    if chunk_duration >= MIN_EMBED_DURATION:
                        if w.shape[-1] < 1600:
                            w = torch.nn.functional.pad(w, (0, 1600 - w.shape[-1]))
                        all_fbanks.append(extract_fbank(w, 16000))
                        embeddable_indices.append(meta_idx)

                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "type": "progress", "step": "Feature Extraction", "completed": i + 1, "total": len(speech_ts)
                }))

            

            if not all_fbanks:
                fallback_segments = []
                for ts in speech_ts:
                    fallback_segments.append({
                        "start": ts["start"],
                        "end": ts["end"],
                        "speaker": "SPEAKER1"
                    })
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "type": "result", "segments": fallback_segments, "profiles": {}
                }))
                loop.call_soon_threadsafe(queue.put_nowait, None)
                return

            # 3. ONNX Embedding Extraction (Batched) — long windows only
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "progress", "step": "Embedding Extraction", "completed": 0, "total": 1
            }))

            # Apply CMN per sub-segment (critical for ECAPA-TDNN)
            # Vectorized: pad first, then stack and apply CMN
            if all_fbanks:
                max_len = max(fb.shape[1] for fb in all_fbanks)
                
                # Pad each fb to max_len BEFORE stacking
                padded_fbanks = []
                for fb in all_fbanks:
                    if fb.shape[1] < max_len:
                        fb_padded = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
                    else:
                        fb_padded = fb
                    padded_fbanks.append(fb_padded)
                
                batch = torch.stack(padded_fbanks, dim=0)  # [N, 1, max_len, 80]
                cmn_batch = batch - batch.mean(dim=2, keepdim=True)  # CMN on all at once
                
                batch_fbanks = cmn_batch.squeeze(1).numpy()  # [N, max_len, 80]
            else:
                batch_fbanks = np.array([])

            raw_embeddings = []
            batch_size = 32
            for i in range(0, len(batch_fbanks), batch_size):
                out = state.embedding_session.run(None, {"feats": batch_fbanks[i:i+batch_size]})
                raw_embeddings.append(out[0])

            raw_embeddings = np.concatenate(raw_embeddings, axis=0)  # [N_long, D]
            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            raw_embeddings = raw_embeddings / np.maximum(norms, 1e-12)

            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "progress", "step": "Embedding Extraction", "completed": 1, "total": 1
            }))

            # 4. Clustering on long windows only
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "progress", "step": "Clustering", "completed": 0, "total": 1
            }))

            n_clusters_val = req.num_speakers
            # Only use threshold if num_speakers is not specified
            if n_clusters_val is not None:
                dist_thresh_val = None  # Force exact number of clusters
            else:
                dist_thresh_val = req.diarization_threshold if req.diarization_threshold is not None else settings.diarization_threshold

            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters_val,
                metric="cosine",
                linkage="average",
                distance_threshold=dist_thresh_val
            )

            if len(raw_embeddings) > 1:
                long_labels = clusterer.fit_predict(raw_embeddings)
            else:
                long_labels = np.array([0])
            
            n_clusters = len(set(int(l) for l in long_labels))

            # Cap at 15 clusters to preserve granularity before merging
            max_clusters = 15
            if n_clusters > max_clusters:
                print(f"WARNING: {n_clusters} clusters created, capping at {max_clusters}")
                # Re-cluster with exact number
                clusterer = AgglomerativeClustering(
                    n_clusters=max_clusters,
                    metric="cosine",
                    linkage="average"
                )
                if len(raw_embeddings) > 1:
                    long_labels = clusterer.fit_predict(raw_embeddings)
                n_clusters = max_clusters
            
            # Voiceprint-based speaker merging
            # SKIP if user forced exact speaker count
            if n_clusters > 1 and n_clusters_val is None:
                cluster_ids = sorted(set(int(l) for l in long_labels))
                cluster_avgs = {}
                for cid in cluster_ids:
                    mask = long_labels == cid
                    cluster_avgs[cid] = np.mean(raw_embeddings[mask], axis=0)
                
                # Greedy merge: merge closest pairs below threshold
                merge_threshold = 0.25  # Cosine distance threshold for voiceprint merging - P90 of earnings22 distances
            
                # Debug: print pairwise distances
                ids = sorted(cluster_avgs.keys())
                for i_idx in range(len(ids)):
                    for j_idx in range(i_idx + 1, len(ids)):
                        id_i, id_j = ids[i_idx], ids[j_idx]
                        dist = cosine_distances(
                            [cluster_avgs[id_i]], [cluster_avgs[id_j]]
                        )[0][0]
            
                changed = True
                while changed:
                    changed = False
                    ids = sorted(cluster_avgs.keys())
                    for i_idx in range(len(ids)):
                        for j_idx in range(i_idx + 1, len(ids)):
                            id_i, id_j = ids[i_idx], ids[j_idx]
                            if id_i not in cluster_avgs:
                                continue
                            if id_j not in cluster_avgs:  # Already merged
                                continue
                            dist = cosine_distances(
                                [cluster_avgs[id_i]], [cluster_avgs[id_j]]
                            )[0][0]
                            if dist < merge_threshold:
                                # Merge j into i
                                long_labels[long_labels == id_j] = id_i
                                # Update average embedding
                                mask_i = long_labels == id_i
                                cluster_avgs[id_i] = np.mean(raw_embeddings[mask_i], axis=0)
                                del cluster_avgs[id_j]
                                changed = True
                                break
                    if changed:
                        break
            
                n_clusters = len(set(int(l) for l in long_labels))

            # Compute normalized centroid vector for each raw cluster ID
            cluster_centroids = {}
            for cluster_id in set(long_labels):
                mask = (long_labels == cluster_id)
                mean_emb = raw_embeddings[mask].mean(axis=0)
                norm_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
                cluster_centroids[int(cluster_id)] = norm_emb.tolist()

            # Assign cluster labels to embeddable windows
            for idx, label in zip(embeddable_indices, long_labels):
                all_segments_meta[idx]["speaker_raw"] = int(label)

            # Assign short windows to the nearest embeddable window by midpoint
            emb_mids = np.array([
                (all_segments_meta[i]["start"] + all_segments_meta[i]["end"]) / 2
                for i in embeddable_indices
            ])
            for i, seg in enumerate(all_segments_meta):
                if "speaker_raw" not in seg:
                    mid = (seg["start"] + seg["end"]) / 2
                    nearest = int(np.argmin(np.abs(emb_mids - mid)))
                    seg["speaker_raw"] = all_segments_meta[embeddable_indices[nearest]]["speaker_raw"]

            # Map raw cluster IDs to speaker names (with known speaker matching)
            if req.known_speakers:
                log.info("known_speakers_received", keys=list(req.known_speakers.keys()))
                
                # Temporarily populate raw cluster IDs as speaker names
                for seg in all_segments_meta:
                    seg["speaker"] = f"RAW_{seg['speaker_raw']}"
                
                # Match RAW profiles to known speakers
                speaker_map: dict[str, str] = {}
                unknown_idx = 1
                match_thresh = 0.03  # Stricter - only match if very similar
                close_match_thresh = 0.1  # If two clusters match same known speaker within this, merge them

                # First pass: match all clusters
                cluster_matches = {}
                for raw_spk, centroid in cluster_centroids.items():
                    raw_name = f"RAW_{raw_spk}"
                    best_match = None
                    best_dist = float('inf')

                    for known_name, known_prof in req.known_speakers.items():
                        if "embedding" not in known_prof:
                            continue
                        emb_dist = cosine(centroid, known_prof["embedding"])
                        
                        if emb_dist < best_dist:
                            best_dist = emb_dist
                            best_match = known_name

                    cluster_matches[raw_spk] = {"name": best_match, "dist": best_dist, "centroid": centroid}

                # Second pass: detect closely matching clusters to same known speaker
                # If two clusters both match the same known speaker with very small distance, merge them
                merged_clusters = set()
                for raw_spk1, match1 in cluster_matches.items():
                    if raw_spk1 in merged_clusters:
                        continue
                    if not match1["name"] or match1["dist"] > close_match_thresh:
                        continue
                    
                    for raw_spk2, match2 in cluster_matches.items():
                        if raw_spk2 == raw_spk1 or raw_spk2 in merged_clusters:
                            continue
                        if not match2["name"] or match2["dist"] > close_match_thresh:
                            continue
                        
                        # Both closely match known speakers - check if they're the same
                        if match1["name"] == match2["name"]:
                            # Same known speaker, very close matches - merge
                            dist_between = cosine(match1["centroid"], match2["centroid"])
                            if dist_between < 0.2:
                                long_labels[long_labels == raw_spk2] = raw_spk1
                                merged_clusters.add(raw_spk2)

                # Recompute cluster centroids after merging
                cluster_centroids = {}
                for cluster_id in set(long_labels):
                    mask = (long_labels == cluster_id)
                    mean_emb = raw_embeddings[mask].mean(axis=0)
                    norm_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
                    cluster_centroids[int(cluster_id)] = norm_emb.tolist()

                # Third pass: assign final names
                for raw_spk, centroid in cluster_centroids.items():
                    raw_name = f"RAW_{raw_spk}"
                    best_match = None
                    best_dist = float('inf')

                    for known_name, known_prof in req.known_speakers.items():
                        if "embedding" not in known_prof:
                            continue
                        emb_dist = cosine(centroid, known_prof["embedding"])
                        
                        if emb_dist < best_dist:
                            best_dist = emb_dist
                            best_match = known_name

                    if best_match and best_dist <= match_thresh:
                        speaker_map[raw_name] = best_match
                    else:
                        speaker_map[raw_name] = f"SPEAKER{unknown_idx}"
                        unknown_idx += 1

                # Update segments with final speaker names
                for seg in all_segments_meta:
                    raw_name = f"RAW_{seg['speaker_raw']}"
                    seg["speaker"] = speaker_map.get(raw_name, seg["speaker"])
                
                # Extract profiles for known speakers
                profiles = {}
                for raw_name, matched_name in speaker_map.items():
                    if matched_name.startswith("SPEAKER"):
                        continue
                    if matched_name in req.known_speakers:
                        profiles[matched_name] = req.known_speakers[matched_name]
                
                if profiles:
                    loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                        "type": "progress", "step": "Speaker Profiling", "completed": 1, "total": 1
                    }))
            else:
                # Original logic: Map raw cluster int -> SPEAKER1, SPEAKER2, ... in first-appearance order
                speaker_map: dict[int, str] = {}
                for seg in sorted(all_segments_meta, key=lambda x: x["start"]):
                    raw = seg["speaker_raw"]
                    if raw not in speaker_map:
                        speaker_map[raw] = f"SPEAKER{len(speaker_map) + 1}"
                    seg["speaker"] = speaker_map[raw]

            # 5. Merge contiguous same-speaker windows
            #    Split on speaker change or gap > MAX_SPEAKER_GAP — no hard time cap.
            MAX_SPEAKER_GAP = 1.0  # seconds (reduced from 2.0 to avoid over-merging)
            all_segments_meta.sort(key=lambda x: x["start"])
            merged_segments = []
            current_segment = None

            for seg in all_segments_meta:
                if current_segment is None:
                    current_segment = seg.copy()
                elif (current_segment["speaker"] == seg["speaker"] and
                      seg["start"] <= current_segment["end"] + MAX_SPEAKER_GAP):
                    current_segment["end"] = max(current_segment["end"], seg["end"])
                else:
                    if seg["start"] < current_segment["end"]:
                        mid = (seg["start"] + current_segment["end"]) / 2
                        current_segment["end"] = mid
                        seg = dict(seg, start=mid)
                    merged_segments.append(current_segment)
                    current_segment = seg.copy()

            if current_segment:
                merged_segments.append(current_segment)

            # 6. Post-merge: absorb short isolated segments into surrounding speaker
            #    If a segment is < MIN_ISLAND_DUR and is surrounded on both sides by the
            #    same speaker, reassign it to that speaker and re-merge.
            MIN_ISLAND_DUR = 1.0  # seconds (reduced from 2.1 to preserve short turns)
            changed = True
            while changed:
                changed = False
                for i in range(1, len(merged_segments) - 1):
                    seg = merged_segments[i]
                    prev_spk = merged_segments[i-1]["speaker"]
                    next_spk = merged_segments[i+1]["speaker"]
                    dur = seg["end"] - seg["start"]
                    if dur < MIN_ISLAND_DUR and prev_spk == next_spk and seg["speaker"] != prev_spk:
                        seg["speaker"] = prev_spk
                        changed = True
                if changed:
                    new_merged = []
                    cur = None
                    for seg in merged_segments:
                        if cur is None:
                            cur = seg.copy()
                        elif cur["speaker"] == seg["speaker"]:
                            cur["end"] = max(cur["end"], seg["end"])
                        else:
                            new_merged.append(cur)
                            cur = seg.copy()
                    if cur:
                        new_merged.append(cur)
                    merged_segments = new_merged

            # 6b. Boundary refinement — re-examine each speaker transition at fine
            #     granularity (0.5s sub-windows, 0.1s stride) to push the boundary
            #     to the exact point where the dominant speaker switches, correcting
            #     the last-words spill-over caused by wide sliding windows.
            if len(merged_segments) >= 2 and state.embedding_session:
                # Build per-named-speaker centroids from the embeddings already computed.
                # embeddable_indices[k] -> meta index; raw_embeddings[k] -> embedding.
                spk_emb_lists: dict[str, list] = {}
                for k, meta_idx in enumerate(embeddable_indices):
                    spk = all_segments_meta[meta_idx].get("speaker")
                    if spk:
                        spk_emb_lists.setdefault(spk, []).append(raw_embeddings[k])

                named_centroids_np: dict[str, np.ndarray] = {}
                for spk, embs in spk_emb_lists.items():
                    avg = np.mean(embs, axis=0)
                    named_centroids_np[spk] = avg / (np.linalg.norm(avg) + 1e-12)

                if named_centroids_np:
                    merged_segments = refine_speaker_boundaries(
                        merged_segments,
                        waveform_tensor,
                        state.embedding_session,
                        named_centroids_np,
                        sample_rate=16000,
                    )
                    log.debug("boundary_refinement_done", n_segments=len(merged_segments))

            # 7. Speaker profiling — extract voice signatures per cluster
            profiles = profile_speakers(waveform_tensor, merged_segments, sample_rate=16000)

            # Re-label by pitch so SPEAKER1 = lowest pitch (stable across runs)
            merged_segments, profiles = relabel_by_pitch(merged_segments, profiles)

            # Apply known speaker matching AFTER relabeling
            # Only match if we're very confident (distance < 0.1 = confidence > 0.8)
            if req.known_speakers:
                # Compute centroid for each final speaker
                speaker_centroids = {}
                for seg in merged_segments:
                    spk = seg["speaker"]
                    if spk not in speaker_centroids:
                        speaker_centroids[spk] = []
                    
                    # Find embedding for this segment's time range
                    seg_start = seg["start"]
                    seg_end = seg["end"]
                    for i, meta in enumerate(all_segments_meta):
                        if meta["start"] >= seg_start - 0.1 and meta["end"] <= seg_end + 0.1:
                            if "speaker_raw" in meta:
                                raw_id = meta["speaker_raw"]
                                if raw_id in cluster_centroids:
                                    speaker_centroids[spk].append(np.array(cluster_centroids[raw_id]))
                
                # Match each speaker to known voiceprints
                # Use config for threshold
                match_thresh = get("matching", "accept_threshold", 0.35)

                if is_debug():
                    print(f"=== DEBUG SPEAKER MATCHING ===")
                    print(f"Cluster speakers: {list(speaker_centroids.keys())}")
                    print(f"Known speakers: {list(req.known_speakers.keys()) if req.known_speakers else 'None'}")

                # Compute all possible matches for each cluster speaker
                all_matches = {}  # spk -> [(name, distance, confidence), ...]
                debug_matches = {}

                # Cache config values before loops to avoid repeated lookups
                norm_cfg = get("normalization", default={})
                pitch_per_unit = norm_cfg.get("pitch_hz_per_unit", 50)
                energy_per_unit = norm_cfg.get("energy_rms_per_unit", 0.05)
                conf_max_dist = norm_cfg.get("confidence_max_distance", 0.5)
                weights = get("weights", default={})
                w_emb = weights.get("embedding", 0.7)
                w_pitch = weights.get("pitch", 0.2)
                w_energy = weights.get("energy", 0.1)
                matching_cfg = get("matching", default={})
                embed_only_thresh = matching_cfg.get("embed_only_threshold", 0.16)

                for spk, emb_list in speaker_centroids.items():
                    if not emb_list:
                        continue

                    centroid = np.mean(emb_list, axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid = centroid / norm

                    # Get cluster's pitch and energy from profiles
                    cluster_pitch = profiles.get(spk, {}).get("pitch_hz", 0) or 0
                    cluster_energy = profiles.get(spk, {}).get("energy_rms", 0) or 0

                    matches = []
                    debug_distances = {}
                    for known_name, known_prof in req.known_speakers.items():
                        if "embedding" not in known_prof:
                            continue

                        # Embedding distance
                        emb_dist = cosine(centroid, known_prof["embedding"])

                        # Pitch distance (normalize to 0-1 range) - using cached config values
                        known_pitch = known_prof.get("pitch_hz", 0) or 0
                        pitch_dist = abs(cluster_pitch - known_pitch) / pitch_per_unit if cluster_pitch > 0 and known_pitch > 0 else 0.5

                        # Energy distance (normalize) - using cached config values
                        known_energy = known_prof.get("energy_rms", 0) or 0
                        energy_dist = abs(cluster_energy - known_energy) / energy_per_unit if cluster_energy > 0 and known_energy > 0 else 0.5

                        # Combined distance: use weighted combination always (using cached weights)
                        combined_dist = w_emb * emb_dist + w_pitch * min(pitch_dist, 1.0) + w_energy * min(energy_dist, 1.0)

                        conf = max(0, 1 - (combined_dist / conf_max_dist))
                        matches.append((known_name, combined_dist, conf))
                        debug_distances[known_name] = {
                            "emb_dist": round(float(emb_dist), 3),
                            "pitch_dist": round(pitch_dist, 3),
                            "energy_dist": round(energy_dist, 3),
                            "combined": round(combined_dist, 3),
                            "conf": round(conf, 3)
                        }

                    # Sort by distance ascending (best match = lowest distance)
                    matches.sort(key=lambda x: x[1])
                    all_matches[spk] = matches
                    debug_matches[spk] = debug_distances

                    if is_debug():
                        print(f"Cluster '{spk}' (pitch={cluster_pitch:.0f}Hz, energy={cluster_energy:.3f}) matches: {debug_distances}")

                if is_debug():
                    print("=== END DEBUG ===")

                # Apply best match if distance below threshold, store alternatives
                for spk, emb_list in speaker_centroids.items():
                    if not emb_list or spk not in all_matches:
                        continue

                    matches = all_matches[spk]
                    if not matches:
                        continue

                    best_match, best_dist, best_conf = matches[0]
                    
                    # Check if it's a clear winner
                    clear_winner = True
                    matching_cfg = get("matching", default={})
                    gap_threshold = matching_cfg.get("clear_winner_gap", 0.02)
                    embed_only_thresh = matching_cfg.get("embed_only_threshold", 0.16)
                    
                    if len(matches) > 1:
                        second_dist = matches[1][1]
                        gap = second_dist - best_dist
                        if gap < gap_threshold:
                            # When gap is small, prefer speaker with more training data
                            first_dur = req.known_speakers.get(matches[0][0], {}).get("total_speech_sec", 0)
                            second_dur = req.known_speakers.get(matches[1][0], {}).get("total_speech_sec", 0)
                            
                            # If best has significantly more data, use it despite small gap
                            if first_dur > second_dur * 2 and best_dist < embed_only_thresh:
                                clear_winner = True
                                if is_debug():
                                    print(f"  -> Using {matches[0][0]} (more training data: {first_dur:.0f}s vs {second_dur:.0f}s)")
                            else:
                                clear_winner = False

                    # Debug: show why this match was chosen
                    if is_debug():
                        print(f"Cluster '{spk}' -> best_match='{best_match}' best_dist={best_dist:.3f} match_thresh={match_thresh:.3f} clear_winner={clear_winner} -> {'MATCH' if best_dist <= match_thresh and clear_winner else 'NO MATCH'}")

                    # Store alternatives in each segment
                    alternatives = []
                    for name, dist, conf in matches[1:4]:  # Top 3 alternatives
                        conf_thresh = matching_cfg.get("confidence_threshold", 0.3)
                        if conf >= conf_thresh:
                            alternatives.append({"speaker": name, "confidence": round(conf, 2)})

                    if best_match and best_dist <= match_thresh and clear_winner:
                        # Update all segments with this speaker
                        for seg in merged_segments:
                            if seg["speaker"] == spk:
                                seg["speaker"] = best_match
                                seg["alternatives"] = alternatives
                                # Store debug for logging and output
                                seg["debug"] = {"from_cluster": spk, "confidence": best_conf, "distances": debug_matches[spk]}
                        # Update profiles
                        profiles[best_match] = profiles.pop(spk)
                        profiles[best_match]["matched_from"] = spk
                        profiles[best_match]["match_confidence"] = best_conf

            # Post-match merging: combine clusters that both matched to the same speaker
            # This handles cases where one speaker's voice was split into multiple clusters
            speaker_to_clusters = {}
            for spk, matches_list in all_matches.items():
                if not matches_list:
                    continue
                best_match, best_dist, _ = matches_list[0]
                if best_match and best_dist <= match_thresh:
                    if best_match not in speaker_to_clusters:
                        speaker_to_clusters[best_match] = []
                    speaker_to_clusters[best_match].append(spk)

            # Merge clusters with same speaker match
            for speaker, cluster_list in speaker_to_clusters.items():
                if len(cluster_list) > 1:
                    if is_debug():
                        print(f"MERGING: {cluster_list} -> {speaker}")
                    primary = cluster_list[0]
                    for extra in cluster_list[1:]:
                        # Update all segments from extra cluster to primary speaker
                        for seg in merged_segments:
                            if seg["speaker"] == extra:
                                seg["speaker"] = speaker
                        # Merge profiles
                        if extra in profiles and primary in profiles:
                            profiles[primary]["speech_sec"] = profiles[primary].get("speech_sec", 0) + profiles[extra].get("speech_sec", 0)
                            profiles[primary]["segment_count"] = profiles[primary].get("segment_count", 0) + profiles[extra].get("segment_count", 0)
                            del profiles[extra]

            # Additional merge: clusters with similar distance profiles to known speakers
            # This catches cases where two clusters don't both meet threshold but are same speaker
            cluster_ids = list(all_matches.keys())
            merged_already = set()
            for i in range(len(cluster_ids)):
                if cluster_ids[i] in merged_already:
                    continue
                for j in range(i + 1, len(cluster_ids)):
                    if cluster_ids[j] in merged_already:
                        continue
                    # Compare distance vectors
                    matches_i = all_matches[cluster_ids[i]]
                    matches_j = all_matches[cluster_ids[j]]
                    if not matches_i or not matches_j:
                        continue
                    # Build distance vectors for same speakers
                    dist_vec_i = {m[0]: m[1] for m in matches_i}
                    dist_vec_j = {m[0]: m[1] for m in matches_j}
                    common_speakers = set(dist_vec_i.keys()) & set(dist_vec_j.keys())
                    if len(common_speakers) >= 3:
                        # Check if distances are similar (within 0.05)
                        max_diff = max(abs(dist_vec_i[s] - dist_vec_j[s]) for s in common_speakers)
                        if max_diff < 0.05:
                            print(f"PROFILE MERGE: {cluster_ids[i]} and {cluster_ids[j]} (similar profiles)")
                            # Merge j into i
                            target_spk = cluster_ids[i]
                            for seg in merged_segments:
                                if seg["speaker"] == cluster_ids[j]:
                                    seg["speaker"] = target_spk
                            if cluster_ids[j] in profiles and target_spk in profiles:
                                profiles[target_spk]["speech_sec"] = profiles[target_spk].get("speech_sec", 0) + profiles[cluster_ids[j]].get("speech_sec", 0)
                                profiles[target_spk]["segment_count"] = profiles[target_spk].get("segment_count", 0) + profiles[cluster_ids[j]].get("segment_count", 0)
                                del profiles[cluster_ids[j]]
                            merged_already.add(cluster_ids[j])

            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "progress", "step": "Clustering", "completed": 1, "total": 1
            }))

            # ------------------------------------------------------------------ #
            # Ghost-speaker elimination: speakers with < 2s total speech are      #
            # almost always mis-clustered boundary fragments. For each such        #
            # segment, try the stored alternative speakers first; if none pass     #
            # the match threshold, fall back to the temporally adjacent speaker.   #
            # ------------------------------------------------------------------ #
            ghost_threshold_sec = 10.0
            speaker_total = {}
            for seg in merged_segments:
                spk = seg["speaker"]
                speaker_total[spk] = speaker_total.get(spk, 0.0) + (seg["end"] - seg["start"])

            ghost_speakers = {spk for spk, total in speaker_total.items() if total < ghost_threshold_sec}

            if ghost_speakers:
                if is_debug():
                    print(f"GHOST SPEAKERS (< {ghost_threshold_sec}s total): {ghost_speakers}")

                for seg in merged_segments:
                    if seg["speaker"] not in ghost_speakers:
                        continue

                    # Try stored alternatives first (already sorted by confidence)
                    reassigned = False
                    for alt in seg.get("alternatives", []):
                        alt_spk = alt["speaker"]
                        if alt_spk not in ghost_speakers:
                            if is_debug():
                                print(f"  GHOST REASSIGN {seg['start']:.1f}-{seg['end']:.1f}: "
                                      f"{seg['speaker']} -> {alt_spk} (alt conf={alt['confidence']:.2f})")
                            seg["speaker"] = alt_spk
                            reassigned = True
                            break

                    if not reassigned:
                        # Fall back to nearest non-ghost neighbour in time
                        seg_mid = (seg["start"] + seg["end"]) / 2
                        best_neighbor = None
                        best_dist_t = float("inf")
                        for other in merged_segments:
                            if other is seg or other["speaker"] in ghost_speakers:
                                continue
                            other_mid = (other["start"] + other["end"]) / 2
                            d = abs(other_mid - seg_mid)
                            if d < best_dist_t:
                                best_dist_t = d
                                best_neighbor = other["speaker"]
                        if best_neighbor:
                            if is_debug():
                                print(f"  GHOST NEIGHBOUR {seg['start']:.1f}-{seg['end']:.1f}: "
                                      f"{seg['speaker']} -> {best_neighbor}")
                            seg["speaker"] = best_neighbor

                # Re-merge newly adjacent same-speaker segments
                merged_segments.sort(key=lambda s: s["start"])
                collapsed = [merged_segments[0]]
                for seg in merged_segments[1:]:
                    prev = collapsed[-1]
                    if seg["speaker"] == prev["speaker"] and seg["start"] - prev["end"] <= 1.0:
                        prev["end"] = seg["end"]
                    else:
                        collapsed.append(seg)
                merged_segments = collapsed

                # Remove ghost speakers from profiles — their segments have been
                # reassigned so they no longer appear in the final output.
                for ghost in ghost_speakers:
                    profiles.pop(ghost, None)

            # Compute confidence for each segment based on embedding distances to all cluster centroids
            # For each embeddable window: compute distance to ALL centroids, measure assignment clarity
            # Clarity = distance_to_assigned / distance_to_2nd_closest (lower = clearer assignment)

            # First compute per-window distances to centroids
            if cluster_centroids and raw_embeddings is not None:
                centroid_arrays = {cid: np.array(centroid) for cid, centroid in cluster_centroids.items()}

                # For each embeddable window, store distances to all centroids
                for idx, (meta_idx, label) in enumerate(zip(embeddable_indices, long_labels)):
                    if meta_idx < len(raw_embeddings):
                        emb = raw_embeddings[meta_idx]
                        all_dists = {}
                        for cid, centroid in centroid_arrays.items():
                            dist = cosine(emb, centroid)
                            all_dists[cid] = dist

                        # Get sorted distances to measure clarity
                        sorted_dists = sorted(all_dists.values())
                        assigned_dist = all_dists.get(int(label), 1.0)

                        # Clarity: ratio of assigned distance to 2nd closest (lower = more confident)
                        if len(sorted_dists) > 1:
                            second_closest = sorted_dists[1]
                            clarity = assigned_dist / second_closest if second_closest > 0 else 0.0
                        else:
                            clarity = 0.0  # Only one cluster

                        # Store in metadata: distance to assigned cluster and clarity score
                        all_segments_meta[meta_idx]["assigned_dist"] = assigned_dist
                        all_segments_meta[meta_idx]["clarity"] = min(1.0, clarity)
                        all_segments_meta[meta_idx]["all_dists"] = all_dists

            # Now compute segment-level confidence
            segment_confidences = []
            for seg in merged_segments:
                seg_start = seg["start"]
                seg_end = seg["end"]

                clarities = []
                assigned_dists = []
                window_count = 0
                embeddable_count = 0
                speaker_consistency = {}  # count per speaker in this segment

                for meta in all_segments_meta:
                    if meta["start"] >= seg_start - 0.1 and meta["end"] <= seg_end + 0.1:
                        window_count += 1
                        if "speaker_raw" in meta:
                            embeddable_count += 1
                            raw_spk = meta["speaker_raw"]
                            speaker_consistency[raw_spk] = speaker_consistency.get(raw_spk, 0) + 1

                            # Collect clarity and distance if available
                            if "clarity" in meta:
                                clarities.append(meta["clarity"])
                            if "assigned_dist" in meta:
                                assigned_dists.append(meta["assigned_dist"])

                # Confidence components:
                # 1. Clarity score (average assignment clarity) - lower ratio = clearer
                clarity_conf = 0.5  # default
                if clarities:
                    avg_clarity = sum(clarities) / len(clarities)
                    # Convert clarity ratio to confidence: 0.0 = perfect, 1.0 = ambiguous
                    # confidence = 1 - clarity (so high clarity = high confidence)
                    clarity_conf = max(0, 1 - avg_clarity)

                # 2. Embedding distance to centroid (closer = higher confidence)
                dist_conf = 0.5  # default
                if assigned_dists:
                    avg_dist = sum(assigned_dists) / len(assigned_dists)
                    # Convert distance to confidence: 0.5 is max expected, scale to 0-1
                    dist_conf = max(0, 1 - (avg_dist / 0.5))

                # 3. Speaker consistency (do all windows agree?)
                consistency_conf = 0.5
                if speaker_consistency:
                    max_count = max(speaker_consistency.values())
                    total = sum(speaker_consistency.values())
                    consistency_conf = max_count / total if total > 0 else 0.5

                # 4. Duration factor (longer segments = more reliable)
                seg_dur = seg_end - seg_start
                dur_conf = min(1.0, seg_dur / 5.0)  # 5+ seconds = full confidence

                # Aggregate: weighted combination
                # Weight clarity and distance more heavily
                final_conf = (
                    clarity_conf * 0.3 +
                    dist_conf * 0.3 +
                    consistency_conf * 0.25 +
                    dur_conf * 0.15
                )

                # Boost if known speaker matching (inferred from profiles)
                if req.known_speakers and seg["speaker"] in req.known_speakers:
                    final_conf = min(1.0, final_conf + 0.15)  # Known speaker gets boost

                segment_confidences.append(final_conf)

            # Format final response — include profiles, confidence, and alternatives
            final_data = []
            for i, seg in enumerate(merged_segments):
                if seg["end"] <= seg["start"]:
                    continue
                seg_data = {
                    "start": round(float(seg["start"]), 3),
                    "end": round(float(seg["end"]), 3),
                    "speaker": seg["speaker"],
                    "confidence": round(segment_confidences[i], 3) if i < len(segment_confidences) else 0.5
                }
                # Add alternatives if present
                if "alternatives" in seg and seg["alternatives"]:
                    seg_data["alternatives"] = seg["alternatives"]
                final_data.append(seg_data)

            # Debug: log each segment's assignment
            for i, seg in enumerate(merged_segments):
                debug = seg.get("debug", {})
                conf = segment_confidences[i] if i < len(segment_confidences) else 0.5
                print(f"SEGMENT {seg['start']:.1f}-{seg['end']:.1f}: assigned='{seg['speaker']}' conf={conf:.2f}")

            # Ensure sorted by start time
            final_data.sort(key=lambda x: x["start"])

            unique_speakers = list(set(s["speaker"] for s in final_data))
            log.info("diarization_complete", unique_speakers=unique_speakers, num_segments=len(final_data))

            msg = {"type": "result", "segments": final_data, "profiles": profiles}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(msg))
            loop.call_soon_threadsafe(queue.put_nowait, None)

        except Exception as e:
            log.error("diarization_failed", error=str(e), exc_info=True)
            err_msg = {"error": f"Diarization processing failed: {str(e)}"}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(err_msg))
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Start thread
    task = asyncio.create_task(asyncio.to_thread(run_diarization_thread))

    async def event_generator():
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield msg + "\n"

            # Stop if we hit a terminal message
            if '"type": "result"' in msg or '"type": "error"' in msg:
                break

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/transcribe/upload", response_model=TranscribeResponse)
async def transcribe_upload(
    files: list[UploadFile] = File(..., description="Audio files to transcribe"),
    language: str = "en",
    _: str = Depends(verify_api_key)
):
    """
    Transcribe uploaded audio files.
    
    Accepts multiple audio files and returns transcriptions.
    Supports WAV, MP3, MP4, M4A, FLAC, OGG formats.
    """
    settings = get_settings()
    start_time = time.perf_counter()
    
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many files. Max: {settings.max_batch_size}"
        )
    
    if metrics:
        metrics.inc_request("/transcribe/upload", "started")
    
    results = []
    
    with metrics.track_request() if metrics else contextmanager(lambda: (yield))():
        for file in files:
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(
                    suffix=Path(file.filename or "audio.wav").suffix,
                    delete=True
                ) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp.flush()
                    
                    # Load audio
                    audio, _ = await asyncio.to_thread(
                        librosa.load, tmp.name, sr=16000, mono=True
                    )
                    
                    # Process audio in 30s chunks (model expects fixed-length input)
                    full_text = ""
                    total_duration = 0
                    total_inference = 0
                    total_tokens = 0
                    
                    chunk_samples = TARGET_SAMPLES  # 30s at 16kHz
                    for start in range(0, len(audio), chunk_samples):
                        chunk = audio[start:start + chunk_samples]
                        
                        # Pad short final chunk to exactly 30s
                        if len(chunk) < chunk_samples:
                            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
                        
                        result = await transcribe_audio_async(
                            chunk, 
                            language, 
                            settings.request_timeout
                        )
                        
                        full_text += result["text"] + " "
                        total_duration += result["audio_duration_sec"]
                        total_inference += result["inference_time_sec"]
                        total_tokens += result["tokens_generated"]
                    
                    result = {
                        "text": full_text.strip(),
                        "audio_duration_sec": len(audio) / 16000,
                        "inference_time_sec": total_inference,
                        "tokens_generated": total_tokens
                    }
                    
                    results.append(TranscribeResult(
                        text=result["text"],
                        audio_duration_sec=result["audio_duration_sec"],
                        inference_time_sec=result["inference_time_sec"],
                        tokens_generated=result["tokens_generated"]
                    ))
                    
            except Exception as e:
                log.error("transcription_failed", filename=file.filename, error=str(e))
                results.append(TranscribeResult(
                    text="",
                    audio_duration_sec=0,
                    inference_time_sec=0,
                    tokens_generated=0,
                    error=str(e)
                ))
    
    if metrics:
        status = "success" if all(r.error is None for r in results) else "partial"
        metrics.inc_request("/transcribe/upload", status)
    
    return TranscribeResponse(
        results=results,
        total_time_sec=time.perf_counter() - start_time
    )


@app.post("/transcribe/paths", response_model=TranscribeResponse)
async def transcribe_paths(
    req: TranscribePathsRequest,
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_api_key)
):
    """
    Transcribe audio files by path.
    
    Paths must be accessible by the server and within allowed directories
    if TRANSCRIBE_ALLOWED_AUDIO_DIR is set.
    """
    start_time = time.perf_counter()
    
    if metrics:
        metrics.inc_request("/transcribe/paths", "started")
    
    results = []
    
    with metrics.track_request() if metrics else contextmanager(lambda: (yield))():
        for path in req.wav_paths:
            try:
                # Validate path
                resolved = validate_path_security(path, settings)
                
                # Load audio
                audio, _ = await asyncio.to_thread(
                    librosa.load, str(resolved), sr=16000, mono=True
                )
                
                # Process audio in 30s chunks (model expects fixed-length input)
                full_text = ""
                total_duration = 0
                total_inference = 0
                total_tokens = 0
                
                chunk_samples = TARGET_SAMPLES  # 30s at 16kHz
                for start in range(0, len(audio), chunk_samples):
                    chunk = audio[start:start + chunk_samples]
                    
                    # Pad short final chunk to exactly 30s
                    if len(chunk) < chunk_samples:
                        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
                    
                    result = await transcribe_audio_async(
                        chunk, 
                        req.language, 
                        settings.request_timeout
                    )
                    
                    full_text += result["text"] + " "
                    total_duration += result["audio_duration_sec"]
                    total_inference += result["inference_time_sec"]
                    total_tokens += result["tokens_generated"]
                
                result = {
                    "text": full_text.strip(),
                    "audio_duration_sec": len(audio) / 16000,
                    "inference_time_sec": total_inference,
                    "tokens_generated": total_tokens
                }
                
                results.append(TranscribeResult(
                    text=result["text"],
                    audio_duration_sec=result["audio_duration_sec"],
                    inference_time_sec=result["inference_time_sec"],
                    tokens_generated=result["tokens_generated"],
                    speakers=result.get("speakers")
                ))
                
            except PathSecurityError as e:
                log.warning("path_security_error", path=path, error=str(e))
                results.append(TranscribeResult(
                    text="",
                    audio_duration_sec=0,
                    inference_time_sec=0,
                    tokens_generated=0,
                    error=f"Access denied: {e.message}"
                ))
                
            except Exception as e:
                log.error("transcription_failed", path=path, error=str(e))
                results.append(TranscribeResult(
                    text="",
                    audio_duration_sec=0,
                    inference_time_sec=0,
                    tokens_generated=0,
                    error=str(e)
                ))
    
    if metrics:
        status = "success" if all(r.error is None for r in results) else "partial"
        metrics.inc_request("/transcribe/paths", status)
    
    return TranscribeResponse(
        results=results,
        total_time_sec=time.perf_counter() - start_time
    )


# Legacy endpoint for backward compatibility
@app.post("/transcribe")
async def transcribe_legacy(
    req: TranscribePathsRequest,
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_api_key)
):
    """
    Legacy transcription endpoint (deprecated).
    Use /transcribe/paths or /transcribe/upload instead.
    """
    response = await transcribe_paths(req, settings, _)
    
    # Return legacy format
    return {
        "results": [r.text if r.error is None else f"[ERROR: {r.error}]" for r in response.results]
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False,  # We have our own logging
    )