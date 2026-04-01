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
from pathlib import Path
from typing import Optional
from functools import lru_cache
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio

from dotenv import load_dotenv
load_dotenv()

import torch
import numpy as np
import json
import librosa
import onnxruntime as ort
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
    max_request_size_mb: int = 100
    
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
    embedding_model_repo: str = Field(default="pyannote/wespeaker-voxceleb-resnet34-LM", description="HuggingFace repo for the embedding ONNX model")
    embedding_model_filename: str = Field(default="pytorch_model.onnx", description="Filename of the ONNX embedding model")
    diarization_threshold: float = Field(default=0.5, description="Distance threshold for AgglomerativeClustering (cosine metric)")
    
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
    state.use_dml = use_dml
    state.device = device
    state.kv_pool = KVCachePool(settings, device)

    if settings.enable_diarization:
        log.info("loading_vad_model")
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            # Ensure VAD runs on CPU or the requested device
            model = model.to(torch.device("cpu"))
            state.vad_model = model
            state.get_speech_timestamps = utils[0]
            log.info("vad_model_loaded")
        except Exception as e:
            log.error("vad_model_failed", error=str(e))

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
        
        # Run encoder
        enc_io = encoder.io_binding()
        enc_io.bind_cpu_input("audio", audio.reshape(1, -1).astype(np.float32))
        enc_io.bind_output("n_layer_cross_k", device)
        enc_io.bind_output("n_layer_cross_v", device)
        encoder.run_with_iobinding(enc_io)
        
        enc_out = enc_io.get_outputs()
        cross_k_ov = enc_out[0]
        cross_v_ov = enc_out[1]
        
        # Build prompt
        lang_token = f"<|{language}|>"
        prompt_tokens = [
            "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
            lang_token, lang_token, "<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>",
        ]
        prompt_ids = [token_to_id[t] for t in prompt_tokens if t in token_to_id]
        eos_id = token_to_id.get("<|endoftext|>", -1)
        
        # Get KV cache from pool
        with state.kv_pool.acquire() as kv_cache:
            self_k_ov = kv_cache["self_k"]
            self_v_ov = kv_cache["self_v"]
            
            generated = list(prompt_ids)
            current = np.array([prompt_ids], dtype=np.int64)
            offset = np.array(0, dtype=np.int64)
            
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
                self_k_ov = dec_out[1]
                self_v_ov = dec_out[2]
                
                next_id = int(np.argmax(logits[0, -1, :]))
                if next_id == eos_id:
                    break
                    
                generated.append(next_id)
                offset = np.array(int(offset) + current.shape[1], dtype=np.int64)
                current = np.array([[next_id]], dtype=np.int64)
        
        # Decode text
        text = "".join(
            tokens.get(t, "").replace("\u2581", " ")
            for t in generated[len(prompt_ids):]
            if not tokens.get(t, "").startswith("<|")
        ).strip()

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
    if not state.diarization_pipeline:
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

    def progress_hook(step_name, step_artefact, file=None, completed=None, total=None):
        """Callback fired by Pyannote during processing."""
        # Convert internal pyannote hook names to user-friendly ones
        if completed is not None and total is not None:
            msg = {
                "type": "progress",
                "step": step_name,
                "completed": int(completed),  # Fix int64 json serialization
                "total": int(total)
            }
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(msg))

    def run_diarization_thread():
        try:
            # 1. Load audio (doing this inside the thread now so we can stream load progress if needed)
            waveform, sr = librosa.load(str(resolved), sr=16000, mono=True)
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()

            if not state.vad_model or not state.get_speech_timestamps:
                log.warning("vad_not_loaded_falling_back_to_full_audio")
                annotation = state.diarization_pipeline(
                    {"waveform": waveform_tensor, "sample_rate": 16000},
                    hook=progress_hook
                ).exclusive_speaker_diarization
            else:
                # Run VAD
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "type": "progress", "step": "Voice Activity Detection", "completed": 0, "total": 1
                }))

                speech_ts = state.get_speech_timestamps(
                    waveform_tensor,
                    state.vad_model,
                    sampling_rate=16000,
                    return_seconds=True
                )

                if not speech_ts:
                    log.info("vad_found_no_speech", path=req.wav_path)
                    annotation = Annotation()
                else:
                    # Build dense waveform
                    dense_parts = []
                    mapping = []
                    current_dense_start = 0.0

                    for ts in speech_ts:
                        orig_start, orig_end = ts['start'], ts['end']
                        start_sample = int(orig_start * 16000)
                        end_sample = int(orig_end * 16000)

                        segment = waveform_tensor[:, start_sample:end_sample]
                        dense_parts.append(segment)

                        duration = (end_sample - start_sample) / 16000.0
                        current_dense_end = current_dense_start + duration

                        mapping.append({
                            'dense_start': current_dense_start,
                            'dense_end': current_dense_end,
                            'orig_start': orig_start,
                            'orig_end': orig_end
                        })
                        current_dense_start = current_dense_end

                    # Concatenate and run Pyannote
                    dense_waveform = torch.cat(dense_parts, dim=1)

                    dense_diarization = state.diarization_pipeline(
                        {"waveform": dense_waveform, "sample_rate": 16000},
                        hook=progress_hook
                    )

                    # Remap timestamps
                    annotation = Annotation()
                    dense_exclusive = dense_diarization.exclusive_speaker_diarization

                    for turn, _, speaker in dense_exclusive.itertracks(yield_label=True):
                        dense_turn_start = turn.start
                        dense_turn_end = turn.end

                        for m in mapping:
                            overlap_start = max(dense_turn_start, m['dense_start'])
                            overlap_end = min(dense_turn_end, m['dense_end'])

                            if overlap_start < overlap_end:
                                rel_start = overlap_start - m['dense_start']
                                rel_end = overlap_end - m['dense_start']

                                orig_turn_start = m['orig_start'] + rel_start
                                orig_turn_end = m['orig_start'] + rel_end

                                annotation[Segment(orig_turn_start, orig_turn_end)] = speaker

            # Extract segments
            segments = []
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker)
                })

            # Send final result
            msg = {
                "type": "result",
                "segments": segments,
                "total_time_sec": time.perf_counter() - start_time
            }
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(msg))

        except Exception as e:
            log.error("diarization_failed", path=req.wav_path, error=str(e))
            msg = {"type": "error", "error": str(e)}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(msg))

    # Start thread
    task = asyncio.create_task(asyncio.to_thread(run_diarization_thread))

    async def event_generator():
        while True:
            msg = await queue.get()
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
                    
                    # Transcribe
                    result = await transcribe_audio_async(
                        audio, 
                        language, 
                        settings.request_timeout
                    )
                    
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
                
                # Transcribe
                result = await transcribe_audio_async(
                    audio, 
                    req.language, 
                    settings.request_timeout
                )
                
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