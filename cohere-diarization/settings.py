"""
Application settings and logging configuration.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model settings
    provider_type: str = "DirectML"  # Options: "DirectML", "OpenVINO", "CPU"
    model_repo: str = "onnx-community/cohere-transcribe-03-2026-ONNX"
    model_dir: Path = Path(__file__).parent.parent / "models/cohere-transcribe-onnx"
    encoder_model_type: str = ""  # options: _fp16, _quantized, _q4, _q4f16, 
    encoder_dtype: type = np.float16 if encoder_model_type in ["_fp16", "_q4f16"] else np.float32 if encoder_model_type in ["_quantized", "_q4"] else np.float32

    # Decoder model ALWAYS on CPU for maximum compatibility, as it's only used for short sequences and token-by-token 
    # generation where GPU acceleration is less critical. We can still use quantized/FP16 models for the decoder to save memory, but we'll run them on CPU to avoid DirectML issues.
    decoder_model_type: str = ""  # options: _fp16, _quantized, _q4, _q4f16, — q4 fastest on this CPU (fp16 adds dtype conversion overhead)
    decoder_dtype: type = np.float16 if decoder_model_type in ["_fp16", "_q4f16"] else np.float32  # KV cache dtype (encoder_hidden_states remains float32)
    cpu_threads: int = max(1, os.cpu_count() - 1)

    # f16 is extremely slow on CPU and GPU due to covnversion overhead

    # VAD model settings
    # VAD models are always run on CPU for maximum compatibility, as they are only used for short audio chunks and we want to avoid any DirectML issues. We can still use quantized/FP16 models for VAD to save memory, but we'll run them on CPU.
    vad_model_repo: str = Field(default="onnx-community/silero-vad", description="HuggingFace repo for the Silero VAD model")
    vad_model_dir: Path = Path(__file__).parent.parent / "models/silero-vad-onnx"
    vad_model_type: str = ""  # options: _bnb4, _fp16, _int8, _uint8, _quantized, _q4, _q4f16, 
    vad_dtype: type = np.float16 if vad_model_type in ["_fp16", "_q4f16"] else np.float32 if vad_model_type in ["_quantized", "_q4"] else np.int8 if vad_model_type == "_int8" else np.uint8 if vad_model_type == "_uint8" else np.float32

    # Embedding model settings
    embedding_model_repo: str = Field(default="Wespeaker/wespeaker-ecapa-tdnn512-LM", description="HuggingFace repo for the embedding ONNX model")
    embedding_model_filename: str = Field(default="voxceleb_ECAPA512_LM.onnx", description="Filename of the ONNX embedding model")
    embedding_model_dir: Path = Path(__file__).parent.parent / "models/embedding"
    
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
    enable_rate_limit: bool = True
    hf_token: Optional[str] = None
    rate_limit: str = "30/minute"
    
    # Diarization settings
    diarization_threshold: float = Field(default=0.35, description="Distance threshold for AgglomerativeClustering - higher = fewer clusters")
    vad_threshold: float = Field(default=0.5, description="Speech probability cutoff (0.0 to 1.0) for Silero VAD")
    vad_min_speech_duration_ms: int = Field(default=250, description="Minimum speech chunk length (ms) for Silero VAD")
    
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
    # Configure stdlib logging with console handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
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


# Setup logging on module import
setup_logging()
