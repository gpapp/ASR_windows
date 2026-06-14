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


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Granite ASR model settings
    model_repo: str = "ibm-granite/granite-speech-4.1-2b-plus"
    model_dir: Path = Path(__file__).parent.parent / "models/granite-speech"

    # SAA prompt for speaker-attributed transcription
    system_prompt: str = (
        "Knowledge Cutoff Date: April 2024.\n"
        "Today's Date: December 19, 2024.\n"
        "You are Granite, developed by IBM. You are a helpful AI assistant"
    )
    saa_prompt: str = (
        "<|audio|> Speaker attribution: Transcribe and denote who is speaking "
        "by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
    )

    # VAD model settings
    vad_model_repo: str = Field(default="onnx-community/silero-vad", description="HuggingFace repo for the Silero VAD model")
    vad_model_dir: Path = Path(__file__).parent.parent / "models/silero-vad-onnx"
    vad_model_type: str = ""
    vad_dtype: type = np.float16 if vad_model_type in ["_fp16", "_q4f16"] else np.float32 if vad_model_type in ["_quantized", "_q4"] else np.int8 if vad_model_type == "_int8" else np.uint8 if vad_model_type == "_uint8" else np.float32

    # Embedding model settings
    embedding_model_repo: str = Field(default="Wespeaker/wespeaker-ecapa-tdnn512-LM", description="HuggingFace repo for the embedding ONNX model")
    embedding_model_filename: str = Field(default="voxceleb_ECAPA512_LM.onnx", description="Filename of the ONNX embedding model")
    embedding_model_dir: Path = Path(__file__).parent.parent / "models/embedding"

    # ASR chunking
    chunk_duration: int = 60
    prefix_words_per_speaker: int = 10
    max_new_tokens: int = 400

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 2
    request_timeout: int = 120
    max_request_size_mb: int = 200

    # Batch settings
    max_batch_size: int = 10
    max_audio_duration_sec: int = 600

    # Security settings
    allowed_audio_dir: Optional[Path] = None
    api_keys: Optional[str] = None
    enable_cors: bool = True
    cors_origins: str = "*"

    # Feature flags
    enable_rate_limit: bool = True
    hf_token: Optional[str] = None
    rate_limit: str = "30/minute"

    # Diarization settings
    diarization_threshold: float = Field(default=0.35, description="Distance threshold for AgglomerativeClustering")
    vad_threshold: float = Field(default=0.5, description="Speech probability cutoff (0.0 to 1.0) for Silero VAD")
    vad_min_speech_duration_ms: int = Field(default=250, description="Minimum speech chunk length (ms) for Silero VAD")

    # VAD chunking (for partitioning long audio)
    vad_chunk_duration: int = 30
    vad_overlap: int = 5

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


def setup_logging():
    """Configure structured JSON logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

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


setup_logging()
