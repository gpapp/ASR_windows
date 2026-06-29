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
    # Model settings
    provider_type: str = "DirectML"  # Options: "DirectML", "CPU"
    model_repo: str = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4"
    model_dir: Path = Path(__file__).parent.parent / "models/nemotron-onnx-int4"
    cpu_threads: int = max(1, os.cpu_count() - 1)

    # Nemotron model constants
    nemotron_sample_rate: int = 16000
    nemotron_chunk_samples: int = 8960

    # VAD model settings
    vad_model_repo: str = Field(default="onnx-community/silero-vad", description="HuggingFace repo for the Silero VAD model")
    vad_model_dir: Path = Path(__file__).parent.parent / "models/silero-vad-onnx"
    vad_model_type: str = ""
    vad_dtype: type = np.float32

    # Embedding model settings
    embedding_model_repo: str = Field(default="Wespeaker/wespeaker-ecapa-tdnn512-LM", description="HuggingFace repo for the embedding ONNX model")
    embedding_model_filename: str = Field(default="voxceleb_ECAPA512_LM.onnx", description="Filename of the ONNX embedding model")
    embedding_model_dir: Path = Path(__file__).parent.parent / "models/embedding"

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8001
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
    diarization_threshold: float = Field(default=0.35, description="Distance threshold for AgglomerativeClustering - higher = fewer clusters")
    vad_threshold: float = Field(default=0.5, description="Speech probability cutoff (0.0 to 1.0) for Silero VAD")
    vad_min_speech_duration_ms: int = Field(default=250, description="Minimum speech chunk length (ms) for Silero VAD")

    # Language
    asr_language: str = Field(default="en", description="Language code for Nemotron ASR (e.g. 'en', 'de', 'fr')")

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
