"""
Model state for Granite ASR (ONNX) and auxiliary models.
"""

import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import structlog

log = structlog.get_logger()


class ModelState:
    """Thread-safe container for model state."""

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.processor = None
        self.tokenizer = None

        # ONNX sessions (Granite ASR)
        self.encoder_session: Optional[ort.InferenceSession] = None
        self.embed_tokens_session: Optional[ort.InferenceSession] = None
        self.prompt_encode_session: Optional[ort.InferenceSession] = None
        self.decode_step_session: Optional[ort.InferenceSession] = None

        # Auxiliary models
        self.embedding_session = None
        self.vad_session = None

        self.tokens: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self.status: str = "initializing"
        self.embedding_cache: dict[str, np.ndarray] = {}
        self.lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    @property
    def onnx_ready(self) -> bool:
        return all([
            self.encoder_session is not None,
            self.embed_tokens_session is not None,
            self.prompt_encode_session is not None,
            self.decode_step_session is not None,
        ])


state = ModelState()
executor: Optional[ThreadPoolExecutor] = None
