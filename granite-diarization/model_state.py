"""
Model state for Granite ASR and auxiliary models.
"""

import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import structlog

log = structlog.get_logger()


class ModelState:
    """Thread-safe container for model state."""

    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cpu"
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


state = ModelState()
executor: Optional[ThreadPoolExecutor] = None
