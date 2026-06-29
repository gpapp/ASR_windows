"""
Model state for Nemotron ASR model and embedding/VAD sessions.
"""

import gc
import threading
from collections import OrderedDict
from typing import Optional
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og
import structlog

from settings import Settings

log = structlog.get_logger()

EMBEDDING_CACHE_MAX_SIZE = 5000


class LRUCache:
    """Thread-safe LRU cache with max size."""

    def __init__(self, max_size: int = EMBEDDING_CACHE_MAX_SIZE):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: np.ndarray):
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                removed_key, _ = self._cache.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self):
        with self._lock:
            self._cache.clear()


class ModelState:
    """Thread-safe container for model state — embeddings, VAD, and Nemotron ASR."""

    def __init__(self):
        self.model: Optional[og.Model] = None
        self.processor: Optional[og.StreamingProcessor] = None
        self.tokenizer: Optional[og.Tokenizer] = None
        self.embedding_session: Optional[ort.InferenceSession] = None
        self.vad_session: Optional[ort.InferenceSession] = None
        self.status: str = "initializing"
        self.embedding_cache: LRUCache = LRUCache()
        self.lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    def clear_gpu_memory(self):
        log.info("clearing_gpu_memory")
        self.embedding_cache.clear()
        gc.collect()


def run_embedding(input_feed: dict) -> list[np.ndarray]:
    """Run embedding session with automatic CPU fallback on GPU OOM."""
    try:
        return state.embedding_session.run(None, input_feed)
    except Exception as e:
        log.warning("embedding_inference_failed_reloading_cpu", error=str(e))
        from model_loader import reload_embedding_session
        reload_embedding_session(get_settings(), force_cpu=True)
        try:
            return state.embedding_session.run(None, input_feed)
        except Exception as e2:
            log.error("embedding_inference_failed_after_reload", error=str(e2))
            raise


# Global state
state = ModelState()
executor: Optional[ThreadPoolExecutor] = None
