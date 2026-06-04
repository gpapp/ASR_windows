"""
Model state and KV cache pool management.
"""

import threading
from typing import Optional
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import structlog

from settings import Settings

log = structlog.get_logger()


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
        try:
            return {
                "self_k": ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((s.n_layers, 1, s.heads, s.max_ctx, s.head_dim), dtype=s.decoder_dtype),
                    self.device, 0
                ),
                "self_v": ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((s.n_layers, 1, s.heads, s.max_ctx, s.head_dim), dtype=s.decoder_dtype),
                    self.device, 0
                ),
            }
        except Exception as e:
            log.warning("kv_cache_device_allocation_failed", device=self.device, error=str(e), fallback="cpu")
            self.device = "cpu"
            return {
                "self_k": ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((s.n_layers, 1, s.heads, s.max_ctx, s.head_dim), dtype=s.decoder_dtype),
                    "cpu", 0
                ),
                "self_v": ort.OrtValue.ortvalue_from_numpy(
                    np.zeros((s.n_layers, 1, s.heads, s.max_ctx, s.head_dim), dtype=s.decoder_dtype),
                    "cpu", 0
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
        self.vad_session: Optional[ort.InferenceSession] = None
        self.tokens: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self.pre_computed_prompt_ids: list[int] = []
        self.pre_computed_eos_id: int = -1
        self.pre_computed_prompt_array: Optional[np.ndarray] = None
        self.status: str = "initializing"
        self.kv_pool: Optional[KVCachePool] = None
        self.embedding_cache: dict[str, np.ndarray] = {}
        self.lock = threading.Lock()
    
    @property
    def is_ready(self) -> bool:
        return self.status == "ready"


# Global state
state = ModelState()
executor: Optional[ThreadPoolExecutor] = None
