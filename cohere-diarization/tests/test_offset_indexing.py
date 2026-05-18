"""
Regression test for 0-dimensional offset array bug in transcribe_audio_sync.

Root cause: offset = np.array(0, dtype=np.int64) creates a 0-d scalar array.
The ONNX model REQUIRES this to be 0-d for its Slice node. But Python-side
code was doing offset[0] which fails on 0-d arrays.

Fix: use int(offset) instead of offset[0] to extract the scalar value, and
keep offset as a 0-d array throughout the loop so the ONNX model is happy.
"""
import numpy as np
import pytest


def simulate_decode_loop_buggy(offset_init):
    """
    Reproduces the BROKEN code path: uses offset[0] to index a 0-d array.
    """
    current = np.array([[1, 2, 3]], dtype=np.int64)  # shape (1, 3)
    offset = offset_init

    # Old buggy line (was server.py line 657):
    new_offset = offset[0] + current.shape[1]
    return int(new_offset)


def simulate_decode_loop_fixed(offset_init):
    """
    Simulates the FIXED code path: uses int(offset) to safely extract the
    scalar from a 0-d numpy array, then creates a new 0-d array.
    """
    current = np.array([[1, 2, 3]], dtype=np.int64)  # shape (1, 3)
    offset = offset_init

    # Fixed line: int(offset) works on both 0-d and 1-d arrays
    offset = np.array(int(offset) + current.shape[1], dtype=np.int64)
    return int(offset)


class TestOffsetIndexing:
    def test_0d_offset_indexing_raises(self):
        """Reproduces the original bug: 0-d np.array cannot be indexed with [0]."""
        offset_0d = np.array(0, dtype=np.int64)
        assert offset_0d.ndim == 0, "Precondition: must be 0-dimensional"

        with pytest.raises(IndexError, match="too many indices"):
            simulate_decode_loop_buggy(offset_0d)

    def test_0d_offset_fixed_works(self):
        """Verifies that int(offset) works on 0-d arrays (the fix)."""
        offset_0d = np.array(0, dtype=np.int64)
        result = simulate_decode_loop_fixed(offset_0d)
        assert result == 3  # 0 + current.shape[1] == 3

    def test_fixed_offset_stays_0d(self):
        """Verifies the fix keeps offset as 0-d for ONNX compatibility."""
        offset = np.array(0, dtype=np.int64)
        current = np.array([[1, 2, 3]], dtype=np.int64)

        # Simulate one iteration of the fixed loop
        offset = np.array(int(offset) + current.shape[1], dtype=np.int64)

        assert offset.ndim == 0, "offset must remain 0-d for ONNX Slice node"
        assert int(offset) == 3

    def test_fixed_multiple_iterations(self):
        """Verifies correct accumulation over multiple decode iterations."""
        offset = np.array(0, dtype=np.int64)

        # First iteration: prompt has 3 tokens
        current = np.array([[1, 2, 3]], dtype=np.int64)
        offset = np.array(int(offset) + current.shape[1], dtype=np.int64)
        assert int(offset) == 3
        assert offset.ndim == 0

        # Subsequent iterations: single token
        current_buffer = np.zeros((1, 1), dtype=np.int64)
        current_buffer[0, 0] = 42
        current = current_buffer

        offset = np.array(int(offset) + current.shape[1], dtype=np.int64)
        assert int(offset) == 4
        assert offset.ndim == 0

        offset = np.array(int(offset) + current.shape[1], dtype=np.int64)
        assert int(offset) == 5
        assert offset.ndim == 0
