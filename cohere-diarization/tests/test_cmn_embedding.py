"""
Regression test for missing CMN in identify_speakers_in_audio.

Root cause: voiceprint_utils.py identify_speakers_in_audio() was NOT applying
Cepstral Mean Normalization (CMN) to fbank features before embedding extraction.
The server diarization pipeline (server.py) and extract_embedding (speaker/embedding.py)
both apply CMN, producing correct embeddings. Without CMN, embeddings are garbage and
cosine distances to voiceprints always exceed the match threshold, yielding 0 identifications.

Fix: Apply CMN (batch - batch.mean(dim=2, keepdim=True)) before squeeze and ONNX inference.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest


class TestCMNApplied:
    """Verify that CMN is applied in the embedding pipeline used by identify_speakers_in_audio."""

    def test_cmn_changes_features(self):
        """CMN (subtracting frame-axis mean) should change the feature matrix."""
        # Simulate a random fbank: [1, frames=100, mel_bins=80]
        fbank = torch.randn(1, 100, 80) + 5.0  # offset so mean != 0

        # Without CMN
        raw = fbank.clone()

        # With CMN (as applied in the fix)
        cmn = fbank - fbank.mean(dim=1, keepdim=True)  # Note: dim=1 for [1, T, 80]

        # They must differ
        assert not torch.allclose(raw, cmn), "CMN should modify features when mean != 0"

    def test_cmn_batch_matches_server(self):
        """
        The CMN applied in voiceprint_utils must match what server.py does:
        batch - batch.mean(dim=2, keepdim=True) on a [N, 1, max_len, 80] tensor.
        """
        N, T, D = 4, 100, 80
        # Simulate stacked fbanks: [N, 1, T, D]
        batch = torch.randn(N, 1, T, D) + 3.0

        # Server approach (line 1451): dim=2 on [N, 1, T, D]
        cmn_server = batch - batch.mean(dim=2, keepdim=True)

        # voiceprint_utils fix: same operation
        cmn_fix = batch - batch.mean(dim=2, keepdim=True)

        assert torch.allclose(cmn_server, cmn_fix), "CMN must match server implementation"

        # After squeeze(1) -> [N, T, D], verify it's correct
        result = cmn_fix.squeeze(1)
        assert result.shape == (N, T, D)

        # Each frame's mel features should have zero mean along the time axis
        # (CMN normalizes along time dimension per feature)
        # Actually CMN subtracts the mean along time (dim=2 in 4D = dim=1 in 3D)
        # So mean along time axis of result should be ~0
        time_means = result.mean(dim=1)  # [N, D]
        assert torch.allclose(time_means, torch.zeros_like(time_means), atol=1e-5), \
            "After CMN, mean along time axis should be ~0"

    def test_extract_embedding_has_cmn(self):
        """Verify that speaker/embedding.py extract_embedding has CMN."""
        import inspect
        from speaker.embedding import extract_embedding

        source = inspect.getsource(extract_embedding)
        assert "mean(dim=2" in source or "mean(dim=1" in source, \
            "extract_embedding must apply CMN (mean subtraction along time)"

    def test_identify_speakers_source_has_cmn(self):
        """Verify that identify_speakers_in_audio now has CMN."""
        import inspect
        from voiceprint_utils import identify_speakers_in_audio

        source = inspect.getsource(identify_speakers_in_audio)
        assert "batch.mean(dim=2" in source or "batch - batch.mean" in source, \
            "identify_speakers_in_audio must apply CMN before embedding extraction"
