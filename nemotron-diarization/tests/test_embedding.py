"""Unit tests for speaker embedding functions."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest
from speaker.embedding import normalize_embedding, compute_pitch, compute_energy


class TestNormalizeEmbedding:
    def test_l2_normalizes_vector(self):
        emb = [3.0, 4.0]  # Length 5, should become [0.6, 0.8]
        
        result = normalize_embedding(emb)
        
        expected = [0.6, 0.8]
        assert np.allclose(result, expected, atol=0.01)
    
    def test_preserves_unit_vector(self):
        emb = [0.6, 0.8]
        
        result = normalize_embedding(emb)
        
        # Should be unchanged (already normalized)
        assert np.allclose(result, emb, atol=0.01)
    
    def test_handles_zero_vector(self):
        emb = [0.0, 0.0]
        
        result = normalize_embedding(emb)
        
        # Should remain zeros
        assert result == [0.0, 0.0]
    
    def test_handles_numpy_array(self):
        emb = np.array([3.0, 4.0])
        
        result = normalize_embedding(emb)
        
        expected = [0.6, 0.8]
        assert np.allclose(result, expected, atol=0.01)


class TestComputePitch:
    def test_returns_pitch_for_sine_wave(self):
        sample_rate = 16000
        frequency = 200  # 200 Hz
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.tensor(np.sin(2 * np.pi * frequency * t))
        
        pitch, std = compute_pitch(waveform, sample_rate)
        
        assert 180 < pitch < 220  # Should be close to 200 Hz
    
    def test_returns_zero_for_silence(self):
        waveform = torch.zeros(16000)
        
        pitch, std = compute_pitch(waveform, 16000)
        
        assert pitch == 0.0
        assert std == 0.0
    
    def test_returns_median_and_std(self):
        sample_rate = 16000
        # Create waveform with varying pitch
        t = np.linspace(0, 1.0, 16000)
        waveform = torch.tensor(np.sin(2 * np.pi * 150 * t) + 0.1 * np.random.randn(16000))
        
        pitch, std = compute_pitch(waveform, sample_rate)
        
        assert pitch > 0
        assert std >= 0


class TestComputeEnergy:
    def test_returns_rms_energy(self):
        # Pure sine wave with amplitude 1.0 -> RMS = 1/sqrt(2) ≈ 0.707
        waveform = torch.tensor([1.0, -1.0, 1.0, -1.0])
        
        energy = compute_energy(waveform)
        
        expected = 1.0 / np.sqrt(2)
        assert np.isclose(energy, expected, atol=0.01)
    
    def test_handles_stereo_waveform(self):
        # 2-channel waveform, should be averaged
        waveform = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
        
        energy = compute_energy(waveform)
        
        assert energy > 0
    
    def test_returns_zero_for_silence(self):
        waveform = torch.zeros(1000)
        
        energy = compute_energy(waveform)
        
        assert energy == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])