"""
Unit tests for split_at_energy_dips in speaker/vad.py.

Verifies that long VAD segments containing a clear energy dip in the middle
are correctly split into two pieces, reducing cross-speaker window contamination.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from speaker.vad import split_at_energy_dips


SAMPLE_RATE = 16000


def make_speech_then_silence_then_speech(
    speech_dur=3.0, silence_dur=0.4, total_dur=10.0, amplitude=0.3
):
    """Create a synthetic waveform: speech | silence | speech.
    The silence is in the middle, creating a clear energy dip.
    """
    n_total = int(total_dur * SAMPLE_RATE)
    waveform = np.zeros(n_total, dtype=np.float32)

    # First speech block: 0 to speech_dur
    n_speech = int(speech_dur * SAMPLE_RATE)
    t = np.linspace(0, speech_dur, n_speech)
    waveform[:n_speech] = amplitude * np.sin(2 * np.pi * 200 * t).astype(np.float32)

    # Silence: speech_dur to speech_dur + silence_dur (stays zero)

    # Second speech block: speech_dur + silence_dur to end
    start2 = int((speech_dur + silence_dur) * SAMPLE_RATE)
    n_speech2 = n_total - start2
    t2 = np.linspace(0, n_speech2 / SAMPLE_RATE, n_speech2)
    waveform[start2:] = amplitude * np.sin(2 * np.pi * 300 * t2).astype(np.float32)

    return waveform


class TestSplitAtEnergyDips:
    def test_short_segment_not_split(self):
        """Segments shorter than min_segment_dur should be returned unchanged."""
        waveform = make_speech_then_silence_then_speech(total_dur=4.0)
        speech_ts = [{"start": 0.0, "end": 4.0}]

        result = split_at_energy_dips(
            speech_ts, waveform, sample_rate=SAMPLE_RATE, min_segment_dur=5.0
        )

        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 4.0

    def test_long_segment_with_clear_dip_is_split(self):
        """A long segment with a clear silence dip should be split into 2+ pieces."""
        silence_start = 3.0
        silence_dur = 0.4
        waveform = make_speech_then_silence_then_speech(
            speech_dur=silence_start, silence_dur=silence_dur, total_dur=10.0, amplitude=0.5
        )
        speech_ts = [{"start": 0.0, "end": 10.0}]

        result = split_at_energy_dips(
            speech_ts,
            waveform,
            sample_rate=SAMPLE_RATE,
            min_segment_dur=5.0,
            dip_ratio=0.35,
            min_dip_dur=0.1,
            min_split_piece=1.0,
        )

        assert len(result) >= 2, f"Expected split, got {len(result)} segments: {result}"

        # First piece should start at 0
        assert result[0]["start"] == pytest.approx(0.0, abs=0.1)
        # Last piece should end near 10.0
        assert result[-1]["end"] == pytest.approx(10.0, abs=0.1)

        # The split should be somewhere near the silence (between 2.5s and 5.0s)
        split_point = result[0]["end"]
        assert 2.0 < split_point < 5.5, f"Split point {split_point:.2f}s not near expected {silence_start}s"

    def test_continuous_speech_not_split(self):
        """Uniform speech (no dip) should not be split."""
        n = int(10.0 * SAMPLE_RATE)
        t = np.linspace(0, 10.0, n)
        waveform = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        speech_ts = [{"start": 0.0, "end": 10.0}]
        result = split_at_energy_dips(
            speech_ts, waveform, sample_rate=SAMPLE_RATE, min_segment_dur=5.0,
            dip_ratio=0.35, min_dip_dur=0.15
        )

        assert len(result) == 1

    def test_multiple_segments_only_long_ones_split(self):
        """Short segments should pass through; only long ones get split."""
        silence_start = 3.0
        silence_dur = 0.5
        total_long = 10.0

        waveform_long = make_speech_then_silence_then_speech(
            speech_dur=silence_start, silence_dur=silence_dur,
            total_dur=total_long, amplitude=0.5
        )
        # Short waveform with no dip
        n_short = int(4.0 * SAMPLE_RATE)
        waveform_short = 0.3 * np.ones(n_short, dtype=np.float32)

        # Combine: short segment [0, 4), long segment [5, 15)
        full_waveform = np.zeros(int(15.0 * SAMPLE_RATE), dtype=np.float32)
        full_waveform[:n_short] = waveform_short
        long_start_sample = int(5.0 * SAMPLE_RATE)
        n_long = int(total_long * SAMPLE_RATE)
        full_waveform[long_start_sample:long_start_sample + n_long] = waveform_long

        speech_ts = [
            {"start": 0.0, "end": 4.0},   # short — not split
            {"start": 5.0, "end": 15.0},  # long — should split
        ]
        result = split_at_energy_dips(
            speech_ts, full_waveform, sample_rate=SAMPLE_RATE,
            min_segment_dur=5.0, dip_ratio=0.35, min_dip_dur=0.1, min_split_piece=1.0
        )

        # Short segment unchanged
        assert result[0] == {"start": 0.0, "end": 4.0}
        # Long segment split
        long_pieces = [s for s in result if s["start"] >= 5.0]
        assert len(long_pieces) >= 2

    def test_empty_input(self):
        """Empty speech_ts returns empty list."""
        waveform = np.zeros(16000, dtype=np.float32)
        result = split_at_energy_dips([], waveform)
        assert result == []

    def test_output_is_sorted(self):
        """Result segments are always sorted by start time."""
        waveform = make_speech_then_silence_then_speech(total_dur=10.0, amplitude=0.5)
        speech_ts = [{"start": 0.0, "end": 10.0}]
        result = split_at_energy_dips(
            speech_ts, waveform, SAMPLE_RATE,
            min_segment_dur=5.0, min_dip_dur=0.1
        )
        starts = [s["start"] for s in result]
        assert starts == sorted(starts)

    def test_pieces_do_not_overlap(self):
        """No two output segments should overlap."""
        waveform = make_speech_then_silence_then_speech(total_dur=10.0, amplitude=0.5)
        speech_ts = [{"start": 0.0, "end": 10.0}]
        result = split_at_energy_dips(
            speech_ts, waveform, SAMPLE_RATE,
            min_segment_dur=5.0, min_dip_dur=0.1
        )
        for i in range(len(result) - 1):
            assert result[i]["end"] <= result[i + 1]["start"] + 1e-4, (
                f"Overlap between segment {i} and {i+1}: {result[i]} / {result[i+1]}"
            )
