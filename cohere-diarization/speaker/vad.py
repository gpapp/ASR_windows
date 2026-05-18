"""Voice Activity Detection (VAD) utilities."""
import numpy as np


def split_at_energy_dips(
    speech_ts: list[dict],
    waveform: "np.ndarray",
    sample_rate: int = 16000,
    min_segment_dur: float = 5.0,
    frame_ms: float = 20.0,
    dip_ratio: float = 0.35,
    min_dip_dur: float = 0.15,
    min_split_piece: float = 1.0,
) -> list[dict]:
    """Split long VAD segments at local energy dips to reduce cross-speaker window contamination.

    When two speakers alternate with minimal silence, Silero VAD treats them as a single
    speech segment. Sliding windows extracted from that segment straddle the speaker
    boundary, producing blended embeddings that confuse clustering and cause spill-over.

    This function post-processes the VAD output by re-examining each segment that is
    longer than `min_segment_dur` and splitting it wherever short-time RMS energy drops
    below `dip_ratio * median_energy` for at least `min_dip_dur` seconds — a reliable
    indicator of a natural pause between speakers even when VAD didn't detect it.

    Args:
        speech_ts:        VAD output — list of {"start": float, "end": float}.
        waveform:         1-D float32 numpy array of the full audio at `sample_rate`.
        sample_rate:      Audio sample rate (default 16000).
        min_segment_dur:  Only split segments longer than this (seconds, default 5.0).
        frame_ms:         Short-time energy frame size in ms (default 20 ms).
        dip_ratio:        Energy frames below (dip_ratio × median_energy) are "dip" frames.
        min_dip_dur:      Minimum contiguous dip duration to count as a split point (seconds).
        min_split_piece:  Minimum duration of a piece after splitting (seconds).
                          Split points that would create shorter pieces are skipped.

    Returns:
        Refined list of {"start": float, "end": float} segments (always sorted).
    """
    frame_samples = int(frame_ms / 1000 * sample_rate)
    hop_samples = frame_samples  # non-overlapping frames for efficiency
    min_dip_frames = max(1, int(min_dip_dur * sample_rate / hop_samples))

    result = []
    for seg in speech_ts:
        dur = seg["end"] - seg["start"]
        if dur < min_segment_dur:
            result.append(seg)
            continue

        s = int(seg["start"] * sample_rate)
        e = int(seg["end"] * sample_rate)
        chunk = waveform[s:e]

        # Compute short-time RMS energy per frame
        n_frames = (len(chunk) - frame_samples) // hop_samples
        if n_frames < 2:
            result.append(seg)
            continue

        energies = np.array([
            np.sqrt(np.mean(chunk[i * hop_samples: i * hop_samples + frame_samples] ** 2))
            for i in range(n_frames)
        ])

        median_e = np.median(energies)
        if median_e < 1e-6:
            result.append(seg)
            continue

        # Identify dip frames
        is_dip = energies < (dip_ratio * median_e)

        # Find contiguous dip regions long enough to split on
        split_times = []  # seconds relative to segment start
        i = 0
        while i < len(is_dip):
            if is_dip[i]:
                j = i
                while j < len(is_dip) and is_dip[j]:
                    j += 1
                dip_len = j - i
                if dip_len >= min_dip_frames:
                    # Split at the center of the dip
                    center = (i + dip_len // 2) * hop_samples / sample_rate
                    split_times.append(center)
                i = j
            else:
                i += 1

        if not split_times:
            result.append(seg)
            continue

        # Build pieces, discarding any that are too short
        boundaries = [0.0] + split_times + [dur]
        pieces = []
        for k in range(len(boundaries) - 1):
            piece_start = seg["start"] + boundaries[k]
            piece_end = seg["start"] + boundaries[k + 1]
            if (piece_end - piece_start) >= min_split_piece:
                pieces.append({"start": round(piece_start, 4), "end": round(piece_end, 4)})

        if len(pieces) <= 1:
            result.append(seg)
        else:
            result.extend(pieces)

    result.sort(key=lambda x: x["start"])
    return result


def run_vad_chunked(waveform_tensor, vad_model, get_speech_timestamps, sample_rate=16000, 
                   chunk_duration=30, overlap=5, threshold=0.5, min_speech_duration_ms=250):
    """Run VAD on audio in chunks for better performance on long audio."""
    total_samples = waveform_tensor.shape[-1]
    chunk_samples = int(chunk_duration * sample_rate)
    stride_samples = int((chunk_duration - overlap) * sample_rate)
    
    all_speech_ts = []
    
    for start in range(0, total_samples, stride_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform_tensor[..., start:end]
        
        ts = get_speech_timestamps(
            chunk,
            vad_model,
            sampling_rate=sample_rate,
            return_seconds=True,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms
        )
        
        for t in ts:
            all_speech_ts.append({
                "start": t["start"] + start / sample_rate,
                "end": t["end"] + start / sample_rate
            })
    
    if not all_speech_ts:
        return []
    
    # Merge overlapping segments
    all_speech_ts.sort(key=lambda x: x["start"])
    merged = [all_speech_ts[0]]
    for seg in all_speech_ts[1:]:
        if seg["start"] <= merged[-1]["end"] + 0.1:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)
    
    return merged


def run_vad_onnx(waveform_tensor, vad_session, sample_rate=16000, 
                 chunk_duration=30, overlap=5, threshold=0.5, min_speech_duration_ms=250):
    """Run VAD using ONNX model with DirectML acceleration."""
    import torch
    
    total_samples = waveform_tensor.shape[-1]
    chunk_samples = int(chunk_duration * sample_rate)
    stride_samples = int((chunk_duration - overlap) * sample_rate)
    
    all_speech_ts = []
    
    for start in range(0, total_samples, stride_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform_tensor[..., start:end]
        
        # Convert to numpy and prepare input
        chunk_np = chunk.squeeze(0).numpy().astype(np.float32)
        
        # Silero ONNX expects [batch, time] input
        input_tensor = np.expand_dims(chunk_np, axis=0)
        
        # Run inference
        out = vad_session.run(None, {"input": input_tensor})
        probs = out[0][0]  # [time]
        
        # Find speech segments
        speech_mask = probs > threshold
        
        # Find contiguous segments
        if speech_mask.any():
            indices = np.where(np.diff(np.concatenate([[False], speech_mask, [False]])))[0]
            for i in range(0, len(indices), 2):
                if i + 1 < len(indices):
                    seg_start = indices[i] / 100  # 100 Hz output
                    seg_end = indices[i + 1] / 100
                    seg_dur_ms = (seg_end - seg_start) * 1000
                    if seg_dur_ms >= min_speech_duration_ms:
                        all_speech_ts.append({
                            "start": seg_start + start / sample_rate,
                            "end": seg_end + start / sample_rate
                        })
    
    if not all_speech_ts:
        return []
    
    # Merge overlapping segments
    all_speech_ts.sort(key=lambda x: x["start"])
    merged = [all_speech_ts[0]]
    for seg in all_speech_ts[1:]:
        if seg["start"] <= merged[-1]["end"] + 0.1:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)
    
    return merged
