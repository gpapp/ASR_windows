"""Audio feature extraction utilities for speaker processing."""
import numpy as np
import torch
import torchaudio


def extract_fbank(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Extracts 80-dim log-mel filterbanks from waveform matching WeSpeaker expectations.
    
    Note: CMN is NOT applied here - it's applied per sub-segment in extract_embedding.
    """
    # Ensure waveform is 2D: [1, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Scale waveform to 16-bit PCM range for kaldi.fbank
    waveform = waveform * 32768.0

    # torchaudio.compliance.kaldi.fbank expects [B, T]
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        energy_floor=0.0,
        sample_frequency=sample_rate,
        dither=0.0,
        window_type='hamming'
    )
    
    # NO CMN here - will be applied per sub-segment window
    return fbank.unsqueeze(0)  # [1, frames, 80]


def generate_sliding_windows(waveform: torch.Tensor, sample_rate: int, window_sec: float = 3.0, stride_sec: float = 1.5):
    """Generates overlapping sliding windows from a continuous waveform."""
    window_samples = int(window_sec * sample_rate)
    stride_samples = int(stride_sec * sample_rate)
    total_samples = waveform.shape[-1]

    windows = []
    start_times = []

    if total_samples < window_samples:
        return [waveform], [0.0]

    for start in range(0, total_samples - window_samples + 1, stride_samples):
        windows.append(waveform[:, start:start + window_samples])
        start_times.append(start / sample_rate)

    # Handle the last remaining chunk if it doesn't align perfectly
    last_start = len(windows) * stride_samples if windows else 0
    if last_start < total_samples and (total_samples - last_start) > (sample_rate * 0.1): # min 0.1s
        windows.append(waveform[:, last_start:])
        start_times.append(last_start / sample_rate)

    return windows, start_times


def refine_speaker_boundaries(
    segments: list[dict],
    waveform: torch.Tensor,
    embedding_session,
    cluster_centroids: dict,
    sample_rate: int = 16000,
    search_sec: float = 1.5,
    sub_window_sec: float = 0.5,
    sub_stride_sec: float = 0.1,
    min_segment_dur: float = 0.3,
) -> list[dict]:
    """Refine speaker boundaries after initial clustering.

    At each transition point between two different speakers, re-examines the audio
    in a ±search_sec window around the boundary using fine sub-windows (0.5s, 0.1s
    stride). Each sub-window's embedding is compared to the two adjacent cluster
    centroids; the boundary is moved to the first sub-window where the dominant
    speaker switches allegiance.

    All sub-windows across all transitions are collected first and embedded in a
    single batched ONNX call, eliminating per-window inference overhead.

    Args:
        segments:           Merged speaker segments (dicts with start/end/speaker).
        waveform:           Full audio tensor [1, T] at sample_rate.
        embedding_session:  ONNX InferenceSession for the speaker embedding model.
        cluster_centroids:  {speaker_label: np.ndarray (192-d, L2-normalised)}.
        sample_rate:        Audio sample rate (default 16000).
        search_sec:         Half-width of the refinement search region in seconds.
        sub_window_sec:     Sub-window duration for fine-grained embedding.
        sub_stride_sec:     Stride between sub-windows.
        min_segment_dur:    Minimum segment duration after adjustment; shorter
                            segments produced by refinement are discarded.

    Returns:
        Refined list of speaker segments.
    """
    if len(segments) < 2:
        return segments

    sub_samples = int(sub_window_sec * sample_rate)
    stride_samples = int(sub_stride_sec * sample_rate)
    min_sub_samples = int(0.3 * sample_rate)  # 0.3s minimum embeddable length
    total_samples = waveform.shape[-1]
    input_name = embedding_session.get_inputs()[0].name

    # Pre-build centroid matrix for fast cosine similarity
    spk_labels = list(cluster_centroids.keys())
    centroid_matrix = np.stack([
        np.array(cluster_centroids[s]) for s in spk_labels
    ])  # [N_speakers, D]

    # ------------------------------------------------------------------ #
    # Pass 1: collect every sub-window chunk across all transitions,
    # recording which transition and position each belongs to.
    # ------------------------------------------------------------------ #
    # transition_meta[i] = (left_seg_idx, nominal_boundary, search_start,
    #                        search_end, left_speaker, right_speaker)
    transition_meta = []
    # all_fbanks[k] = fbank tensor for the k-th sub-window
    all_fbanks = []
    # slot_map[k] = (transition_idx, center_t) — maps flat index → transition
    slot_map = []

    refined = list(segments)

    for i in range(len(refined) - 1):
        left = refined[i]
        right = refined[i + 1]
        if left["speaker"] == right["speaker"]:
            continue

        nominal_boundary = left["end"]
        search_start = max(0.0, nominal_boundary - search_sec)
        search_end   = min(total_samples / sample_rate, nominal_boundary + search_sec)
        region_start = int(search_start * sample_rate)
        region_end   = int(search_end   * sample_rate)

        t_idx = len(transition_meta)
        transition_meta.append((i, nominal_boundary, search_start, search_end,
                                 left["speaker"], right["speaker"]))

        pos = region_start
        while pos + min_sub_samples <= region_end:
            end_pos   = min(pos + sub_samples, total_samples)
            chunk     = waveform[:, pos:end_pos]
            if chunk.shape[-1] >= min_sub_samples:
                if chunk.shape[-1] < sub_samples:
                    chunk = torch.nn.functional.pad(chunk, (0, sub_samples - chunk.shape[-1]))
                fb = extract_fbank(chunk, sample_rate)   # [1, T, 80]
                fb = fb - fb.mean(dim=1, keepdim=True)   # CMN
                all_fbanks.append(fb)
                center_t = (pos + min(pos + sub_samples, end_pos)) / 2 / sample_rate
                slot_map.append((t_idx, center_t))
            pos += stride_samples

    if not all_fbanks:
        return refined

    # ------------------------------------------------------------------ #
    # Pass 2: single batched ONNX inference over all sub-windows.
    # ------------------------------------------------------------------ #
    max_len = max(fb.shape[1] for fb in all_fbanks)
    padded  = []
    for fb in all_fbanks:
        if fb.shape[1] < max_len:
            fb = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
        padded.append(fb.squeeze(0))                     # [T, 80]

    batch    = torch.stack(padded).numpy()               # [N, T, 80]
    raw_embs = embedding_session.run(None, {input_name: batch})[0]  # [N, D]
    norms    = np.linalg.norm(raw_embs, axis=1, keepdims=True)
    embs     = raw_embs / np.maximum(norms, 1e-12)       # L2-normalised

    # nearest speaker for every sub-window: [N]
    dists    = 1.0 - (embs @ centroid_matrix.T)          # [N, S]
    nearest  = [spk_labels[int(np.argmin(d))] for d in dists]

    # ------------------------------------------------------------------ #
    # Pass 3: group results back per transition, then walk the candidates.
    # ------------------------------------------------------------------ #
    # candidates_per_transition[t_idx] = [(center_t, speaker), ...]
    candidates_per_transition: dict[int, list] = {
        t: [] for t in range(len(transition_meta))
    }
    for k, (t_idx, center_t) in enumerate(slot_map):
        candidates_per_transition[t_idx].append((center_t, nearest[k]))

    for t_idx, (seg_i, nominal_boundary, search_start, search_end,
                left_spk, right_spk) in enumerate(transition_meta):
        candidates = candidates_per_transition[t_idx]
        if not candidates:
            continue

        last_left_t  = search_start
        first_right_t = search_end

        for center_t, spk in candidates:
            if spk == left_spk:
                last_left_t = center_t
        for center_t, spk in candidates:
            if spk == right_spk and center_t > last_left_t:
                first_right_t = center_t
                break

        new_boundary = round((last_left_t + first_right_t) / 2, 4)

        left  = refined[seg_i]
        right = refined[seg_i + 1]
        if (abs(new_boundary - nominal_boundary) > 0.05 and
                new_boundary - left["start"] >= min_segment_dur and
                right["end"] - new_boundary >= min_segment_dur):
            refined[seg_i]     = dict(left,  end=new_boundary)
            refined[seg_i + 1] = dict(right, start=new_boundary)

    refined = [s for s in refined if s["end"] - s["start"] >= min_segment_dur]
    return refined



def extract_fbank(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Extracts 80-dim log-mel filterbanks from waveform matching WeSpeaker expectations.
    
    Note: CMN is NOT applied here - it's applied per sub-segment in extract_embedding.
    """
    # Ensure waveform is 2D: [1, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Scale waveform to 16-bit PCM range for kaldi.fbank
    waveform = waveform * 32768.0

    # torchaudio.compliance.kaldi.fbank expects [B, T]
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        energy_floor=0.0,
        sample_frequency=sample_rate,
        dither=0.0,
        window_type='hamming'
    )
    
    # NO CMN here - will be applied per sub-segment window
    return fbank.unsqueeze(0)  # [1, frames, 80]


def generate_sliding_windows(waveform: torch.Tensor, sample_rate: int, window_sec: float = 3.0, stride_sec: float = 1.5):
    """Generates overlapping sliding windows from a continuous waveform."""
    window_samples = int(window_sec * sample_rate)
    stride_samples = int(stride_sec * sample_rate)
    total_samples = waveform.shape[-1]

    windows = []
    start_times = []

    if total_samples < window_samples:
        return [waveform], [0.0]

    for start in range(0, total_samples - window_samples + 1, stride_samples):
        windows.append(waveform[:, start:start + window_samples])
        start_times.append(start / sample_rate)

    # Handle the last remaining chunk if it doesn't align perfectly
    last_start = len(windows) * stride_samples if windows else 0
    if last_start < total_samples and (total_samples - last_start) > (sample_rate * 0.1): # min 0.1s
        windows.append(waveform[:, last_start:])
        start_times.append(last_start / sample_rate)

    return windows, start_times
