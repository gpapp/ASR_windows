"""Speaker embedding extraction and utilities."""
import numpy as np
import torch
import onnxruntime as ort

from .audio import generate_sliding_windows, extract_fbank


def extract_embedding(waveform, sample_rate, embedding_session: ort.InferenceSession):
    """Extract voice embedding from waveform using ONNX model."""
    windows, start_times = generate_sliding_windows(waveform, sample_rate, window_sec=3.0, stride_sec=1.5)

    if not windows:
        raise ValueError("No valid audio windows found")

    # Process windows - fbank must be per-window, but collect first
    all_fbanks = []
    target_length = 4800

    for w in windows:
        chunk_duration = w.shape[-1] / sample_rate
        if chunk_duration >= 1.5:
            if w.shape[-1] < target_length:
                w = torch.nn.functional.pad(w, (0, target_length - w.shape[-1]))
            fb = extract_fbank(w, sample_rate)
            all_fbanks.append(fb)

    if not all_fbanks:
        raise ValueError("No embeddable segments (need >=1.5s)")

    # Vectorized: pad first, then stack and apply CMN
    max_len = max(fb.shape[1] for fb in all_fbanks)
    
    # Pad each fb to max_len BEFORE stacking
    padded_fbanks = []
    for fb in all_fbanks:
        if fb.shape[1] < max_len:
            fb_padded = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
        else:
            fb_padded = fb
        padded_fbanks.append(fb_padded)
    
    batch = torch.stack(padded_fbanks, dim=0)  # [N, 1, max_len, 80]
    cmn_batch = batch - batch.mean(dim=2, keepdim=True)  # CMN on all at once
    
    batch = cmn_batch.squeeze(1)  # [N, max_len, 80]

    input_onnx = {embedding_session.get_inputs()[0].name: batch.numpy()}
    embeddings = embedding_session.run(None, input_onnx)[0]

    return normalize_embedding(embeddings.mean(axis=0))


def batch_embed_files(
    waveforms: list,
    sample_rates: list[int],
    durations: list[float],
    embedding_session: ort.InferenceSession,
    block_sec: float = 600.0,
) -> list[np.ndarray | None]:
    """Embed a list of waveforms in large batched ONNX blocks.

    Instead of one ONNX call per file, all sliding-window fbanks from all
    files are collected together and run through the model in blocks of
    ``block_sec`` worth of audio (default 10 minutes). This amortises the
    per-call overhead and saturates the inference engine much more efficiently.

    Args:
        waveforms:   List of waveform tensors [1, T] at their respective sample rates.
        sample_rates: Matching sample rate for each waveform.
        durations:   Matching duration (seconds) for each waveform.
        embedding_session: ONNX InferenceSession.
        block_sec:   Maximum total audio seconds to process in one ONNX call.

    Returns:
        List of L2-normalised embeddings (np.ndarray, shape [D]) in the same
        order as the input waveforms. Files with no embeddable windows get None.
    """
    target_length = int(4800)  # 0.3 s @ 16 kHz
    min_chunk_sec = 1.5
    input_name = embedding_session.get_inputs()[0].name

    # ------------------------------------------------------------------ #
    # Pass 1: collect every fbank window across all files.
    # slot_map[k] = file_index so we can group results later.
    # ------------------------------------------------------------------ #
    all_fbanks: list[torch.Tensor] = []   # each [1, T, 80] after CMN
    slot_map:   list[int]          = []   # file index for each fbank

    for file_idx, (waveform, sr) in enumerate(zip(waveforms, sample_rates)):
        windows, _ = generate_sliding_windows(waveform, sr, window_sec=3.0, stride_sec=1.5)
        for w in windows:
            if w.shape[-1] / sr < min_chunk_sec:
                continue
            if w.shape[-1] < target_length:
                w = torch.nn.functional.pad(w, (0, target_length - w.shape[-1]))
            fb = extract_fbank(w, sr)   # [1, T, 80]
            all_fbanks.append(fb)
            slot_map.append(file_idx)

    if not all_fbanks:
        return [None] * len(waveforms)

    # ------------------------------------------------------------------ #
    # Pass 2: run ONNX in block_sec-sized batches.
    # We approximate audio covered per fbank as 3.0 s (window_sec).
    # ------------------------------------------------------------------ #
    window_sec   = 3.0
    block_size   = max(1, int(block_sec / window_sec))   # fbanks per block
    raw_embs_all = np.empty((len(all_fbanks), embedding_session.get_outputs()[0].shape[-1] or 192),
                            dtype=np.float32)

    for block_start in range(0, len(all_fbanks), block_size):
        block_fbanks = all_fbanks[block_start: block_start + block_size]
        max_len = max(fb.shape[1] for fb in block_fbanks)
        padded  = []
        for fb in block_fbanks:
            if fb.shape[1] < max_len:
                fb = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
            padded.append(fb)
        batch = torch.stack(padded, dim=0)                   # [N, 1, T, 80]
        batch = (batch - batch.mean(dim=2, keepdim=True)).squeeze(1).numpy()  # CMN + [N, T, 80]
        raw_embs_all[block_start: block_start + len(block_fbanks)] = \
            embedding_session.run(None, {input_name: batch})[0]

    # L2-normalise
    norms    = np.linalg.norm(raw_embs_all, axis=1, keepdims=True)
    embs_all = raw_embs_all / np.maximum(norms, 1e-12)

    # ------------------------------------------------------------------ #
    # Pass 3: group embeddings back by file, average → one emb per file.
    # ------------------------------------------------------------------ #
    n_files   = len(waveforms)
    file_embs: list[list[np.ndarray]] = [[] for _ in range(n_files)]
    for k, file_idx in enumerate(slot_map):
        file_embs[file_idx].append(embs_all[k])

    result: list[np.ndarray | None] = []
    for file_idx in range(n_files):
        group = file_embs[file_idx]
        if not group:
            result.append(None)
            continue
        mean_emb = np.mean(np.stack(group), axis=0)
        norm     = np.linalg.norm(mean_emb)
        result.append(mean_emb / (norm + 1e-12))

    return result


def normalize_embedding(emb):
    """L2 normalize embedding vector."""
    if isinstance(emb, list):
        emb = np.array(emb)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.tolist()


def compute_pitch(waveform, sample_rate):
    """Estimate pitch using autocorrelation. Returns (median_hz, std_hz)."""
    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    waveform = waveform.numpy()

    min_lag = int(sample_rate / 300)
    max_lag = int(sample_rate / 80)

    pitches = []
    frame_size = 2048
    hop = 512

    for start in range(0, len(waveform) - frame_size, hop):
        frame = waveform[start:start + frame_size]
        frame = frame - np.mean(frame)

        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]

        if len(corr) == 0 or corr[0] == 0:
            continue

        corr = corr / corr[0]

        search_max = min(max_lag, len(corr))
        if search_max <= min_lag:
            continue

        sub = corr[min_lag:search_max]
        if len(sub) == 0 or sub.max() <= 0:
            continue

        peak_idx = np.argmax(sub)
        peak_val = sub[peak_idx]
        peak_lag = peak_idx + min_lag

        if peak_val < 0.2:
            continue

        f0 = sample_rate / peak_lag
        if 60 < f0 < 400:
            pitches.append(f0)

    if not pitches:
        return 0.0, 0.0

    return float(np.median(pitches)), float(np.std(pitches) if len(pitches) > 1 else 0.0)


def compute_energy(waveform):
    """Compute RMS energy of audio."""
    # Handle multi-channel: average across channels if shape[0] > 1 and it's not batch
    if waveform.dim() > 1 and waveform.shape[0] <= 64:  # Assume channel dim if small
        waveform = waveform.mean(dim=0)
    return float(torch.sqrt(torch.mean(waveform ** 2)).numpy())