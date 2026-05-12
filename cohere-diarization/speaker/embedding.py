"""Speaker embedding extraction and utilities."""
import numpy as np
import torch
import onnxruntime as ort


def extract_embedding(waveform, sample_rate, embedding_session: ort.InferenceSession):
    """Extract voice embedding from waveform using ONNX model."""
    from server import generate_sliding_windows, extract_fbank

    windows, start_times = generate_sliding_windows(waveform, sample_rate, window_sec=3.0, stride_sec=1.5)

    if not windows:
        raise ValueError("No valid audio windows found")

    all_fbanks = []
    target_length = 4800

    for w in windows:
        chunk_duration = w.shape[-1] / sample_rate
        if chunk_duration >= 1.5:
            if w.shape[-1] < target_length:
                w = torch.nn.functional.pad(w, (0, target_length - w.shape[-1]))
            fb = extract_fbank(w, sample_rate)
            fb = fb - fb.mean(dim=1, keepdim=True)
            all_fbanks.append(fb)

    if not all_fbanks:
        raise ValueError("No embeddable segments (need >=1.5s)")

    max_len = max(fb.shape[1] for fb in all_fbanks)
    padded_fbanks = []
    for fb in all_fbanks:
        if fb.shape[1] < max_len:
            fb = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
        padded_fbanks.append(fb)

    batch = torch.stack(padded_fbanks).squeeze(1)

    input_onnx = {embedding_session.get_inputs()[0].name: batch.numpy()}
    embeddings = embedding_session.run(None, input_onnx)[0]

    return normalize_embedding(embeddings.mean(axis=0))


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
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    return float(torch.sqrt(torch.mean(waveform ** 2)).numpy())