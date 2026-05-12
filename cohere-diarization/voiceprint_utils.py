#!/usr/bin/env python3
"""
Voiceprint utilities - common functions for creating and refining voiceprints.

Includes:
- Audio loading and processing
- Embedding extraction
- Pitch and energy computation
- Segment extraction from diarization outputs
- Voiceprint refinement with validation
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import onnxruntime as ort
import soundfile as sf


def parse_time(time_str: str) -> float:
    """Parse time string (HH:MM:SS or MM:SS or seconds) to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(time_str)


def format_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def ensure_wav(file_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Ensure audio is converted to WAV format if needed."""
    file_path = Path(file_path)
    wav_path = file_path.with_suffix(".wav")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_path = output_dir / (file_path.stem + ".wav")

    if wav_path.exists():
        return wav_path

    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(file_path),
        "-ar", "16000", "-ac", "1", "-nostdin", str(wav_path)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    return wav_path


def load_audio_segment(wav_path: str, start_sec: float, end_sec: float):
    """Load audio segment from WAV file."""
    info = sf.info(wav_path)
    sample_rate = info.samplerate

    start_sample = int(start_sec * sample_rate)
    if end_sec is None:
        end_sample = None  # Read to end
    else:
        end_sample = int(end_sec * sample_rate)
        end_sample = min(end_sample, info.frames)

    segment, _ = sf.read(wav_path, start=start_sample, stop=end_sample)

    if len(segment.shape) == 1:
        segment = segment.reshape(1, -1)
    else:
        segment = segment.T

    if sample_rate != 16000:
        waveform = torch.tensor(segment, dtype=torch.float32)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        sample_rate = 16000
    else:
        waveform = torch.tensor(segment, dtype=torch.float32)

    return waveform, sample_rate


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
            # Apply CMN per sub-segment (critical for ECAPA-TDNN)
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

    centroid = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    return centroid.tolist()


def compute_pitch(waveform, sample_rate):
    """Estimate pitch using autocorrelation."""
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


def init_embedding_session(settings) -> ort.InferenceSession:
    """Initialize the ONNX embedding session."""
    from server import ensure_embedding_model

    emb_path = ensure_embedding_model(
        settings.embedding_model_repo,
        settings.embedding_model_filename,
        settings.hf_token
    )

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(emb_path, opts, providers=["CPUExecutionProvider"])


def load_voiceprints(path: Path) -> dict:
    """Load voiceprints from JSON file."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_voiceprints(voiceprints: dict, path: Path):
    """Save voiceprints to JSON file."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(voiceprints, f, indent=2)


def identify_speakers_in_audio(
    audio_path: str,
    voiceprints: dict,
    embedding_session: ort.InferenceSession,
    vad_model=None,
    get_speech_timestamps=None,
    sample_rate: int = 16000,
    vad_threshold: float = 0.5,
    vad_min_speech_ms: int = 250,
    match_threshold: float = 0.4,
    single_speaker_threshold: float = 0.8,
) -> list[dict]:
    """
    Identify speakers in audio using existing voiceprints.
    Only returns segments that are clearly a SINGLE speaker.
    Ambiguous segments (multiple speakers) are skipped.

    Args:
        single_speaker_threshold: Fraction of windows that must match the dominant speaker

    Returns:
        List of dicts with: start, end, speaker, confidence
    """
    import librosa
    import soundfile as sf

    # Load audio using soundfile (faster than librosa)
    waveform, sr = sf.read(audio_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)  # Stereo to mono
    waveform = waveform.astype(np.float32)
    if sr != sample_rate:
        # Use torchaudio for resampling (already imported)
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
        waveform_tensor = torchaudio.functional.resample(waveform_tensor, orig_freq=sr, new_freq=sample_rate)
        waveform = waveform_tensor.squeeze(0).numpy()
        sr = sample_rate
    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()

    # Run VAD to get speech segments
    if vad_model and get_speech_timestamps:
        speech_ts = get_speech_timestamps(
            waveform_tensor,
            vad_model,
            sampling_rate=sample_rate,
            return_seconds=True,
            threshold=vad_threshold,
            min_speech_duration_ms=vad_min_speech_ms
        )
    else:
        # Simple energy-based VAD fallback
        speech_ts = []
        frame_size = int(0.03 * sample_rate)  # 30ms
        hop_size = int(0.01 * sample_rate)    # 10ms
        energy_threshold = 0.01

        for start in range(0, len(waveform) - frame_size, hop_size):
            frame = waveform[start:start + frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            if energy > energy_threshold:
                speech_ts.append({"start": start / sample_rate, "end": (start + frame_size) / sample_rate})

        # Merge consecutive segments
        if speech_ts:
            merged = [speech_ts[0].copy()]
            for seg in speech_ts[1:]:
                if seg["start"] - merged[-1]["end"] < 0.5:
                    merged[-1]["end"] = seg["end"]
                else:
                    merged.append(seg.copy())
            speech_ts = merged
        else:
            speech_ts = [{"start": 0, "end": len(waveform) / sample_rate}]

    if not speech_ts:
        return []

    # Import server functions for embedding extraction
    from server import generate_sliding_windows, extract_fbank

    # Prepare known speaker embeddings
    known_embeddings = {}
    for name, profile in voiceprints.items():
        if "embedding" in profile:
            known_embeddings[name] = np.array(profile["embedding"])

    if not known_embeddings:
        return []

    # Track new speakers (unknown speakers get assigned SPEAKER1, SPEAKER2, etc.)
    new_speaker_counter = 1
    new_speaker_map = {}  # Maps internal ID to SPEAKER name

    # Process each VAD segment - check if it's a single speaker
    all_segments = []

    from tqdm import tqdm

    total_segments = len(speech_ts)
    for ts in tqdm(speech_ts, desc="Identifying speakers", unit="segment"):
        start_sample = int(ts['start'] * sample_rate)
        end_sample = int(ts['end'] * sample_rate)
        segment_wav = waveform_tensor[:, start_sample:end_sample]

        # Generate sliding windows and extract embeddings
        windows, start_times = generate_sliding_windows(segment_wav, sample_rate, window_sec=3.0, stride_sec=2.0)

        all_fbanks = []
        window_info = []

        for w, rel_start in zip(windows, start_times):
            chunk_duration = w.shape[-1] / sample_rate
            if chunk_duration >= 1.5:
                target_length = 4800
                if w.shape[-1] < target_length:
                    w = torch.nn.functional.pad(w, (0, target_length - w.shape[-1]))
                all_fbanks.append(extract_fbank(w, sample_rate))
                window_info.append({
                    "start": ts['start'] + rel_start,
                    "end": ts['start'] + rel_start + chunk_duration
                })

        if not all_fbanks:
            continue

        # Batch embed
        max_len = max(fb.shape[1] for fb in all_fbanks)
        padded_fbanks = []
        for fb in all_fbanks:
            if fb.shape[1] < max_len:
                fb = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
            padded_fbanks.append(fb)

        batch = torch.stack(padded_fbanks).squeeze(1).numpy()

        embeddings = embedding_session.run(None, {embedding_session.get_inputs()[0].name: batch})[0]

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

        # Batch compare all embeddings to all known voiceprints
        # Build matrix of known embeddings (n_known x embedding_dim)
        speaker_names = list(known_embeddings.keys())
        known_matrix = np.array([known_embeddings[name] for name in speaker_names])

        # Compute cosine distances: (n_windows x n_known)
        # cosine(a, b) = 1 - dot(a, b) / (||a|| * ||b||)
        # Since embeddings are already normalized, this simplifies
        distances = 1.0 - (embeddings @ known_matrix.T)

        # Find best match for each window
        best_indices = np.argmin(distances, axis=1)
        best_distances = distances[np.arange(len(embeddings)), best_indices]

        window_matches = []
        for i, (idx, dist) in enumerate(zip(best_indices, best_distances)):
            if dist <= match_threshold:
                best_match = speaker_names[idx]
                confidence = 1.0 - dist
                window_matches.append((best_match, confidence, dist))
            else:
                window_matches.append((None, 0.0, 1.0))

        # Check if this is a SINGLE speaker segment
        # Count matches per speaker
        speaker_counts = {}
        for spk, conf, dist in window_matches:
            if spk is None:
                continue
            if spk not in speaker_counts:
                speaker_counts[spk] = {"count": 0, "total_conf": 0, "total_dist": 0}
            speaker_counts[spk]["count"] += 1
            speaker_counts[spk]["total_conf"] += conf
            speaker_counts[spk]["total_dist"] += dist

        if not speaker_counts:
            continue  # No identifiable speakers in this segment

        # Find dominant speaker
        total_windows = len(window_matches)
        dominant_speaker = max(speaker_counts.items(), key=lambda x: x[1]["count"])
        dominant_name = dominant_speaker[0]
        dominant_count = dominant_speaker[1]["count"]

        # Check if this is clearly a single speaker
        # At least 80% of windows must match the same speaker
        dominant_ratio = dominant_count / total_windows

        if dominant_ratio < single_speaker_threshold:
            # Ambiguous - multiple speakers in this segment, skip it
            continue

        # This is a single speaker segment - use the dominant speaker
        avg_conf = dominant_speaker[1]["total_conf"] / dominant_count
        avg_dist = dominant_speaker[1]["total_dist"] / dominant_count

        # Add segment
        all_segments.append({
            "start": round(ts['start'], 3),
            "end": round(ts['end'], 3),
            "speaker": dominant_name,
            "confidence": round(avg_conf, 3),
            "distance": round(avg_dist, 4)
        })

    # Merge adjacent segments for same speaker
    if not all_segments:
        return []

    all_segments.sort(key=lambda x: x["start"])
    merged = [all_segments[0].copy()]

    for seg in all_segments[1:]:
        if (seg["speaker"] == merged[-1]["speaker"] and
            seg["start"] - merged[-1]["end"] < 1.0):
            merged[-1]["end"] = seg["end"]
            # Update confidence to weighted average
            dur1 = merged[-1]["end"] - merged[-1]["start"]
            dur2 = seg["end"] - seg["start"]
            merged[-1]["confidence"] = round(
                (merged[-1]["confidence"] * dur1 + seg["confidence"] * dur2) / (dur1 + dur2), 3
            )
        else:
            merged.append(seg.copy())

    return merged


def extract_speaker_segments(
    audio_file: str,
    diarization_file: str,
    speaker_name: str,
    output_dir: str,
    min_duration: float = 1.5
) -> list[dict]:
    """
    Extract all audio segments matching a specific speaker from diarization output.

    Args:
        audio_file: Path to audio/video file
        diarization_file: Path to JSON diarization output
        speaker_name: Speaker identifier (e.g., "SPEAKER_00" or actual name)
        output_dir: Output directory for extracted segments
        min_duration: Minimum segment duration in seconds

    Returns:
        List of extracted segment info dicts
    """
    audio_file = Path(audio_file)
    diarization_file = Path(diarization_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(diarization_file, "r", encoding="utf-8") as f:
        diarization = json.load(f)

    segments = diarization.get("segments", [])
    matching = [s for s in segments if s.get("speaker") == speaker_name]

    if not matching:
        print(f"[WARN] No segments found for speaker '{speaker_name}'")
        print(f"  Available speakers: {set(s.get('speaker') for s in segments)}")
        return []

    wav_path = ensure_wav(audio_file, output_dir)

    extracted = []
    for i, seg in enumerate(matching):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        duration = end - start

        if duration < min_duration:
            print(f"[SKIP] Segment {i} too short: {duration:.1f}s < {min_duration}s")
            continue

        output_file = output_dir / f"{speaker_name}_{i:03d}_{format_time(start).replace(':', '')}.wav"

        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(wav_path),
            "-ss", str(start),
            "-to", str(end),
            "-ar", "16000", "-ac", "1", "-nostdin", str(output_file)
        ], capture_output=True)

        if result.returncode != 0:
            print(f"[WARN] Failed to extract segment {i}: {result.stderr.decode()}")
            continue

        extracted.append({
            "index": i,
            "start": start,
            "end": end,
            "duration": duration,
            "file": str(output_file),
            "text": seg.get("text", "")
        })
        print(f"[OK] Extracted: {output_file.name} ({duration:.1f}s)")

    return extracted


def refine_voiceprint_from_segments(
    voiceprints_file: Path,
    speaker_name: str,
    segments_dir: Path,
    min_duration: float = 1.5
) -> dict:
    """
    Refine a voiceprint by processing all segments in a directory.

    Args:
        voiceprints_file: Path to voiceprints.json
        speaker_name: Speaker to refine
        segments_dir: Directory containing extracted segments
        min_duration: Minimum segment duration

    Returns:
        Updated voiceprint dict
    """
    from server import Settings, state

    voiceprints = load_voiceprints(voiceprints_file)

    is_new_speaker = speaker_name not in voiceprints
    if is_new_speaker:
        print(f"[INFO] Creating new voiceprint for '{speaker_name}'")
        voiceprints[speaker_name] = {
            "pitch_hz": 0.0,
            "pitch_std": 0.0,
            "energy_rms": 0.0,
            "total_speech_sec": 0.0,
        }
    else:
        existing = voiceprints[speaker_name]
        print(f"[INFO] Existing voiceprint for '{speaker_name}':")
        print(f"  - Duration: {existing.get('total_speech_sec', 0):.1f}s")
        print(f"  - Pitch: {existing.get('pitch_hz', 0):.1f} Hz")

    settings = Settings()
    embedding_session = init_embedding_session(settings)

    segments_dir = Path(segments_dir)
    if not segments_dir.exists():
        raise ValueError(f"Segments directory not found: {segments_dir}")

    wav_files = sorted(segments_dir.glob("*.wav"))
    if not wav_files:
        raise ValueError(f"No .wav files found in {segments_dir}")

    all_embeddings = []
    all_pitches = []
    all_energies = []
    all_durations = []  # Track durations for weighted averaging
    pitch_durations = []  # Separate list for pitch durations
    total_duration = 0.0
    valid_count = 0

    for wav_file in wav_files:
        try:
            waveform, sr = load_audio_segment(str(wav_file), 0, None)

            duration = waveform.shape[-1] / sr
            if duration < min_duration:
                print(f"[SKIP] {wav_file.name} too short: {duration:.1f}s")
                continue

            emb = extract_embedding(waveform, sr, embedding_session)
            pitch, pitch_std = compute_pitch(waveform, sr)
            energy = compute_energy(waveform)

            all_embeddings.append(np.array(emb))
            all_energies.append(energy)
            all_durations.append(duration)
            total_duration += duration
            valid_count += 1

            if pitch > 0:
                all_pitches.append(pitch)
                pitch_durations.append(duration)

            print(f"[OK] {wav_file.name}: pitch={pitch:.1f}Hz, energy={energy:.4f}")

        except Exception as e:
            print(f"[WARN] Failed to process {wav_file.name}: {e}")
            continue

    if not all_embeddings:
        raise ValueError("No valid segments processed")

    print(f"[INFO] Processed {valid_count} segments: {total_duration:.1f}s total")

    # Duration-weighted averaging for new embeddings
    durations_array = np.array(all_durations)
    weights = durations_array / durations_array.sum()
    combined_emb = np.average(np.stack(all_embeddings), axis=0, weights=weights)
    combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-12)

    # Blend with existing embedding using duration-weighted averaging
    current = voiceprints.get(speaker_name, {})
    prev_duration = current.get("total_speech_sec", 0)
    prev_embedding = current.get("embedding")
    
    if prev_embedding is not None and prev_duration > 0 and total_duration > 0:
        # Duration-weighted blend: existing_weight = prev_duration / (prev_duration + new_duration)
        prev_emb = np.array(prev_embedding)
        prev_emb = prev_emb / (np.linalg.norm(prev_emb) + 1e-12)
        prev_weight = prev_duration / (prev_duration + total_duration)
        new_weight = total_duration / (prev_duration + total_duration)
        combined_emb = prev_emb * prev_weight + combined_emb * new_weight
        combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-12)
    
    combined_emb = combined_emb.tolist()

    # Duration-weighted average for pitch (fallback to simple average if pitch issues)
    if all_pitches and pitch_durations and len(all_pitches) == len(pitch_durations):
        pitches_array = np.array(all_pitches)
        avg_pitch = np.average(pitches_array, weights=np.array(pitch_durations))
    else:
        # Fallback: simple average if lengths don't match
        avg_pitch = np.mean(all_pitches) if all_pitches else 0.0
    
    # Duration-weighted average for energy
    avg_energy = np.average(np.array(all_energies), weights=all_durations)

    # Standard deviation of pitches (unweighted, just for variance measure)
    pitch_std = np.std(all_pitches) if len(all_pitches) > 1 else 0.0

    # Get other values from existing voiceprint
    prev_pitch = current.get("pitch_hz", 0)
    prev_pitch_std = current.get("pitch_std", 0)
    prev_energy = current.get("energy_rms", 0)

    new_duration = prev_duration + total_duration

    # Duration-weighted blend: if existing has 600s and new adds 60s, existing contributes 600/(600+60)=0.91
    if prev_duration > 0 and total_duration > 0:
        prev_weight = prev_duration / new_duration
        new_weight = total_duration / new_duration
        # Weighted average for pitch and energy
        new_pitch = (prev_pitch * prev_weight + avg_pitch * new_weight) if avg_pitch > 0 else prev_pitch
        new_energy = prev_energy * prev_weight + avg_energy * new_weight
    elif avg_pitch > 0:
        new_pitch = avg_pitch
        new_energy = avg_energy
    else:
        new_pitch = prev_pitch
        new_energy = prev_energy

    new_pitch_std = pitch_std if pitch_std > 0 else prev_pitch_std

    voiceprints[speaker_name] = {
        "pitch_hz": round(new_pitch, 1),
        "pitch_std": round(new_pitch_std, 1),
        "energy_rms": round(new_energy, 4),
        "total_speech_sec": round(new_duration, 1),
        "embedding": combined_emb
    }

    save_voiceprints(voiceprints, voiceprints_file)

    action = "Created" if is_new_speaker else "Updated"
    print(f"[SUCCESS] {action} voiceprint for '{speaker_name}'")
    print(f"  - Total duration: {new_duration:.1f}s")
    print(f"  - Pitch: {new_pitch:.1f} Hz")

    return voiceprints[speaker_name]