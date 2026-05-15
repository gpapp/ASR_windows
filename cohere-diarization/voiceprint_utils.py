#!/usr/bin/env python3
"""
Voiceprint utilities - common functions for creating and refining voiceprints.

Includes:
- Audio loading and processing
- Embedding extraction (from speaker.embedding)
- Pitch and energy computation (from speaker.embedding)
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
from tqdm import tqdm

# Re-export from speaker module for backwards compatibility
from speaker.embedding import extract_embedding, compute_pitch, compute_energy


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





# Cached embedding session - created once and reused
_cached_embedding_session = None
_cached_embedding_path = None


def get_embedding_session(settings) -> ort.InferenceSession:
    """Get cached ONNX embedding session. Creates once, reuses for all calls."""
    global _cached_embedding_session, _cached_embedding_path
    
    from server import ensure_embedding_model
    
    emb_path = ensure_embedding_model(
        settings.embedding_model_repo,
        settings.embedding_model_filename,
        settings.hf_token
    )
    
    # Return cached session if already created for this path
    if _cached_embedding_session is not None and _cached_embedding_path == emb_path:
        return _cached_embedding_session
    
    # Create new session
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _cached_embedding_session = ort.InferenceSession(emb_path, opts, providers=["CPUExecutionProvider"])
    _cached_embedding_path = emb_path
    
    return _cached_embedding_session


def init_embedding_session(settings) -> ort.InferenceSession:
    """Initialize the ONNX embedding session. DEPRECATED - use get_embedding_session() instead."""
    return get_embedding_session(settings)


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
    match_threshold: float = 0.3,
    single_speaker_threshold: float = 0.8,
    include_unknown: bool = False,
) -> tuple[list[dict], np.ndarray, int]:
    """
    Identify speakers in audio using existing voiceprints.
    Only returns segments that are clearly a SINGLE speaker.
    Ambiguous segments (multiple speakers) are skipped.

    Args:
        single_speaker_threshold: Fraction of windows that must match the dominant speaker
        include_unknown: If True, also return segments that don't match any known voiceprint,
                         labelled as UNKNOWN_1, UNKNOWN_2, etc. based on embedding similarity.

    Returns:
        Tuple of (segments, waveform, sample_rate) where segments is a list of dicts
        with: start, end, speaker, confidence. Waveform is float32 numpy array at sample_rate.
    """
    # Decode audio to raw PCM via ffmpeg pipe — handles any container (MKV, MP4, WAV, ...)
    # without writing a temp file and without depending on torchaudio backends.
    cmd = [
        "ffmpeg", "-nostdin", "-i", audio_path,
        "-ar", str(sample_rate), "-ac", "1",
        "-f", "f32le", "-loglevel", "error", "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed decoding {audio_path}: {result.stderr.decode()}")
    waveform = np.frombuffer(result.stdout, dtype=np.float32).copy()
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
        return [], waveform, sample_rate

    # Split long VAD segments at energy dips — same pre-processing as the server
    # diarization pipeline. Reduces cross-speaker window contamination when two
    # speakers alternate with minimal silence.
    from speaker.vad import split_at_energy_dips
    speech_ts = split_at_energy_dips(speech_ts, waveform, sample_rate=sample_rate)

    # Import audio helpers
    from speaker.audio import generate_sliding_windows, extract_fbank, refine_speaker_boundaries

    # Prepare known speaker embeddings
    known_embeddings = {}
    for name, profile in voiceprints.items():
        if "embedding" in profile:
            known_embeddings[name] = np.array(profile["embedding"])

    if not known_embeddings:
        return [], waveform, sample_rate

    # Track new speakers (unknown speakers get assigned SPEAKER1, SPEAKER2, etc.)
    new_speaker_counter = 1
    new_speaker_map = {}  # Maps internal ID to SPEAKER name

    # Process each VAD segment - check if it's a single speaker
    all_segments = []
    unknown_segments = []  # Collect unmatched segments with embeddings for clustering

    total_segments = len(speech_ts)
    for ts in tqdm(speech_ts, desc="Identifying speakers", unit="segment"):
        start_sample = int(ts['start'] * sample_rate)
        end_sample = int(ts['end'] * sample_rate)
        segment_wav = waveform_tensor[:, start_sample:end_sample]

        # Generate sliding windows and extract embeddings
        # Use same parameters as server diarization pipeline (2.0s window, 0.75s stride)
        # to keep contamination at speaker boundaries below 0.75s.
        windows, start_times = generate_sliding_windows(segment_wav, sample_rate, window_sec=2.0, stride_sec=0.75)

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

        batch = torch.stack(padded_fbanks)  # [N, 1, max_len, 80]
        # Apply CMN (Cepstral Mean Normalization) — critical for ECAPA-TDNN
        batch = batch - batch.mean(dim=2, keepdim=True)
        batch = batch.squeeze(1).numpy()  # [N, max_len, 80]

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
        # Count matches per speaker (including None for unmatched)
        speaker_counts = {}
        unmatched_count = 0
        for spk, conf, dist in window_matches:
            if spk is None:
                unmatched_count += 1
                continue
            if spk not in speaker_counts:
                speaker_counts[spk] = {"count": 0, "total_conf": 0, "total_dist": 0}
            speaker_counts[spk]["count"] += 1
            speaker_counts[spk]["total_conf"] += conf
            speaker_counts[spk]["total_dist"] += dist

        total_windows = len(window_matches)

        if not speaker_counts:
            # No known speaker matched — this is an unknown speaker segment
            if include_unknown:
                mean_emb = embeddings.mean(axis=0)
                mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
                unknown_segments.append({
                    "start": round(ts['start'], 3),
                    "end": round(ts['end'], 3),
                    "embedding": mean_emb,
                })
            continue

        # Find dominant speaker
        dominant_speaker = max(speaker_counts.items(), key=lambda x: x[1]["count"])
        dominant_name = dominant_speaker[0]
        dominant_count = dominant_speaker[1]["count"]

        # Check if this is clearly a single speaker
        # At least 80% of windows must match the same speaker
        dominant_ratio = dominant_count / total_windows

        if dominant_ratio < single_speaker_threshold:
            # Ambiguous - if mostly unmatched, treat as unknown
            if include_unknown and unmatched_count / total_windows > 0.5:
                mean_emb = embeddings.mean(axis=0)
                mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
                unknown_segments.append({
                    "start": round(ts['start'], 3),
                    "end": round(ts['end'], 3),
                    "embedding": mean_emb,
                })
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

    # Cluster unknown segments into distinct speakers (same logic as server.py diarization)
    unknown_centroids: dict[str, np.ndarray] = {}
    if include_unknown and unknown_segments:
        from sklearn.cluster import AgglomerativeClustering
        from scipy.spatial.distance import cosine

        unknown_embeddings = np.array([s["embedding"] for s in unknown_segments])

        if len(unknown_segments) == 1:
            labels = np.array([0])
        else:
            # Initial clustering with distance threshold
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=0.35,
            )
            labels = clusterer.fit_predict(unknown_embeddings)

            n_clusters = len(set(int(l) for l in labels))

            # Cap at 10 clusters max (like server caps at 15, then merges)
            max_clusters = 10
            if n_clusters > max_clusters:
                clusterer = AgglomerativeClustering(
                    n_clusters=max_clusters,
                    metric="cosine",
                    linkage="average",
                )
                labels = clusterer.fit_predict(unknown_embeddings)
                n_clusters = max_clusters

            # Greedy merge: merge closest cluster pairs below threshold
            if n_clusters > 1:
                merge_threshold = 0.25
                cluster_ids = sorted(set(int(l) for l in labels))
                cluster_avgs = {}
                for cid in cluster_ids:
                    mask = labels == cid
                    avg = unknown_embeddings[mask].mean(axis=0)
                    cluster_avgs[cid] = avg / (np.linalg.norm(avg) + 1e-12)

                changed = True
                while changed:
                    changed = False
                    ids = sorted(cluster_avgs.keys())
                    for i_idx in range(len(ids)):
                        for j_idx in range(i_idx + 1, len(ids)):
                            id_i, id_j = ids[i_idx], ids[j_idx]
                            if id_i not in cluster_avgs or id_j not in cluster_avgs:
                                continue
                            dist = cosine(cluster_avgs[id_i], cluster_avgs[id_j])
                            if dist < merge_threshold:
                                # Merge j into i
                                labels[labels == id_j] = id_i
                                mask_i = labels == id_i
                                avg = unknown_embeddings[mask_i].mean(axis=0)
                                cluster_avgs[id_i] = avg / (np.linalg.norm(avg) + 1e-12)
                                del cluster_avgs[id_j]
                                changed = True
                                break
                        if changed:
                            break

        # Remap labels to contiguous 1-based numbering
        unique_labels = sorted(set(int(l) for l in labels))
        label_map = {old: new + 1 for new, old in enumerate(unique_labels)}

        # Build per-UNKNOWN centroid from clustered embeddings (for boundary refinement)
        unknown_centroids: dict[str, np.ndarray] = {}
        unk_emb_matrix = np.array([s["embedding"] for s in unknown_segments])
        for old_label in unique_labels:
            mask = labels == old_label
            avg = unk_emb_matrix[mask].mean(axis=0)
            unknown_centroids[f"UNKNOWN_{label_map[old_label]}"] = avg / (np.linalg.norm(avg) + 1e-12)

        for seg, label in zip(unknown_segments, labels):
            all_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": f"UNKNOWN_{label_map[int(label)]}",
                "confidence": 0.0,
                "distance": 1.0,
            })

    # Merge adjacent segments for same speaker
    if not all_segments:
        return [], waveform, sample_rate

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

    # Refine speaker boundaries: all sub-windows across all transitions are
    # collected and embedded in a single batched ONNX call, so the cost is
    # one inference round-trip regardless of the number of transitions.
    if len(merged) >= 2:
        centroids: dict[str, np.ndarray] = {}
        for name, emb in known_embeddings.items():
            centroids[name] = np.array(emb)
        if include_unknown and unknown_segments:
            centroids.update(unknown_centroids)
        active_speakers = set(s["speaker"] for s in merged)
        active_centroids = {k: v for k, v in centroids.items() if k in active_speakers}
        if len(active_centroids) >= 2:
            merged = refine_speaker_boundaries(
                merged,
                waveform_tensor,
                embedding_session,
                active_centroids,
                sample_rate=sample_rate,
            )

    return merged, waveform, sample_rate


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
    for i, seg in enumerate(tqdm(matching, desc=f"Extracting {speaker_name}", unit="seg")):
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
    min_duration: float = 1.5,
    block_sec: float = 600.0,
) -> dict:
    """
    Refine a voiceprint by processing all segments in a directory.

    All audio files are loaded first, then embedded in batched ONNX blocks of
    ``block_sec`` seconds (default 10 minutes) so the inference engine is used
    efficiently regardless of how many files are present.

    Args:
        voiceprints_file: Path to voiceprints.json
        speaker_name: Speaker to refine
        segments_dir: Directory containing extracted segments
        min_duration: Minimum segment duration
        block_sec: Total audio seconds to process per ONNX call (default 600 = 10 min)

    Returns:
        Updated voiceprint dict
    """
    from server import Settings, state
    from speaker.embedding import batch_embed_files

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

    wav_files = sorted(segments_dir.glob("*.wav")) + sorted(segments_dir.glob("*.mp3")) + sorted(segments_dir.glob("*.flac"))
    if not wav_files:
        raise ValueError(f"No .wav, .mp3, or .flac files found in {segments_dir}")

    # ------------------------------------------------------------------ #
    # Pass 1: load all audio files, filter by duration.
    # ------------------------------------------------------------------ #
    valid_files   = []
    waveforms     = []
    sample_rates  = []
    all_durations = []
    all_pitches   = []
    all_energies  = []
    pitch_durations = []

    print(f"[INFO] Loading {len(wav_files)} files...")
    for wav_file in tqdm(wav_files, desc=f"Loading {speaker_name}", unit="file"):
        try:
            waveform, sr = load_audio_segment(str(wav_file), 0, None)
            duration = waveform.shape[-1] / sr
            if duration < min_duration:
                print(f"[SKIP] {wav_file.name} too short: {duration:.1f}s")
                continue

            pitch, pitch_std = compute_pitch(waveform, sr)
            energy = compute_energy(waveform)

            valid_files.append(wav_file)
            waveforms.append(waveform)
            sample_rates.append(sr)
            all_durations.append(duration)
            all_energies.append(energy)
            if pitch > 0:
                all_pitches.append(pitch)
                pitch_durations.append(duration)

        except Exception as e:
            print(f"[WARN] Failed to load {wav_file.name}: {e}")
            continue

    if not waveforms:
        raise ValueError("No valid segments processed")

    total_duration = sum(all_durations)
    print(f"[INFO] Loaded {len(waveforms)} segments: {total_duration:.1f}s total")
    print(f"[INFO] Embedding in {block_sec:.0f}s blocks...")

    # ------------------------------------------------------------------ #
    # Pass 2: batch-embed all files — one (or a few) ONNX calls total.
    # ------------------------------------------------------------------ #
    embeddings = batch_embed_files(waveforms, sample_rates, all_durations,
                                   embedding_session, block_sec=block_sec)

    all_embeddings = []
    valid_durations = []
    for i, emb in enumerate(embeddings):
        if emb is None:
            print(f"[WARN] No embeddable windows in {valid_files[i].name}")
            continue
        all_embeddings.append(emb)
        valid_durations.append(all_durations[i])

    if not all_embeddings:
        raise ValueError("No valid embeddings produced")

    valid_count = len(all_embeddings)
    print(f"[INFO] Embedded {valid_count} segments")

    # ------------------------------------------------------------------ #
    # Pass 3: duration-weighted averaging — identical logic to before.
    # ------------------------------------------------------------------ #
    durations_array = np.array(valid_durations)
    weights = durations_array / durations_array.sum()
    combined_emb = np.average(np.stack(all_embeddings), axis=0, weights=weights)
    combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-12)

    current = voiceprints.get(speaker_name, {})
    prev_duration = current.get("total_speech_sec", 0)
    prev_embedding = current.get("embedding")

    if prev_embedding is not None and prev_duration > 0 and total_duration > 0:
        prev_emb = np.array(prev_embedding)
        prev_emb = prev_emb / (np.linalg.norm(prev_emb) + 1e-12)
        prev_weight = prev_duration / (prev_duration + total_duration)
        new_weight = total_duration / (prev_duration + total_duration)
        combined_emb = prev_emb * prev_weight + combined_emb * new_weight
        combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-12)

    combined_emb = combined_emb.tolist()

    if all_pitches and pitch_durations and len(all_pitches) == len(pitch_durations):
        avg_pitch = np.average(np.array(all_pitches), weights=np.array(pitch_durations))
    else:
        avg_pitch = np.mean(all_pitches) if all_pitches else 0.0

    avg_energy  = np.average(np.array(all_energies), weights=all_durations)
    pitch_std   = np.std(all_pitches) if len(all_pitches) > 1 else 0.0

    prev_pitch     = current.get("pitch_hz", 0)
    prev_pitch_std = current.get("pitch_std", 0)
    prev_energy    = current.get("energy_rms", 0)
    new_duration   = prev_duration + total_duration

    if prev_duration > 0 and total_duration > 0:
        prev_weight = prev_duration / new_duration
        new_weight  = total_duration / new_duration
        new_pitch   = (prev_pitch * prev_weight + avg_pitch * new_weight) if avg_pitch > 0 else prev_pitch
        new_energy  = prev_energy * prev_weight + avg_energy * new_weight
    elif avg_pitch > 0:
        new_pitch  = avg_pitch
        new_energy = avg_energy
    else:
        new_pitch  = prev_pitch
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
