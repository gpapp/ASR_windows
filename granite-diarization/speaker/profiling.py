"""Speaker profiling and relabeling utilities."""
import numpy as np
import torch
import torchaudio


def _extract_mfcc_stats(chunk: np.ndarray, sr: int, n_mfcc: int = 13) -> dict:
    """Extract MFCC statistics from audio chunk.
    
    Returns dict with mean and std of first 13 MFCC coefficients.
    """
    try:
        # Convert to tensor for torchaudio
        audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
        
        # Extract MFCCs
        mfcc = torchaudio.compliance.kaldi.mfcc(
            audio_tensor,
            sample_frequency=sr,
            num_mel_bins=23,
            num_ceps=n_mfcc,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
        )  # [T, n_mfcc]
        
        mfcc_np = mfcc.numpy()
        
        # Return mean and std of each coefficient
        stats = {}
        for i in range(n_mfcc):
            stats[f"mfcc{i}_mean"] = float(np.mean(mfcc_np[:, i]))
            stats[f"mfcc{i}_std"] = float(np.std(mfcc_np[:, i]))
        
        return stats
    except Exception:
        return {}


def _extract_spectral_features(chunk: np.ndarray, sr: int) -> dict:
    """Extract spectral features from audio chunk.
    
    Returns spectral centroid, bandwidth, and rolloff.
    """
    try:
        audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
        
        # Spectral centroid
        centroid = torchaudio.transforms.SpectralCentroid(sample_rate=sr)(audio_tensor)
        centroid_mean = float(centroid.mean())
        
        # Spectral rolloff (85% energy)
        spec = torch.abs(torch.stft(torch.from_numpy(chunk), n_fft=512, hop_length=256, 
                                     win_length=512, window=torch.hann_window(512), 
                                     return_complex=True))
        energy_cumsum = torch.cumsum(spec ** 2, dim=0)
        total_energy = energy_cumsum[-1]
        if total_energy > 0:
            rolloff_threshold = 0.85 * total_energy
            rolloff_idx = torch.searchsorted(energy_cumsum, rolloff_threshold)
            rolloff_freq = float(rolloff_idx * sr / 512)
        else:
            rolloff_freq = 0.0
        
        return {
            "spectral_centroid": centroid_mean,
            "spectral_rolloff": rolloff_freq,
        }
    except Exception:
        return {}


def profile_speakers(
    waveform: "torch.Tensor",
    merged_segments: list[dict],
    sample_rate: int = 16000,
) -> dict[str, dict]:
    """
    Analyse each speaker's audio to extract a voice signature.

    Returns a dict keyed by speaker label, e.g.:
        {
          "SPEAKER1": {
            "pitch_hz": 142.3,
            "pitch_std": 18.1,
            "energy_rms": 0.042,
            "total_speech_sec": 34.2,
            "spectral_centroid": 1234.5,
            "spectral_rolloff": 3456.7,
            "mfcc0_mean": -12.3,
            "mfcc0_std": 5.6,
            ... (mfcc1-12 mean/std)
          }, ...
        }

    Pitch is estimated via autocorrelation on 30 ms frames.
    Gender hint: <165 Hz median = male, >=165 Hz = female.
    """

    sr = sample_rate
    frame_len = int(0.030 * sr)   # 30 ms
    hop_len   = int(0.010 * sr)   # 10 ms
    # Fundamental frequency search range
    f0_min, f0_max = 60, 400      # Hz

    wav_np = waveform.squeeze(0).numpy()  # shape [T]

    profiles: dict[str, dict] = {}

    for spk in set(s["speaker"] for s in merged_segments):
        pitches, energies, total_sec = [], [], 0.0
        # Collect all audio for spectral/MFCC analysis
        all_audio_chunks = []

        for seg in merged_segments:
            if seg["speaker"] != spk:
                continue
            s_idx = int(seg["start"] * sr)
            e_idx = int(seg["end"]   * sr)
            chunk = wav_np[s_idx:e_idx]
            total_sec += seg["end"] - seg["start"]
            
            # Collect chunks for batch spectral analysis
            if len(chunk) > 0:
                all_audio_chunks.append(chunk)

            # Slide over frames
            for start in range(0, len(chunk) - frame_len, hop_len):
                frame = chunk[start: start + frame_len]
                frame = frame - frame.mean()

                # RMS energy
                rms = float(np.sqrt(np.mean(frame ** 2)))
                if rms < 1e-4:          # silence / near-silence — skip
                    continue
                energies.append(rms)

                # Autocorrelation-based pitch
                corr = np.correlate(frame, frame, mode="full")
                corr = corr[len(corr) // 2:]   # keep positive lags only

                # Restrict lag range to F0 bounds
                lag_min = int(sr / f0_max)
                lag_max = int(sr / f0_min)
                lag_max = min(lag_max, len(corr) - 1)

                if lag_min >= lag_max:
                    continue

                sub = corr[lag_min:lag_max]
                if sub.max() <= 0:
                    continue

                peak_lag = int(np.argmax(sub)) + lag_min
                # Voiced confidence: normalised peak height
                confidence = corr[peak_lag] / (corr[0] + 1e-9)
                if confidence < 0.25:   # unvoiced frame
                    continue

                f0 = sr / peak_lag
                pitches.append(f0)

        # Base profile
        if not pitches:
            profiles[spk] = {
                "pitch_hz": 0.0, "pitch_std": 0.0,
                "energy_rms": float(np.mean(energies)) if energies else 0.0,
                "total_speech_sec": total_sec,
            }
            continue

        median_f0  = float(np.median(pitches))
        std_f0     = float(np.std(pitches))
        mean_rms   = float(np.mean(energies)) if energies else 0.0

        profile = {
            "pitch_hz":        round(median_f0, 1),
            "pitch_std":       round(std_f0, 1),
            "energy_rms":      round(mean_rms, 4),
            "total_speech_sec": round(total_sec, 1),
        }

        # Extract spectral features if we have enough audio
        if all_audio_chunks:
            # Concatenate chunks for spectral analysis (limit to first 30s for efficiency)
            combined = np.concatenate(all_audio_chunks)
            max_samples = 30 * sr  # 30 seconds max
            if len(combined) > max_samples:
                combined = combined[:max_samples]
            
            # Spectral features
            spectral = _extract_spectral_features(combined, sr)
            profile.update(spectral)
            
            # MFCC features
            mfcc_stats = _extract_mfcc_stats(combined, sr)
            profile.update(mfcc_stats)

        profiles[spk] = profile

    return profiles


def relabel_by_pitch(
    merged_segments: list[dict],
    profiles: dict[str, dict],
) -> tuple[list[dict], dict[str, dict]]:
    """
    Re-order speaker labels so SPEAKER1 = lowest pitch (most distinctive anchor),
    ascending.  Returns updated segments and profiles dicts.
    """
    # Sort by pitch ascending; unknowns go last
    ordered = sorted(
        profiles.keys(),
        key=lambda s: profiles[s]["pitch_hz"] if profiles[s]["pitch_hz"] > 0 else 9999,
    )
    remap = {old: f"SPEAKER{i+1}" for i, old in enumerate(ordered)}

    new_profiles: dict[str, dict] = {}
    for old, new in remap.items():
        new_profiles[new] = profiles[old]

    for seg in merged_segments:
        seg["speaker"] = remap.get(seg["speaker"], seg["speaker"])

    return merged_segments, new_profiles
