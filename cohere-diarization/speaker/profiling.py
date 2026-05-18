"""Speaker profiling and relabeling utilities."""
import numpy as np


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

        for seg in merged_segments:
            if seg["speaker"] != spk:
                continue
            s_idx = int(seg["start"] * sr)
            e_idx = int(seg["end"]   * sr)
            chunk = wav_np[s_idx:e_idx]
            total_sec += seg["end"] - seg["start"]

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

        profiles[spk] = {
            "pitch_hz":        round(median_f0, 1),
            "pitch_std":       round(std_f0, 1),
            "energy_rms":      round(mean_rms, 4),
            "total_speech_sec": round(total_sec, 1),
        }

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
