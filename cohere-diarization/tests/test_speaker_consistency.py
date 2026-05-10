"""
Test speaker consistency: same speaker in different parts of file should be identified as same.
"""
import requests
import json
from pathlib import Path

API_URL = "http://127.0.0.1:8000"

def get_speakers(audio_path, start_time=None, end_time=None, num_speakers=None):
    """Get speaker info for a file or segment."""
    payload = {"wav_path": str(audio_path)}

    if start_time is not None and end_time is not None:
        payload["start_time"] = start_time
        payload["end_time"] = end_time
        print(f"\nSegment: {start_time}-{end_time}s")
    else:
        print(f"\nFull file: {Path(audio_path).name}")

    if num_speakers:
        payload["num_speakers"] = num_speakers
        print(f"Forcing {num_speakers} speakers")

    try:
        resp = requests.post(
            f"{API_URL}/diarize/path",
            json=payload,
            stream=True,
            timeout=600
        )

        result = None
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("type") == "result":
                    result = data
                    break

        if result:
            profiles = result.get("profiles", {})
            print(f"Speakers: {len(profiles)}")
            for spk, prof in profiles.items():
                print(f"  {spk}: {prof['total_speech_sec']:.1f}s, "
                      f"Pitch: {prof['pitch_hz']:.0f}Hz, "
                      f"Gender: {prof['gender_hint']}")
            return profiles
        else:
            print("No result")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    audio_path = "test_data/earnings22/4443871.mp3"

    print("="*60)
    print("TESTING SPEAKER CONSISTENCY")
    print("="*60)

    # Get full file speakers (force 3)
    full_profiles = get_speakers(audio_path, num_speakers=3)

    if full_profiles:
        # Test segments
        segments = [
            (0, 300, "Early"),
            (600, 900, "Middle"),
            (1800, 2100, "Late")
        ]

        for start, end, label in segments:
            seg_profiles = get_speakers(audio_path, start, end, num_speakers=3)

            if seg_profiles and full_profiles:
                print(f"\n{label} segment vs Full file comparison:")
                for spk_full, prof_full in full_profiles.items():
                    pitch_full = prof_full['pitch_hz']
                    for spk_seg, prof_seg in seg_profiles.items():
                        pitch_seg = prof_seg['pitch_hz']
                        diff = abs(pitch_full - pitch_seg)
                        if diff < 30:  # Within 30Hz
                            print(f"  {spk_full} (full) ~ {spk_seg} ({label}): "
                                  f"pitch diff = {diff:.0f}Hz")

    print("\n" + "="*60)
    print("SPEAKER CONSISTENCY TEST COMPLETE!")
