"""
Test different diarization parameters on earnings22 samples.
Focus on understanding why only 1 speaker is detected.
"""
import json
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000"

def test_diarization(audio_path, params):
    """Test diarization with given parameters."""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(audio_path).name}")
    print(f"Params: {params}")
    print(f"{'='*60}")

    payload = {"wav_path": str(audio_path)}
    payload.update(params)

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
                elif "DEBUG" in str(data):
                    print(f"  {data}")

        if result:
            segments = result.get("segments", [])
            profiles = result.get("profiles", {})

            print(f"\nSpeakers detected: {len(profiles)}")
            for speaker, profile in profiles.items():
                print(f"  {speaker}: {profile['total_speech_sec']:.1f}s "
                      f"(pitch: {profile['pitch_hz']:.0f}Hz, "
                      f"gender: {profile['gender_hint']})")

            print(f"\nSegments: {len(segments)}")
            for seg in segments[:5]:
                print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}")

            return result
        else:
            print("No result returned")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    audio_file = "test_data/earnings22/4443871.mp3"

    # Test different parameter combinations
    tests = [
        {"label": "Default", "params": {}},
        {"label": "Threshold 0.10 (low)", "params": {"distance_threshold": 0.10}},
        {"label": "Threshold 0.40 (high)", "params": {"distance_threshold": 0.40}},
        {"label": "Forced 2 speakers", "params": {"num_speakers": 2}},
        {"label": "Forced 3 speakers", "params": {"num_speakers": 3}},
        {"label": "Threshold 0.10 + 2 speakers", "params": {"distance_threshold": 0.10, "num_speakers": 2}},
    ]

    for test in tests:
        print(f"\n{'#'*60}")
        print(f"TEST: {test['label']}")
        result = test_diarization(audio_file, test['params'])

        if result and len(result.get('profiles', {})) > 1:
            print(f"\n✓ Multiple speakers detected!")
