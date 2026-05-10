"""
Test automatic speaker detection without forcing count.
Focus on tuning threshold and voiceprint merging.
"""
import requests
import json

API_URL = "http://127.0.0.1:8000"

def test_automatic(audio_path, threshold=None):
    """Test with automatic detection."""
    print(f"\n{'='*60}")
    print(f"File: {audio_path}")
    if threshold:
        print(f"Threshold: {threshold}")
    print(f"{'='*60}")

    payload = {"wav_path": audio_path}
    if threshold:
        payload["distance_threshold"] = threshold

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
            profiles = result.get("profiles", {})
            segments = result.get("segments", [])

            print(f"\nSpeakers detected: {len(profiles)}")
            for spk, prof in profiles.items():
                print(f"  {spk}: {prof['total_speech_sec']:.1f}s, "
                      f"Pitch: {prof['pitch_hz']:.0f}Hz, "
                      f"Gender: {prof['gender_hint']}")

            print(f"\nSegments: {len(segments)}")
            return len(profiles)
        else:
            print("No result")
            return 0

    except Exception as e:
        print(f"Error: {e}")
        return 0

if __name__ == "__main__":
    # Test files
    test_files = [
        "test_data/earnings22/4443871.mp3",
        "test_data/earnings22/4443920.mp3",
        "test_data/earnings22/4450488.mp3",
    ]

    for audio_path in test_files:
        # Test with different thresholds
        for threshold in [None, 0.25, 0.30, 0.35, 0.40]:
            count = test_automatic(audio_path, threshold)
            if count >= 2:
                print(f"✓ {count} speakers detected!")
                break
