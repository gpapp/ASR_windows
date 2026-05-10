"""
Simple test - automatic speaker detection.
"""
import requests
import json

API_URL = "http://127.0.0.1:8000"

def test_file(audio_path):
    """Test automatic detection."""
    print(f"\n{'='*60}")
    print(f"File: {audio_path}")
    print(f"{'='*60}")

    try:
        resp = requests.post(
            f"{API_URL}/diarize/path",
            json={"wav_path": audio_path},
            stream=True,
            timeout=600
        )

        result = None
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if "DEBUG" in str(data):
                    print(f"  {data}")
                if data.get("type") == "result":
                    result = data
                    break

        if result:
            profiles = result.get("profiles", {})
            print(f"\nSpeakers detected: {len(profiles)}")
            for spk, prof in profiles.items():
                print(f"  {spk}: {prof['total_speech_sec']:.1f}s, "
                      f"Pitch: {prof['pitch_hz']:.0f}Hz, "
                      f"Gender: {prof['gender_hint']}")
            return len(profiles)
        else:
            print("No result")
            return 0

    except Exception as e:
        print(f"Error: {e}")
        return 0

if __name__ == "__main__":
    # Test automatic detection
    count = test_file("test_data/earnings22/4443871.mp3")
    if count >= 2:
        print(f"\n✓ Automatic detection found {count} speakers!")
    else:
        print(f"\n✗ Only {count} speaker(s) detected - need tuning")
