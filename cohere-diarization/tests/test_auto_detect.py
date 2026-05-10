import requests
import json

API_URL = "http://127.0.0.1:8000"

def test_auto(audio_path, label):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")

    payload = {"wav_path": str(audio_path)}
    resp = requests.post(f"{API_URL}/diarize/path", json=payload, stream=True, timeout=600)

    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            if data.get("type") == "result":
                profiles = data.get("profiles", {})
                print(f"Auto-detected speakers: {len(profiles)}")
                for spk, prof in profiles.items():
                    print(f"  {spk}: {prof['total_speech_sec']:.1f}s, "
                          f"Pitch: {prof['pitch_hz']:.0f}Hz, "
                          f"Gender: {prof['gender_hint']}")
                return profiles
    return None

if __name__ == "__main__":
    test_auto("test_data/earnings22/4466399.mp3", "4466399.mp3 (16MB, ~16min)")
    test_auto("test_data/earnings22/4466607.mp3", "4466607.mp3 (20MB, ~20min)")
