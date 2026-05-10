"""
Test diarization on multiple earnings22 samples.
"""
import requests
import json

API_URL = "http://127.0.0.1:8000"

test_files = [
    ('test_data/earnings22/4443871.mp3', 3),
    ('test_data/earnings22/4443920.mp3', 2),
    ('test_data/earnings22/4450488.mp3', None),
]

print("Testing diarization on earnings22 samples...")
print("=" * 60)

for audio_path, num_speakers in test_files:
    print(f"\nFile: {audio_path}")
    if num_speakers:
        print(f"Forcing {num_speakers} speakers")

    payload = {"wav_path": str(audio_path)}
    if num_speakers:
        payload["num_speakers"] = num_speakers

    try:
        resp = requests.post(
            f"{API_URL}/diarize/path",
            json=payload,
            stream=True,
            timeout=600
        )

        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("type") == "result":
                    profiles = data.get("profiles", {})
                    print(f"Speakers: {len(profiles)}")
                    for spk, prof in profiles.items():
                        print(f"  {spk}: {prof['total_speech_sec']:.1f}s, "
                              f"Pitch: {prof['pitch_hz']:.0f}Hz, "
                              f"Gender: {prof['gender_hint']}")
                    break

    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 60)
print("Test complete!")
