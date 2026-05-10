"""
Get debug info from server - cosine distances.
"""
import requests
import json

API_URL = "http://127.0.0.1:8000"

audio_path = "test_data/earnings22/4443871.mp3"

print("Getting debug info...")
resp = requests.post(
    f"{API_URL}/diarize/path",
    json={"wav_path": audio_path},
    stream=True,
    timeout=600
)

for line in resp.iter_lines():
    if line:
        data = json.loads(line)
        if "DEBUG" in str(data):
            print(data)
        if data.get("type") == "result":
            profiles = data.get("profiles", {})
            print(f"\nSpeakers: {len(profiles)}")
            for spk, p in profiles.items():
                print(f"  {spk}: {p['total_speech_sec']:.1f}s")
            break
