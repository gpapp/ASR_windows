"""
Test diarization using earnings22 dataset samples.
Instructions:
1. Download earnings22 samples from: https://github.com/revdotcom/speech-datasets/tree/main/earnings22
   - Use Git LFS: git lfs pull
   - Or download subset10 manually
2. Place audio files in test_data/earnings22/
3. Run this script
"""
import requests
import json
from pathlib import Path
import time

API_URL = "http://127.0.0.1:8000"

def check_server():
    """Check if server is running."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False

def diarize_audio(audio_path, num_speakers=None):
    """Run diarization on audio file."""
    print(f"\n{'='*60}")
    print(f"Processing: {Path(audio_path).name}")
    print(f"{'='*60}")

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

        segments = []
        profiles = {}

        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("type") == "result":
                    segments = data.get("segments", [])
                    profiles = data.get("profiles", {})
                elif "error" in data:
                    print(f"Error: {data['error']}")
                    return None

        return {"segments": segments, "profiles": profiles}

    except Exception as e:
        print(f"Error: {e}")
        return None

def print_results(result):
    """Print diarization results."""
    if not result:
        return

    segments = result["segments"]
    profiles = result["profiles"]

    print(f"\nFound {len(segments)} speaker segments:")
    for seg in segments:
        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}")

    print(f"\nSpeaker profiles:")
    for speaker, profile in profiles.items():
        print(f"  {speaker}:")
        print(f"    Pitch: {profile['pitch_hz']:.1f} Hz (std: {profile['pitch_std']:.1f})")
        print(f"    Gender hint: {profile['gender_hint']}")
        print(f"    Total speech: {profile['total_speech_sec']:.1f}s")
        print(f"    Energy RMS: {profile['energy_rms']:.4f}")

if __name__ == "__main__":
    # Check server
    if not check_server():
        print("ERROR: Server not running!")
        print("Start it with: cd cohere-diarization && python server.py")
        exit(1)

    # Find earnings22 audio files
    earnings_dir = Path("test_data/earnings22")
    audio_files = list(earnings_dir.glob("**/*.mp3")) + \
                  list(earnings_dir.glob("**/*.wav"))

    if not audio_files:
        print("No audio files found in test_data/earnings22/")
        print("\nTo download earnings22 samples:")
        print("1. Clone with Git LFS:")
        print("   git clone https://github.com/revdotcom/speech-datasets.git")
        print("   cd speech-datasets/earnings22")
        print("   git lfs pull")
        print("2. Or download subset10 from:")
        print("   https://github.com/revdotcom/speech-datasets/tree/main/earnings22/subset10")
        exit(1)

    print(f"Found {len(audio_files)} audio file(s)")

    # Process each file
    for audio_file in audio_files[:3]:  # Test first 3 files
        result = diarize_audio(audio_file)
        if result:
            print_results(result)

    print("\n" + "="*60)
    print("Earnings22 diarization test complete!")
