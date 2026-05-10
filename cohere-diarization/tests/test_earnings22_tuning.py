"""
Test diarization on earnings22 samples and tune parameters.

Uses the earnings22 audio files already downloaded in test_data/earnings22/
Focuses on tuning distance_threshold and num_speakers for optimal results.
"""
import json
import requests
from pathlib import Path
import time

API_URL = "http://127.0.0.1:8000"

def test_diarization(audio_path, num_speakers=None, threshold=None):
    """Test diarization with given parameters."""
    payload = {"wav_path": str(audio_path)}
    if num_speakers:
        payload["num_speakers"] = num_speakers
    if threshold:
        payload["distance_threshold"] = threshold

    print(f"\n{'='*60}")
    print(f"File: {Path(audio_path).name}")
    params_str = f"speakers={num_speakers}, threshold={threshold}"
    print(f"Params: {params_str}")
    print(f"{'='*60}")

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
            segments = result.get("segments", [])
            profiles = result.get("profiles", {})

            num_pred_speakers = len(profiles)
            print(f"Speakers detected: {num_pred_speakers}")
            print(f"Total segments: {len(segments)}")

            # Show speaker distribution
            print("\nSpeaker profiles:")
            for speaker, profile in profiles.items():
                duration = profile.get('total_speech_sec', 0)
                pitch = profile.get('pitch_hz', 0)
                gender = profile.get('gender_hint', 'unknown')
                print(f"  {speaker}: {duration:.1f}s (pitch: {pitch:.0f}Hz, gender: {gender})")

            return {
                'num_speakers': num_pred_speakers,
                'segments': segments,
                'profiles': profiles
            }
        else:
            print("No result returned")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """Run diarization tests on earnings22 samples."""
    # Get audio files
    audio_dir = Path("test_data/earnings22")
    audio_files = [f for f in audio_dir.glob("*.mp3") if f.stat().st_size > 10000]

    if not audio_files:
        print("No valid audio files found!")
        return

    print(f"Found {len(audio_files)} audio files")
    print(f"Testing on first 3 files")

    # Test configurations to try
    configs = [
        {"num_speakers": None, "threshold": None, "label": "Default"},
        {"num_speakers": 2, "threshold": None, "label": "2 speakers"},
        {"num_speakers": 3, "threshold": None, "label": "3 speakers"},
        {"num_speakers": None, "threshold": 0.25, "label": "Threshold 0.25"},
        {"num_speakers": None, "threshold": 0.35, "label": "Threshold 0.35"},
        {"num_speakers": 2, "threshold": 0.25, "label": "2 spk + thresh 0.25"},
    ]

    results = []

    for audio_file in audio_files[:3]:
        print(f"\n{'#'*60}")
        print(f"TESTING: {audio_file.name}")
        print(f"{'#'*60}")

        file_results = []

        for config in configs:
            label = config.pop("label")
            print(f"\n>>> Test: {label}")

            result = test_diarization(
                audio_file,
                num_speakers=config.get("num_speakers"),
                threshold=config.get("threshold")
            )

            if result:
                file_results.append({
                    "label": label,
                    "num_speakers": result['num_speakers']
                })

        results.append({
            "file": audio_file.name,
            "tests": file_results
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for file_result in results:
        print(f"\n{file_result['file']}:")
        for test in file_result['tests']:
            print(f"  {test['label']}: {test['num_speakers']} speakers")

if __name__ == "__main__":
    # Check server
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code != 200:
            print("ERROR: Server not running!")
            print("Start with: python server.py")
            exit(1)
    except:
        print("ERROR: Server not running!")
        print("Start with: python server.py")
        exit(1)

    main()
