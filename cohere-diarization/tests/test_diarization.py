"""
Quick test to verify diarization works on multi-speaker audio.
Uses existing test files from the mini-test-data dataset.
"""
import subprocess
import time
import requests
import json
from pathlib import Path
import sys

def check_server_running(url="http://127.0.0.1:8000"):
    """Check if the diarization server is running."""
    try:
        resp = requests.get(f"{url}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False

def start_server():
    """Start the diarization server if not running."""
    print("Starting diarization server...")
    # Start server in background
    subprocess.Popen(
        [".venv/Scripts/python.exe", "server.py"],
        cwd="C:/Users/gerge/source/repos/ASR_windows/cohere-diarization",
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    # Wait for server to be ready
    for i in range(60):
        if check_server_running():
            print("Server is ready!")
            return True
        print(f"Waiting for server... ({i+1}/60)")
        time.sleep(5)
    print("Server failed to start in time")
    return False

def test_diarization(audio_path):
    """Test diarization on a given audio file."""
    print(f"\nTesting diarization on: {Path(audio_path).name}")
    print("=" * 60)

    # Call the diarize endpoint
    try:
        resp = requests.post(
            "http://127.0.0.1:8000/diarize/path",
            json={"wav_path": str(audio_path)},
            stream=True,
            timeout=300
        )

        segments = []
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if "speaker" in str(data):
                    segments.append(data)
                    print(f"  {data}")

        print(f"\nFound {len(segments)} speaker segments")
        return segments

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Check if server is running
    if not check_server_running():
        print("Server not running. Please start it manually:")
        print("  cd cohere-diarization")
        print("  .venv\\Scripts\\activate")
        print("  python server.py")
        sys.exit(1)

    # Test files - use existing test data with potential multiple speakers
    test_files = [
        "tests/mini-test-data/dataset/TOEFL listening practice test 2020.mp3",
        "tests/mini-test-data/dataset/test_1.wav",
    ]

    for test_file in test_files:
        path = Path(test_file)
        if path.exists():
            test_diarization(path)
        else:
            print(f"File not found: {test_file}")

    print("\n" + "=" * 60)
    print("Diarization test complete!")
