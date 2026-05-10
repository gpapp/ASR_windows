"""
Download a few samples from earnings22 dataset for diarization testing.
Uses huggingface_hub to download files directly.
"""
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path
import soundfile as sf
import json

output_dir = Path("test_data/earnings22")
output_dir.mkdir(parents=True, exist_ok=True)

print("Listing files in earnings22 dataset...")
# List audio files in the dataset
files = list_repo_files("revdotcom/earnings22", repo_type="dataset")

# Filter for audio files
audio_files = [f for f in files if f.endswith(('.wav', '.mp3', '.flac'))][:3]

print(f"Found {len(audio_files)} audio files to download")

for audio_file in audio_files:
    print(f"Downloading {audio_file}...")
    try:
        local_path = hf_hub_download(
            repo_id="revdotcom/earnings22",
            filename=audio_file,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"Saved to: {local_path}")
    except Exception as e:
        print(f"Error downloading {audio_file}: {e}")

# Also download some transcript files if available
transcript_files = [f for f in files if f.endswith('.csv') or 'transcript' in f.lower()][:2]
for txt_file in transcript_files:
    print(f"Downloading transcript {txt_file}...")
    try:
        hf_hub_download(
            repo_id="revdotcom/earnings22",
            filename=txt_file,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        print(f"Error downloading transcript: {e}")

print("\nDone! Files saved to test_data/earnings22/")
