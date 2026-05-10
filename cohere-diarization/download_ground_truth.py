"""
Download and compare earnings22 ground truth with diarization output.
"""
import json
import urllib.request
import os
from pathlib import Path

def download_file(url, output_path):
    """Download file from GitHub API."""
    req = urllib.request.Request(url)
    req.add_header('Accept', 'application/vnd.github.v3.raw')
    
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
            with open(output_path, 'w') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def parse_ground_truth(json_path):
    """Parse normalization JSON to get speaker segments with timestamps."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # The JSON structure is a list of segments
    if isinstance(data, list):
        segments = []
        for seg in data:
            if 'start' in seg and 'end' in seg and 'speaker' in seg:
                segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'speaker': seg['speaker']
                })
        return segments
    return []

def get_audio_files():
    """Get list of earnings22 audio files with valid sizes."""
    audio_dir = Path("test_data/earnings22")
    audio_files = []
    
    for mp3_file in audio_dir.glob("*.mp3"):
        if mp3_file.stat().st_size > 10000:  # Valid audio file
            audio_files.append(mp3_file)
    
    return audio_files

if __name__ == "__main__":
    # Download ground truth files
    base_url = "https://api.github.com/repos/revdotcom/speech-datasets/contents/earnings22/subset10/verbatim_transcripts/normalizations"
    
    audio_files = get_audio_files()
    print(f"Found {len(audio_files)} valid audio files")
    
    for audio_file in audio_files[:3]:
        json_filename = f"{audio_file.stem}.norm.json"
        output_path = f"test_data/earnings22/{json_filename}"
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            print(f"\nDownloading ground truth for {audio_file.name}...")
            url = f"{base_url}/{json_filename}"
            if download_file(url, output_path):
                print(f"Saved to {output_path}")
        else:
            print(f"\nGround truth already exists: {output_path}")
        
        # Parse and show summary
        segments = parse_ground_truth(output_path)
        if segments:
            speakers = set(s['speaker'] for s in segments)
            print(f"Speakers: {len(speakers)}")
            print(f"Segments: {len(segments)}")
            print(f"Duration: {max(s['end'] for s in segments):.1f}s")
            
            # Show first few segments
            print("First 3 segments:")
            for seg in segments[:3]:
                print(f"  Speaker {seg['speaker']}: [{seg['start']:.1f}s - {seg['end']:.1f}s]")
