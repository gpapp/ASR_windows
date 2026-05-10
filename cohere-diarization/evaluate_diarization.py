"""
Evaluate diarization accuracy against earnings22 ground truth.

Usage:
1. Ensure earnings22 NLP reference files are downloaded (git lfs pull)
2. Run: python evaluate_diarization.py
"""
import json
import requests
from pathlib import Path
import sys

API_URL = "http://127.0.0.1:8000"

def load_ground_truth(nlp_path):
    """Parse earnings22 NLP file to get speaker timestamps."""
    segments = []
    try:
        with open(nlp_path, 'r') as f:
            lines = f.readlines()

        # Skip header line
        for line in lines[1:]:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                token = parts[0]
                speaker = int(parts[1])
                start_ts = float(parts[2]) if parts[2] else None
                end_ts = float(parts[3]) if parts[3] else None

                if start_ts is not None:
                    segments.append({
                        'speaker': speaker,
                        'start': start_ts,
                        'end': end_ts,
                        'token': token
                    })

    except Exception as e:
        print(f"Error reading {nlp_path}: {e}")
        return None

    return segments

def get_diarization(audio_path, num_speakers=None):
    """Get diarization results from server."""
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
                    return data

    except Exception as e:
        print(f"Error during diarization: {e}")
        return None

def calculate_diarization_error(ground_truth, diarization_result):
    """Calculate diarization error rate (simplified)."""
    gt_segments = []
    for seg in ground_truth:
        if seg['start'] is not None and seg['end'] is not None:
            gt_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker']
            })

    pred_segments = diarization_result.get('segments', [])

    # Simple metric: compare speaker overlap
    # This is a simplified version - proper DER requires more complex alignment
    total_duration = max(
        max([s['end'] for s in gt_segments]) if gt_segments else 0,
        max([s['end'] for s in pred_segments]) if pred_segments else 0
    )

    if not gt_segments or not pred_segments:
        return None

    # Count number of unique speakers
    gt_speakers = len(set(s['speaker'] for s in gt_segments))
    pred_speakers = len(set(s['speaker'] for s in pred_segments))

    print(f"\nGround truth: {gt_speakers} speakers")
    print(f"Diarization: {pred_speakers} speakers")

    # Show speaker mapping
    print("\nGround truth speaker segments (first 10):")
    for seg in gt_segments[:10]:
        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] Speaker {seg['speaker']}")

    print("\nDiarization segments (first 10):")
    for seg in pred_segments[:10]:
        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}")

    return {
        'gt_speakers': gt_speakers,
        'pred_speakers': pred_speakers,
        'match': gt_speakers == pred_speakers
    }

def main():
    # Find earnings22 files
    base_dir = Path("test_data/earnings22/temp_earnings/earnings22/subset10")
    nlp_dir = base_dir / "verbatim_transcripts" / "nlp_references"
    audio_dir = base_dir

    if not nlp_dir.exists():
        print(f"NLP reference directory not found: {nlp_dir}")
        print("Please run: git lfs pull --include='*.nlp'")
        return

    # Process each NLP file
    nlp_files = list(nlp_dir.glob("*.nlp"))

    if not nlp_files:
        print("No NLP files found. Make sure Git LFS files are pulled.")
        return

    print(f"Found {len(nlp_files)} files to evaluate")

    for nlp_file in nlp_files[:3]:  # Test first 3 files
        audio_file = audio_dir / f"{nlp_file.stem}.mp3"

        if not audio_file.exists():
            print(f"Audio file not found: {audio_file}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {nlp_file.stem}")
        print(f"{'='*60}")

        # Load ground truth
        gt_segments = load_ground_truth(nlp_file)
        if not gt_segments:
            continue

        # Get diarization
        num_speakers = len(set(s['speaker'] for s in gt_segments))
        result = get_diarization(audio_file, num_speakers)

        if result:
            calculate_diarization_error(gt_segments, result)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Evaluate single file
        nlp_path = Path(sys.argv[1])
        audio_path = nlp_path.parent.parent.parent / f"{nlp_path.stem}.mp3"

        gt = load_ground_truth(nlp_path)
        if gt:
            result = get_diarization(audio_path)
            if result:
                calculate_diarization_error(gt, result)
    else:
        main()
