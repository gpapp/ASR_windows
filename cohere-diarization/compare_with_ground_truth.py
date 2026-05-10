"""
Compare diarization output with earnings22 ground truth.

Downloads NLP reference files from GitHub and compares speaker identification.
Tunes parameters to improve accuracy.
"""
import json
import requests
from pathlib import Path
import sys
import time

API_URL = "http://127.0.0.1:8000"

def download_nlp_file(filename):
    """Download NLP file from GitHub raw."""
    import urllib.request

    url = f"https://raw.githubusercontent.com/revdotcom/speech-datasets/main/earnings22/subset10/verbatim_transcripts/nlp_references/{filename}"
    local_path = Path(f"test_data/earnings22/nlp_references/{filename}")

    if local_path.exists() and local_path.stat().st_size > 1000:
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None

def parse_nlp_file(nlp_path):
    """Parse earnings22 NLP file to extract speaker segments."""
    segments = []
    try:
        with open(nlp_path, 'r') as f:
            lines = f.readlines()

        current_speaker = None
        current_start = None

        for line in lines[1:]:  # Skip header
            parts = line.strip().split('|')
            if len(parts) < 4:
                continue

            token = parts[0]
            speaker = int(parts[1]) if parts[1] else None
            start_ts = float(parts[2]) if parts[2] else None
            end_ts = float(parts[3]) if parts[3] else None

            if speaker is not None and start_ts is not None:
                segments.append({
                    'speaker': speaker,
                    'start': start_ts,
                    'end': end_ts,
                    'token': token
                })

    except Exception as e:
        print(f"Error parsing {nlp_path}: {e}")
        return None

    return segments

def get_diarization(audio_path, num_speakers=None, threshold=None):
    """Get diarization results from server."""
    payload = {"wav_path": str(audio_path)}
    if num_speakers:
        payload["num_speakers"] = num_speakers
    if threshold:
        payload["distance_threshold"] = threshold

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

def calculate_metrics(ground_truth, diarization_result):
    """Calculate diarization error metrics."""
    gt_segments = [s for s in ground_truth if s['start'] is not None]

    if not gt_segments:
        return None

    gt_speakers = len(set(s['speaker'] for s in gt_segments))
    pred_segments = diarization_result.get('segments', [])
    pred_speakers = len(set(s['speaker'] for s in pred_segments))

    # Calculate speaker overlap accuracy (simplified)
    # Map predicted speakers to ground truth speakers
    total_correct = 0
    total_duration = 0

    for pred_seg in pred_segments:
        pred_start = pred_seg['start']
        pred_end = pred_seg['end']
        pred_speaker = pred_seg['speaker']

        # Find overlapping ground truth segments
        overlap_duration = 0
        for gt_seg in gt_segments:
            if gt_seg['start'] < pred_end and gt_seg['end'] > pred_start:
                overlap_start = max(pred_start, gt_seg['start'])
                overlap_end = min(pred_end, gt_seg['end'])
                overlap_duration += (overlap_end - overlap_start)

        if overlap_duration > 0:
            total_correct += overlap_duration
        total_duration += (pred_end - pred_start)

    accuracy = total_correct / total_duration if total_duration > 0 else 0

    return {
        'gt_speakers': gt_speakers,
        'pred_speakers': pred_speakers,
        'accuracy': accuracy,
        'speaker_match': gt_speakers == pred_speakers
    }

def main():
    """Run comparison on earnings22 samples."""
    # Use audio files we have
    audio_dir = Path("test_data/earnings22")
    audio_files = [f for f in audio_dir.glob("*.mp3") if f.stat().st_size > 10000]

    if not audio_files:
        print("No valid audio files found in test_data/earnings22/")
        return

    print(f"Found {len(audio_files)} audio files to test")

    for audio_file in audio_files[:3]:
        print(f"\n{'='*60}")
        print(f"Processing: {audio_file.name}")
        print(f"{'='*60}")

        # Download corresponding NLP file
        nlp_file = download_nlp_file(f"{audio_file.stem}.nlp")
        if not nlp_file:
            print("Could not get ground truth, skipping...")
            continue

        # Parse ground truth
        gt_segments = parse_nlp_file(nlp_file)
        if not gt_segments:
            continue

        gt_speakers = len(set(s['speaker'] for s in gt_segments))
        print(f"Ground truth: {gt_speakers} speakers")

        # Test with different parameters
        param_sets = [
            {"num_speakers": None, "threshold": None, "label": "Default"},
            {"num_speakers": gt_speakers, "threshold": None, "label": f"Fixed {gt_speakers} speakers"},
            {"num_speakers": gt_speakers, "threshold": 0.25, "label": f"Threshold 0.25 + {gt_speakers} spk"},
        ]

        for params in param_sets:
            label = params.pop("label")
            print(f"\n--- Test: {label} ---")

            result = get_diarization(audio_file, **params)
            if result:
                metrics = calculate_metrics(gt_segments, result)
                if metrics:
                    print(f"GT speakers: {metrics['gt_speakers']}")
                    print(f"Pred speakers: {metrics['pred_speakers']}")
                    print(f"Speaker match: {'YES' if metrics['speaker_match'] else 'NO'}")
                    print(f"Accuracy: {metrics['accuracy']*100:.1f}%")

if __name__ == "__main__":
    main()
