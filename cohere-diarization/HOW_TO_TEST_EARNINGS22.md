# How to Test Diarization with Earnings22 Dataset

## Prerequisites
- Server running: `cd cohere-diarization && python server.py`
- Git LFS installed (for downloading audio files)

## Step 1: Download Earnings22 Samples

### Option A: Using Git LFS (Recommended)
```bash
# Clone the earnings22 dataset
cd C:\Users\gerge\source\repos\ASR_windows\cohere-diarization\test_data
git clone --depth 1 https://github.com/revdotcom/speech-datasets.git temp_earnings
cd temp_earnings
git lfs install
git lfs pull --include="earnings22/subset10/*.mp3"

# Copy files to test_data/earnings22
xcopy /E /Y earnings22\subset10 ..\earnings22\
```

### Option B: Direct Download
Download these sample files manually:
- https://github.com/revdotcom/speech-datasets/raw/main/earnings22/subset10/4453225.mp3
- https://github.com/revdotcom/speech-datasets/raw/main/earnings22/subset10/4469088.mp3
- https://github.com/revdotcom/speech-datasets/raw/main/earnings22/subset10/4470684.mp3

Save them to: `cohere-diarization\test_data\earnings22\`

## Step 2: Run Diarization Test

```bash
cd C:\Users\gerge\source\repos\ASR_windows\cohere-diarization

# Make sure server is running in another terminal
# Then run:
.venv\Scripts\python.exe test_earnings22.py
```

## Step 3: Verify Results

The script will output:
- Speaker segments with timestamps
- Speaker profiles (pitch, gender hint, energy, total speech duration)
- Number of speakers detected

## Quick Test with Existing Files

To verify diarization works right now (without earnings22):

```bash
cd C:\Users\gerge\source\repos\ASR_windows\cohere-diarization

# Test with existing test file
.venv\Scripts\python.exe -c "
import requests, json
resp = requests.post('http://127.0.0.1:8000/diarize/path',
    json={'wav_path': 'C:/Users/gerge/source/repos/ASR_windows/cohere-diarization/tests/mini-test-data/dataset/test/test_1.wav'},
    stream=True, timeout=300)
for line in resp.iter_lines():
    if line:
        data = json.loads(line)
        if 'result' in str(data):
            print(json.dumps(data, indent=2))
"
```

## Expected Output

Successful diarization should show:
- 2-4 speakers for earnings22 samples (earnings calls typically have multiple speakers)
- Speaker labels: SPEAKER1, SPEAKER2, etc.
- Profile data with pitch, gender hint, and speech duration
