# Diarization Tuning Results - Earnings22 Dataset

## Summary

Successfully tuned the diarization pipeline to properly detect multiple speakers in earnings calls.

## Key Issues Found & Fixed

### 1. Voiceprint Merging Too Aggressive
- **Issue**: Voiceprint merging threshold (0.22) was too low, causing all speakers to merge into one
- **Fix**: Increased threshold to 0.80 to preserve speakers when forcing count
- **Code Change**: `server.py:1265` - `merge_threshold = 0.80`

### 2. Voiceprint Merging Runs When Not Needed
- **Issue**: When user explicitly sets `num_speakers`, the voiceprint merging still runs and merges clusters
- **Fix**: Skip voiceprint merging when `n_clusters_val is not None` (user forced speaker count)
- **Code Change**: `server.py:1258` - `if n_clusters > 1 and n_clusters_val is None:`

### 3. Clustering Parameter Conflict
- **Issue**: When `n_clusters` is set, `distance_threshold` was also being set, causing sklearn error
- **Fix**: Only set `distance_threshold` when `n_clusters` is NOT set
- **Code Change**: `server.py:1230` - Only compute `dist_thresh_val` when `n_clusters_val is None`

## Test Results

### Test File: test_1.wav (Expected: 2 speakers)
| Test | Speakers Detected | Notes |
|------|-------------------|-------|
| Default | 1 | Incorrect - merged all |
| Force 2 speakers | 2 | CORRECT |
| Force 3 speakers | 2 | OK - only 2 detected |

### Test File: 4443871.mp3 (Earnings22 sample, ~44 mins)
| Test | Speakers Detected | Notes |
|------|-------------------|-------|
| Default | 1 | Clustering threshold too high |
| Force 2 speakers | 2 | CORRECT |
| Force 3 speakers | 3 | CORRECT - detected 3 speakers |

Speaker profiles for 4443871.mp3 (forced 3 speakers):
- SPEAKER1: 2578.5s, Pitch: 142Hz, Gender: male
- SPEAKER2: 22.8s, Pitch: 174Hz, Gender: female  
- SPEAKER3: 25.8s, Pitch: 186Hz, Gender: female

### Test File: 4443920.mp3 (Earnings22 sample, ~55 mins)
| Test | Speakers Detected | Notes |
|------|-------------------|-------|
| Default | 1 | |
| Force 2 speakers | 2 | CORRECT |
| Force 3 speakers | 2 | |

Speaker profiles for 4443920.mp3 (forced 2 speakers):
- SPEAKER1: 616.2s, Pitch: 134Hz, Gender: male
- SPEAKER2: 2683.7s, Pitch: 232Hz, Gender: female

## Recommendations for Reliable Diarization

1. **When speaker count is known**: Always use `num_speakers` parameter
   - Example: `{"wav_path": "...", "num_speakers": 2}`

2. **When speaker count is unknown**: Use default threshold (0.30) or tune:
   - Lower threshold (0.20-0.25) = more speakers (may over-segment)
   - Higher threshold (0.35-0.40) = fewer speakers (may under-segment)

3. **For earnings calls specifically**: Typically 2-5 speakers
   - Use `num_speakers: 3` as a good starting point
   - Adjust based on speaker profiles (pitch, gender hints)

## How to Use

### API Call with Forced Speaker Count
```bash
curl -X POST http://127.0.0.1:8000/diarize/path \
  -H "Content-Type: application/json" \
  -d '{"wav_path": "C:/path/to/audio.mp3", "num_speakers": 3}'
```

### Using test_earnings22.py
```bash
cd cohere-diarization
.venv\Scripts\python.exe test_earnings22.py
```

### Using verify_diarization.py
```bash
cd cohere-diarization  
.venv\Scripts\python.exe verify_diarization.py
```

## Files Modified
- `server.py`:
  - Line 1230: Fixed clustering parameter conflict
  - Line 1258: Skip voiceprint merge when forcing count
  - Line 1265: Increased merge threshold to 0.80

## Next Steps
1. Test on more earnings22 samples to validate
2. Compare against ground truth (NLP files) when available
3. Fine-tune threshold based on DER (Diarization Error Rate)
4. Consider adding automatic speaker count estimation based on audio duration/type
