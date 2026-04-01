import sys
import os
import subprocess
import librosa
import numpy as np
import re
import requests
from tqdm import tqdm

SERVER_URL = "http://127.0.0.1:8000/transcribe"
PUNCTUATE_URL = "http://127.0.0.1:8000/punctuate"

def get_audio_duration(file_path):
    cmd = ['ffmpeg', '-i', file_path]
    res = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", res.stderr)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return 0

def get_speech_segments(file_path, max_dur=30):
    """Detects speech regions by identifying silence and returning (start, end) pairs."""
    # noise=-40dB: Anything quieter than this is silence.
    # d=0.5: Silence must last at least 0.5 seconds to be counted.
    command = [
        'ffmpeg', '-i', file_path,
        '-af', 'silencedetect=noise=-40dB:d=0.5',
        '-f', 'null', '-'
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    output = result.stderr
    
    starts = [float(t) for t in re.findall(r"silence_start: (\d+\.\d+)", output)]
    ends = [float(t) for t in re.findall(r"silence_end: (\d+\.\d+)", output)]
    duration = get_audio_duration(file_path)
    
    speech_segments = []
    current_pos = 0.0
    
    for i in range(len(starts)):
        sil_start = starts[i]
        # Ignore speech blips shorter than 500ms
        if sil_start > current_pos + 0.5:
            speech_segments.append((current_pos, sil_start))
        current_pos = ends[i] if i < len(ends) else duration
        
    if current_pos < duration - 0.5:
        speech_segments.append((current_pos, duration))
        
    # Merge segments that are close (e.g. < 1.0s apart) to keep chunks coherent
    merged = []
    if speech_segments:
        curr_s, curr_e = speech_segments[0]
        for i in range(1, len(speech_segments)):
            next_s, next_e = speech_segments[i]
            if next_s - curr_e < 1.0: 
                curr_e = next_e
            else:
                merged.append((curr_s, curr_e))
                curr_s, curr_e = next_s, next_e
        merged.append((curr_s, curr_e))
    
    # Split long segments (> max_dur) into smaller pieces
    final_segments = []
    for s_start, s_end in merged:
        segment_len = s_end - s_start
        if segment_len <= max_dur:
            final_segments.append((s_start, s_end))
        else:
            num_splits = int(np.ceil(segment_len / max_dur))
            for i in range(num_splits):
                sub_start = s_start + (i * max_dur)
                sub_end = min(s_start + ((i + 1) * max_dur), s_end)
                final_segments.append((sub_start, sub_end))
                
    return final_segments

def format_time(seconds):
    return f"[{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}]"

SUPPORTED_EXTENSIONS = {'.mp3', '.mp4', '.wav', '.m4a', '.flac', '.mov', '.mkv'}

def punctuate_text(text: str) -> str:
    """Send plain text to the /punctuate endpoint and return the polished version."""
    try:
        response = requests.post(PUNCTUATE_URL, json={"text": text}, timeout=120)
        return response.json().get("text", text)
    except Exception as e:
        print(f"[Warning] Punctuation request failed: {e}")
        return text

def process_files(file_paths, punctuate=False):
    chunk_duration = 30 
    batch_size = 4      

    for file_path in tqdm(file_paths, desc="Batch Progress"):
        if not os.path.exists(file_path): continue
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"\n[Skip] Unsupported file type: {os.path.basename(file_path)}")
            continue

        print(f"\n[Processing] {os.path.basename(file_path)} ({ext})")
        
        abs_input = os.path.abspath(file_path)
        temp_wav = os.path.abspath(f"temp_{os.getpid()}.wav")
        
        # Audio Pre-processing (Normalization)
        subprocess.run(['ffmpeg', '-y', '-i', abs_input, '-ar', '16000', '-ac', '1', '-vn', '-loglevel', 'error', temp_wav], check=True)
        
        # Get speech segments (skipping silence)
        segments = get_speech_segments(temp_wav, max_dur=chunk_duration)
        num_segments = len(segments)
        
        print(f" -> Found {num_segments} speech segments to process.")
        
        full_transcript = ""
        last_timestamp = -1

        try:
            for i in tqdm(range(0, num_segments, batch_size), desc="   Processing", leave=False):
                current_batch_files = []
                batch_segments = []

                for j in range(batch_size):
                    idx = i + j
                    if idx >= num_segments: break
                    
                    s_start, s_end = segments[idx]
                    dur = s_end - s_start
                    
                    batch_segments.append((s_start, s_end))
                    
                    chunk_file = os.path.abspath(f"chunk_{os.getpid()}_{idx}.wav")
                    subprocess.run([
                        'ffmpeg', '-y', '-ss', str(s_start), '-t', str(dur), 
                        '-i', temp_wav, '-acodec', 'pcm_s16le', '-loglevel', 'error', chunk_file
                    ], check=True)
                    
                    # Fast RMS check on the extracted small file instead of slow seeking in big file
                    try:
                        import wave as wav_lib
                        skip_chunk = False
                        with wav_lib.open(chunk_file, 'rb') as wf:
                            frames = wf.readframes(wf.getnframes())
                            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                            if len(audio_data) > 0:
                                rms = np.sqrt(np.mean(audio_data**2))
                                # 0.005 is roughly -46dB, which is above the -55dB floor but below speech
                                if rms < 0.005: 
                                    skip_chunk = True
                        
                        if skip_chunk:
                            if os.path.exists(chunk_file): os.remove(chunk_file)
                            current_batch_files.append(None) # Mark as skipped
                            continue
                    except Exception as e:
                        print(f"DEBUG RMS ERROR: {e}")
                    
                    current_batch_files.append(chunk_file)

                # Filter out skipped chunks and keep track of indices
                to_send = [f for f in current_batch_files if f is not None]
                if not to_send: continue

                # Batched Request
                try:
                    response = requests.post(SERVER_URL, json={"wav_paths": to_send}, timeout=600)
                    batch_results = response.json().get("results", [])

                except Exception as e:
                    print(f"\n[Error] Server connection failed: {e}")
                    batch_results = [""] * len(to_send)

                res_idx = 0
                for idx, chunk_file_path in enumerate(current_batch_files):
                    s_start, s_end = batch_segments[idx]
                    
                    if chunk_file_path is None:
                        # This chunk was skipped due to silence
                        continue
                    
                    chunk_text = batch_results[res_idx] if res_idx < len(batch_results) else ""
                    res_idx += 1
                    if chunk_text.strip():
                        # If there was a significant gap since the last segment, start a new speaker line
                        # Or if it's the very first segment
                        if last_timestamp == -1 or (s_start - last_timestamp) > 1.5:
                            full_transcript += f"\n\n{format_time(s_start)} SPEAKER: " + chunk_text.strip()
                        else:
                            full_transcript += " " + chunk_text.strip()
                        last_timestamp = s_end

                # Cleanup
                for f in current_batch_files:
                    if f and os.path.exists(f): os.remove(f)

            # Save File
            out_file = os.path.splitext(abs_input)[0] + ".txt"
            final_text = re.sub(r"\[\d+:\d+:\d+\] SPEAKER:$", "", full_transcript.strip()).strip()
            
            # Optionally run punctuation/capitalization on each speaker block
            if punctuate and final_text:
                print("\n[Punctuating transcript...]")
                # Process each speaker block separately to preserve timestamps
                blocks = re.split(r'(\[\d+:\d+:\d+\] SPEAKER: )', final_text)
                rebuilt = []
                i = 0
                while i < len(blocks):
                    if re.match(r'\[\d+:\d+:\d+\] SPEAKER: ', blocks[i]):
                        header = blocks[i]
                        body = blocks[i + 1].strip() if i + 1 < len(blocks) else ""
                        body = punctuate_text(body) if body else body
                        rebuilt.append(header + body)
                        i += 2
                    else:
                        if blocks[i].strip():
                            rebuilt.append(punctuate_text(blocks[i].strip()))
                        i += 1
                final_text = "\n\n".join(rebuilt)
            
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(final_text)

        finally:
            if os.path.exists(temp_wav):
                try: os.remove(temp_wav)
                except: pass

if __name__ == "__main__":
    args = sys.argv[1:]
    punctuate = "--punctuate" in args
    files = [a for a in args if not a.startswith("--")]
    if files:
        process_files(files, punctuate=punctuate)
