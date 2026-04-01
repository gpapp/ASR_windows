import sys
import os
import re
import wave
import subprocess
import requests
import numpy as np
from pathlib import Path

SERVER_URL = "http://127.0.0.1:8000"
SILENCE_NOISE = "-35dB"
SILENCE_DURATION = "2.0"
MIN_SEGMENT_DUR = 0.5
MERGE_GAP = 2.5
MAX_CHUNK_DUR = 30.0
BATCH_SIZE = 4
RMS_SILENCE_THRESHOLD = 0.005
SPEAKER_TURN_GAP = 1.5
SAMPLE_RATE = 16000

SUPPORTED = {".mp3", ".mp4", ".wav", ".m4a", ".flac", ".mov", ".mkv", ".avi", ".webm", ".ogg"}


def ffmpeg_convert(input_path: str, output_wav: str):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-vn",
        "-loglevel", "error", output_wav,
    ], check=True)


def get_total_duration(wav_path: str) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path,
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def get_speech_segments(wav_path: str) -> list:
    result = subprocess.run([
        "ffmpeg", "-i", wav_path,
        "-af", f"silencedetect=noise={SILENCE_NOISE}:d={SILENCE_DURATION}",
        "-f", "null", "-",
    ], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)

    total_dur = get_total_duration(wav_path)

    silence_starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", result.stderr)]
    silence_ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", result.stderr)]

    if not silence_starts:
        return split_segment(0.0, total_dur)

    speech = []
    pos = 0.0
    for i, ss in enumerate(silence_starts):
        if ss > pos:
            speech.append((pos, ss))
        pos = silence_ends[i] if i < len(silence_ends) else total_dur
    if pos < total_dur:
        speech.append((pos, total_dur))

    # Filter very short blips
    speech = [(s, e) for s, e in speech if (e - s) >= MIN_SEGMENT_DUR]

    # Merge segments with small gaps
    merged = []
    for seg in speech:
        if merged and (seg[0] - merged[-1][1]) < MERGE_GAP:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))

    # Split long segments into MAX_CHUNK_DUR sub-chunks
    result_segs = []
    for s, e in merged:
        result_segs.extend(split_segment(s, e))
    return result_segs


def split_segment(start: float, end: float) -> list:
    dur = end - start
    if dur <= MAX_CHUNK_DUR:
        return [(start, end)]
    n = int(dur / MAX_CHUNK_DUR) + 1
    step = dur / n
    return [(start + i * step, start + (i + 1) * step) for i in range(n)]


def extract_chunk(wav_path: str, start: float, duration: float, output: str):
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path,
        "-ss", str(start), "-t", str(duration),
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-acodec", "pcm_s16le", "-loglevel", "error", output,
    ], check=True)


def rms_check(wav_path: str) -> bool:
    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            if not frames:
                return False
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(samples ** 2)) / 32768.0
            return rms >= RMS_SILENCE_THRESHOLD
    except Exception:
        return False


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_file(input_path: str):
    p = Path(input_path).resolve()
    if p.suffix.lower() not in SUPPORTED:
        print(f"[SKIP] Unsupported format: {input_path}")
        return

    pid = os.getpid()
    temp_wav = str(p.parent / f"temp_{pid}.wav")
    output_txt = str(p.parent / (p.stem + ".txt"))

    print(f"[INFO] Processing: {p.name}")

    try:
        print("[INFO] Converting to 16kHz mono WAV...")
        ffmpeg_convert(str(p), temp_wav)

        segments = get_speech_segments(temp_wav)
        print(f"[INFO] Found {len(segments)} speech segments")

        with open(output_txt, "w", encoding="utf-8") as out_f:
            last_end = 0.0
            batch_idx = 0

            for i in range(0, len(segments), BATCH_SIZE):
                batch = segments[i : i + BATCH_SIZE]
                chunk_files = []
                chunk_starts = []

                for j, (start, end) in enumerate(batch):
                    dur = end - start
                    chunk_path = str(p.parent / f"chunk_{pid}_{batch_idx + j}.wav")
                    try:
                        extract_chunk(temp_wav, start, dur, chunk_path)
                        if rms_check(chunk_path):
                            chunk_files.append(chunk_path)
                            chunk_starts.append((start, end))
                        else:
                            os.remove(chunk_path)
                    except Exception as e:
                        print(f"[WARN] Chunk extraction failed at {fmt_time(start)}: {e}")

                if chunk_files:
                    try:
                        resp = requests.post(
                            f"{SERVER_URL}/transcribe",
                            json={"wav_paths": chunk_files},
                            timeout=120,
                        )
                        resp.raise_for_status()
                        results = resp.json()["results"]

                        for (start, end), text in zip(chunk_starts, results):
                            if not text.strip():
                                continue
                            if last_end == 0.0 or (start - last_end) > SPEAKER_TURN_GAP:
                                out_f.write(f"\n[{fmt_time(start)}] SPEAKER:\n")
                            out_f.write(text.strip() + " ")
                            out_f.flush()
                            last_end = end
                            print(f"  [{fmt_time(start)}] {text.strip()[:80]}")

                    except Exception as e:
                        print(f"[ERROR] Server request failed: {e}")
                    finally:
                        for cf in chunk_files:
                            try:
                                os.remove(cf)
                            except Exception:
                                pass

                batch_idx += len(batch)

        # Strip trailing empty speaker label if present
        with open(output_txt, "r", encoding="utf-8") as f:
            content = f.read()
        content = re.sub(r"\[\d{2}:\d{2}:\d{2}\] SPEAKER:\s*$", "", content.rstrip()).rstrip()
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(content + "\n")

        print(f"[INFO] Saved: {output_txt}")

    except Exception as e:
        print(f"[ERROR] Failed to process {input_path}: {e}")
        raise
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <file1> [file2] ...")
        sys.exit(1)

    for arg in sys.argv[1:]:
        transcribe_file(arg)
