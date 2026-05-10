import os
import re as _re
import json
import socket
import subprocess
import threading
import time

import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID    = "ibm-granite/granite-speech-4.1-2b-plus"
SAMPLE_RATE = 16000

# VAD — silence shorter than this (seconds) is bridged (kept as speech)
VAD_MIN_SILENCE = 0.5
# VAD — speech segments shorter than this are dropped as noise
VAD_MIN_SPEECH  = 0.3
# ASR chunking — max seconds of *speech* audio per model call
CHUNK_DURATION  = 60
# Minimum samples to bother transcribing a chunk
MIN_CHUNK_SAMPLES = SAMPLE_RATE  # 1 second

# Prefix context — max words per speaker turn carried into the next chunk
PREFIX_WORDS_PER_SPEAKER = 10

SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)
SAA_PROMPT = (
    "<|audio|> Speaker attribution: Transcribe and denote who is speaking "
    "by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

processor_model = None
model           = None
vad_model       = None
vad_utils       = None
device          = None

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> bool:
    global processor_model, model, vad_model, vad_utils, device

    print("--- Initialising Granite + Silero VAD ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ASR model
    try:
        processor_model = AutoProcessor.from_pretrained(MODEL_ID)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            device_map=device,
            torch_dtype=dtype,
        )
        model.eval()
        print("Granite ASR loaded.")
    except Exception as e:
        print(f"Error loading ASR model: {e}")
        return False

    # Silero VAD (tiny, CPU is fine)
    try:
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        vad_model.eval()
        print("Silero VAD loaded.")
    except Exception as e:
        print(f"Warning: could not load Silero VAD ({e}). Falling back to fixed chunking.")
        vad_model = None

    return True

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def extract_audio(input_file: str, output_wav: str):
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_file,
             "-vn", "-acodec", "pcm_s16le",
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-y", output_wav],
            capture_output=True, check=True,
        )
        return output_wav
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode()[:200]}")
        return None


def load_audio(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        print(f"  Resampling {sr} Hz -> {SAMPLE_RATE} Hz ...")
        from scipy import signal
        data = signal.resample(data, int(len(data) / sr * SAMPLE_RATE))
    return data

# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def get_speech_segments(audio: np.ndarray) -> list:
    """
    Run Silero VAD over the full audio and return a list of
    (start_sample, end_sample) pairs covering speech regions only.
    Falls back to [(0, len(audio))] if VAD is unavailable.
    """
    if vad_model is None:
        return [(0, len(audio))]

    get_speech_timestamps = vad_utils[0]
    audio_tensor = torch.from_numpy(audio)
    segments = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=int(VAD_MIN_SILENCE * 1000),
        min_speech_duration_ms=int(VAD_MIN_SPEECH * 1000),
        return_seconds=False,
    )
    return [(s["start"], s["end"]) for s in segments]


def group_segments_into_chunks(segments: list, max_samples: int) -> list:
    """
    Greedily pack speech segments into groups so each group's total speech
    duration does not exceed max_samples. Segments are never split.
    """
    groups = []
    current_group = []
    current_samples = 0

    for seg in segments:
        seg_len = seg[1] - seg[0]
        if current_samples + seg_len > max_samples and current_group:
            groups.append(current_group)
            current_group = []
            current_samples = 0
        current_group.append(seg)
        current_samples += seg_len

    if current_group:
        groups.append(current_group)

    return groups


def build_chunk_audio(audio: np.ndarray, group: list) -> np.ndarray:
    """Concatenate speech segments for a group into a dense audio array."""
    return np.concatenate([audio[s:e] for s, e in group])

# ---------------------------------------------------------------------------
# Prefix trimming
# ---------------------------------------------------------------------------

def _trim_prefix(transcript: str) -> str:
    """
    Build a short prefix from the transcript for the next chunk.

    Rules:
    - Only include complete speaker turns (tag + text both present).
    - Keep the last PREFIX_WORDS_PER_SPEAKER words of each turn's text.
    - Never end the prefix mid-sentence — if the final turn has no text
      after its tag (model was cut off), drop that tag entirely.
    - If no speaker tags are found at all, return "" so the model starts
      fresh rather than trying to continue a dangling sentence.
    """
    parts = _re.split(r"(\[Speaker \d+\]:)", transcript)
    # Collect (tag, text) pairs — only where text is non-empty after strip
    turns = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if _re.match(r"\[Speaker \d+\]:", part):
            tag  = part
            text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if text:   # drop tags with no following text
                trimmed = " ".join(text.split()[-PREFIX_WORDS_PER_SPEAKER:])
                turns.append(f"{tag} {trimmed}")
            i += 2
        else:
            i += 1   # skip leading text before first tag

    return " ".join(turns).strip()

# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

@torch.inference_mode()
def transcribe_chunk(wav_chunk: torch.Tensor, prefix_text=None) -> str:
    tokenizer   = processor_model.tokenizer
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": SAA_PROMPT},
    ]
    extra       = {"prefix_text": prefix_text} if prefix_text else {}
    prompt_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, **extra
    )
    inputs  = processor_model(prompt_text, wav_chunk, device=device, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False, num_beams=1, repetition_penalty=1.3)
    new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)

# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_file(file_path: str) -> str:
    if not processor_model or not model:
        return json.dumps({"error": "Model not loaded"})

    wav_file = None
    try:
        print(f"\nProcessing: {file_path}")

        if not file_path.lower().endswith(".wav"):
            wav_file   = os.path.splitext(file_path)[0] + "_extracted.wav"
            print("  Extracting audio ...")
            if not extract_audio(file_path, wav_file):
                return json.dumps({"error": "ffmpeg extraction failed"})
            audio_path = wav_file
        else:
            audio_path = file_path

        print("  Loading audio ...")
        audio    = load_audio(audio_path)
        duration = len(audio) / SAMPLE_RATE
        print(f"  Duration: {duration:.1f}s")

        # Global VAD pass
        print("  Running VAD ...")
        t_vad    = time.time()
        segments = get_speech_segments(audio)
        speech_s = sum(e - s for s, e in segments) / SAMPLE_RATE
        print(f"  VAD: {len(segments)} segments, {speech_s:.1f}s speech "
              f"({100 * speech_s / duration:.0f}% of audio) "
              f"in {time.time() - t_vad:.1f}s")

        # Group VAD segments into ASR chunks
        max_samples = CHUNK_DURATION * SAMPLE_RATE
        groups      = group_segments_into_chunks(segments, max_samples)
        print(f"  {len(groups)} ASR chunk(s), max {CHUNK_DURATION}s speech each")

        # Transcribe
        transcriptions      = []
        previous_transcript = ""

        for i, group in enumerate(groups):
            chunk_audio = build_chunk_audio(audio, group)
            if len(chunk_audio) < MIN_CHUNK_SAMPLES:
                continue

            chunk_tensor = torch.from_numpy(chunk_audio).unsqueeze(0).float()
            speech_dur   = len(chunk_audio) / SAMPLE_RATE

            t0 = time.time()
            print(f"  Chunk {i+1}/{len(groups)} "
                  f"({speech_dur:.0f}s speech, {len(group)} segment(s)) ...")

            text = transcribe_chunk(chunk_tensor, prefix_text=previous_transcript or None)

            del chunk_tensor
            if device == "cuda":
                torch.cuda.empty_cache()

            transcriptions.append(text)
            print(f"    -> [{time.time()-t0:.1f}s] {text[:120]}")

            previous_transcript = _trim_prefix(text)

        full_transcription = " ".join(transcriptions)
        print(f"\n{'='*60}\nDIARIZED TRANSCRIPTION:\n{'='*60}\n{full_transcription}")

        return json.dumps({
            "success":       True,
            "transcription": full_transcription,
            "chunks":        len(groups),
            "speech_ratio":  round(speech_s / duration, 2),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})

    finally:
        if wav_file and os.path.exists(wav_file):
            os.remove(wav_file)

# ---------------------------------------------------------------------------
# Socket server
# ---------------------------------------------------------------------------

def handle_client(conn, addr):
    print(f"Client connected: {addr}")
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        request   = json.loads(data.decode().strip())
        file_path = request.get("file_path")
        result    = process_file(file_path) if file_path else json.dumps({"error": "No file_path"})
        conn.sendall(result.encode() + b"\n")
    except Exception as e:
        print(f"Handler error: {e}")
        try:
            conn.sendall(json.dumps({"error": str(e)}).encode() + b"\n")
        except Exception:
            pass
    finally:
        conn.close()
        print(f"Client disconnected: {addr}")


def run_server(port: int = 8765):
    if not load_model():
        print("Failed to load model — exiting.")
        return

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(5)

    print(f"\n{'='*60}")
    print(f"Granite Diarization Server on port {port}")
    print(f"{'='*60}\n")

    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        srv.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(args.port)