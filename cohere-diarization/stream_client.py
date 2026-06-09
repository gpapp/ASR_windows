"""
Real-time dual-channel streaming client.

Captures microphone (left channel, sounddevice) and system audio loopback
(right channel, pyaudiowpatch WASAPI loopback) simultaneously, sends
2-channel PCM over WebSocket to the transcription server, and writes the
received transcript to a .txt file.
"""

import asyncio
import json
import os
import queue
import sys
import signal
import threading
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pyaudiowpatch as pyaudio
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = os.getenv("STREAM_SERVER_URL", "ws://127.0.0.1:8000/ws/stream")
TARGET_RATE = 16000
CHUNK_SEC = 5.0
CHUNK_SAMPLES = int(CHUNK_SEC * TARGET_RATE)

VAD_FRAME_SEC = 0.25
VAD_FRAME_SAMPLES = int(VAD_FRAME_SEC * TARGET_RATE)
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 2 * 2  # int16 stereo
SILENCE_SEC = 1
LOOKBACK_SEC = 0.5
SILENCE_FRAMES = int(SILENCE_SEC / VAD_FRAME_SEC)
LOOKBACK_FRAMES = int(LOOKBACK_SEC / VAD_FRAME_SEC)
SILENCE_THRESHOLD = 0.002

MY_NAME = os.getenv("MY_NAME", "[ME]")


def _resample(data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample 1D float32/int16 array via linear interpolation."""
    if orig_rate == target_rate:
        return data
    n = int(len(data) * target_rate / orig_rate)
    return np.interp(
        np.linspace(0, len(data) - 1, n),
        np.arange(len(data)),
        data.astype(np.float32),
    ).astype(data.dtype)


def _rms_from_bytes(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767.0
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def _is_voice_active(chunk: bytes) -> bool:
    return _rms_from_bytes(chunk) >= SILENCE_THRESHOLD


def _split_audio_frames(chunk: bytes, frame_bytes: int = VAD_FRAME_BYTES) -> list[bytes]:
    return [chunk[i:i + frame_bytes] for i in range(0, len(chunk), frame_bytes) if len(chunk[i:i + frame_bytes]) == frame_bytes]


def find_mic_device() -> int:
    """Return the best sounddevice microphone index."""
    mic_idx = sd.default.device[0]
    if mic_idx is None:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        wasapi_id = next((i for i, h in enumerate(hostapis)
                          if "wasapi" in h["name"].lower()), None)
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                if wasapi_id is None or d["hostapi"] != wasapi_id:
                    return i
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                return i
    return mic_idx if mic_idx is not None else 0


def find_loopback_device(pa: pyaudio.PyAudio) -> dict | None:
    """Return pyaudiowpatch device-info dict for the default-output loopback."""
    # First try: loopback matching the current default output
    try:
        default_out = pa.get_default_wasapi_loopback()
        if default_out:
            return default_out
    except Exception:
        pass

    # Fallback: first loopback device available
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d.get("isLoopbackDevice"):
            return d

    return None


class StreamWriter:
    """
    Keeps the full transcript in memory and rewrites the file on every update.

    Consecutive utterances from the same speaker are merged into one block,
    so the output reads as a natural conversation rather than fragmented lines.
    """

    # Max gap (seconds) between two same-speaker utterances to still merge them
    MERGE_GAP_SEC = 2.0

    def __init__(self, path: str, recording_start: datetime):
        self.path = path
        self.recording_start = recording_start
        # Each entry: {"speaker", "start", "end", "lines": [str], "confidence": float}
        self._segments: list[dict] = []
        self._flush()

    # ------------------------------------------------------------------
    def _fmt_ts(self, sec: float) -> str:
        return (self.recording_start + timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%S")

    def _flush(self):
        """Rewrite the entire file from the in-memory segment list."""
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"Streaming Transcript — {self.recording_start.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
            f.write(f"Local speaker: {MY_NAME}\n")
            f.write("=" * 60 + "\n\n")
            for seg in self._segments:
                ts = f"[{self._fmt_ts(seg['start'])} - {self._fmt_ts(seg['end'])}]"
                conf = f" ({seg['confidence']:.0%})" if seg["confidence"] else ""
                f.write(f"{ts} {seg['speaker']}{conf}:\n")
                f.write(" ".join(seg["lines"]) + "\n\n")

    # ------------------------------------------------------------------
    def add_segment(self, msg: dict):
        speaker   = msg.get("speaker", "?")
        text      = msg.get("text", "").strip()
        confidence = float(msg.get("confidence", 0))
        start_sec  = float(msg.get("start", 0))
        end_sec    = float(msg.get("end", start_sec))

        if not text:
            return

        # Try to merge into the last segment if same speaker and gap is small
        if self._segments:
            last = self._segments[-1]
            gap = start_sec - last["end"]
            if last["speaker"] == speaker and gap <= self.MERGE_GAP_SEC:
                last["lines"].append(text)
                last["end"] = max(last["end"], end_sec)
                # Running average confidence
                n = len(last["lines"])
                last["confidence"] = last["confidence"] + (confidence - last["confidence"]) / n
                self._flush()
                return

        # New segment
        self._segments.append({
            "speaker":    speaker,
            "start":      start_sec,
            "end":        end_sec,
            "lines":      [text],
            "confidence": confidence,
        })
        self._flush()

    def add_info(self, msg: dict):
        """Append a non-speech annotation (e.g. new speaker persisted)."""
        if msg.get("type") == "new_speaker_persisted":
            spk = msg.get("speaker", "?")
            sec = msg.get("total_speech_sec", msg.get("total_sec", 0))
            self._segments.append({
                "speaker":    f"[{spk} persisted — {sec:.0f}s]",
                "start":      self._segments[-1]["end"] if self._segments else 0,
                "end":        self._segments[-1]["end"] if self._segments else 0,
                "lines":      [""],
                "confidence": 0,
            })
            self._flush()


async def run_stream(output_path: str):
    """Main streaming loop."""
    recording_start = datetime.now(timezone.utc).astimezone()
    recording_ts = recording_start.strftime("%Y%m%d_%H%M%S")

    if not output_path:
        output_path = f"stream_{recording_ts}.txt"

    writer = StreamWriter(output_path, recording_start)

    # --- Device resolution ---
    pa = pyaudio.PyAudio()

    env_mic = os.getenv("STREAM_MIC_DEVICE")
    mic_idx = int(env_mic) if env_mic and env_mic.isdigit() else find_mic_device()
    mic_name = sd.query_devices(mic_idx)["name"]

    env_loop = os.getenv("STREAM_LOOPBACK_DEVICE")
    if env_loop and env_loop.isdigit():
        loop_info = pa.get_device_info_by_index(int(env_loop))
    else:
        loop_info = find_loopback_device(pa)

    loop_name = loop_info["name"] if loop_info else "None"
    loop_rate = int(loop_info["defaultSampleRate"]) if loop_info else TARGET_RATE
    loop_channels = min(2, int(loop_info["maxInputChannels"])) if loop_info else 1

    print(f"[INFO] Output       : {output_path}")
    print(f"[INFO] Local speaker: {MY_NAME}")
    print(f"[INFO] Mic          : [{mic_idx}] {mic_name}")
    print(f"[INFO] Loopback     : {loop_name}  ({loop_rate} Hz, {loop_channels}ch)")
    print(f"[INFO] Connecting to {SERVER_URL} ...")

    send_queue: asyncio.Queue[bytes] = asyncio.Queue()
    running = True

    # Per-channel queues used to merge variable-size blocks into fixed CHUNK_SAMPLES
    mic_pcm:  queue.Queue[np.ndarray] = queue.Queue()
    loop_pcm: queue.Queue[np.ndarray] = queue.Queue()

    # ------------------------------------------------------------------ #
    #  Capture threads                                                     #
    # ------------------------------------------------------------------ #

    def _capture_mic():
        """Sounddevice mic capture → mic_pcm queue."""
        devices = sd.query_devices()
        info = devices[mic_idx]
        native = int(info["default_samplerate"])
        blocksize = int(CHUNK_SAMPLES * native / TARGET_RATE)
        try:
            sd.check_input_settings(device=mic_idx, samplerate=TARGET_RATE)
            native = TARGET_RATE
            blocksize = CHUNK_SAMPLES
        except Exception:
            pass

        def _cb(indata, frames, t, status):
            mono = indata[:, 0].copy()
            if native != TARGET_RATE:
                mono = _resample(mono, native, TARGET_RATE)
            mic_pcm.put(mono)

        with sd.InputStream(device=mic_idx, channels=1, samplerate=native,
                            blocksize=blocksize, dtype=np.float32, callback=_cb):
            while running:
                threading.Event().wait(0.1)

    def _capture_loop():
        """pyaudiowpatch loopback capture → loop_pcm queue."""
        if loop_info is None:
            return
        pa2 = pyaudio.PyAudio()
        loop_blocksize = int(CHUNK_SAMPLES * loop_rate / TARGET_RATE)
        buf = np.array([], dtype=np.float32)

        def _cb(in_data, frame_count, time_info, status):
            raw = np.frombuffer(in_data, dtype=np.float32)
            # Mix stereo → mono if needed
            if loop_channels == 2:
                raw = raw.reshape(-1, 2).mean(axis=1)
            nonlocal buf
            buf = np.concatenate([buf, raw])
            # Emit in CHUNK_SAMPLES-sized blocks (after resample)
            target_block = int(loop_blocksize)
            while len(buf) >= target_block:
                chunk = buf[:target_block]
                buf = buf[target_block:]
                out = _resample(chunk, loop_rate, TARGET_RATE) if loop_rate != TARGET_RATE else chunk
                loop_pcm.put(out)
            return (None, pyaudio.paContinue)

        stream = pa2.open(
            format=pyaudio.paFloat32,
            channels=loop_channels,
            rate=loop_rate,
            input=True,
            input_device_index=int(loop_info["index"]),
            frames_per_buffer=loop_blocksize,
            stream_callback=_cb,
        )
        stream.start_stream()
        while running:
            threading.Event().wait(0.1)
        stream.stop_stream()
        stream.close()
        pa2.terminate()

    # ------------------------------------------------------------------ #
    #  Merge thread: pair mic + loop blocks → send_queue                  #
    # ------------------------------------------------------------------ #

    def _merge():
        diag = 0
        while running or not mic_pcm.empty() or (loop_info and not loop_pcm.empty()):
            try:
                src_mic = mic_pcm.get(timeout=0.1)
            except queue.Empty:
                if not running:
                    break
                continue
            try:
                src_loop = loop_pcm.get(timeout=0.1) if loop_info else None
            except queue.Empty:
                src_loop = None

            if src_loop is None:
                src_loop = np.zeros_like(src_mic)

            n = min(len(src_mic), len(src_loop))
            stereo = np.zeros((n, 2), dtype=np.float32)
            stereo[:, 0] = src_mic[:n]
            stereo[:, 1] = src_loop[:n]
            send_queue.put_nowait((stereo * 32767).astype(np.int16).tobytes())

            diag += 1
            if diag % 10 == 0:
                def _rms(x):
                    return float(np.sqrt(np.mean(x ** 2)))            

    async def audio_capture():
        loop = asyncio.get_running_loop()
        t_mic  = loop.run_in_executor(None, _capture_mic)
        t_loop = loop.run_in_executor(None, _capture_loop)
        t_merge = loop.run_in_executor(None, _merge)
        await asyncio.gather(t_mic, t_loop, t_merge)

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def handle_sigint(sig, frame):
        print("\n[INFO] Ctrl-C received. Shutting down gracefully...")
        loop.call_soon_threadsafe(shutdown_event.set)

    # Register the SIGINT handler
    old_handler = signal.signal(signal.SIGINT, handle_sigint)

    try:
        import websockets
        async with websockets.connect(
            SERVER_URL,
            ping_interval=None,   # disable automatic pings — stream is never idle
            ping_timeout=None,
            open_timeout=30,
            max_size=None,        # no message size limit
        ) as ws:
            await ws.send(json.dumps({
                "my_name": MY_NAME,
                "recording_ts": recording_ts,
            }))

            capture_task = asyncio.create_task(audio_capture())

            async def sender():
                nonlocal running
                lookback = deque(maxlen=LOOKBACK_FRAMES)
                current_audio = bytearray()
                active = False
                silence_frames = 0

                async def _flush_current():
                    nonlocal active, silence_frames
                    if current_audio:
                        await ws.send(bytes(current_audio))
                        current_audio.clear()
                    active = False
                    silence_frames = 0
                    lookback.clear()

                while running or not send_queue.empty():
                    if send_queue.empty() and not running:
                        break
                    try:
                        if not running:
                            try:
                                chunk = await asyncio.wait_for(send_queue.get(), timeout=0.1)
                            except asyncio.TimeoutError:
                                break
                        else:
                            chunk = await send_queue.get()
                    except Exception:
                        running = False
                        break

                    subframes = _split_audio_frames(chunk)
                    for frame in subframes:
                        if active:
                            current_audio.extend(frame)
                            if _is_voice_active(frame):
                                silence_frames = 0
                            else:
                                silence_frames += 1
                                if silence_frames >= SILENCE_FRAMES:
                                    try:
                                        await _flush_current()
                                    except Exception:
                                        running = False
                                        break
                        else:
                            if _is_voice_active(frame):
                                active = True
                                for lookback_frame in lookback:
                                    current_audio.extend(lookback_frame)
                                current_audio.extend(frame)
                                silence_frames = 0
                                lookback.clear()
                            else:
                                lookback.append(frame)
                    if not running:
                        break

                if active and current_audio:
                    try:
                        await ws.send(bytes(current_audio))
                    except Exception:
                        running = False

            async def receiver():
                nonlocal running
                while True:
                    try:
                        raw = await ws.recv()
                        msg = json.loads(raw)
                        msg_type = msg.get("type", "")
                        if msg_type == "transcript":
                            writer.add_segment(msg)
                            ch = msg.get("channel", "?")
                            spk = msg.get("speaker", "?")
                            txt = msg.get("text", "")
                            print(f"[{ch}] {spk}: {txt[:80]}", flush=True)
                        elif msg_type == "new_speaker_persisted":
                            writer.add_info(msg)
                            print(f"[INFO] New speaker: {msg.get('speaker', '?')}", flush=True)
                        elif msg_type == "error":
                            print(f"[ERROR] Server: {msg.get('message', '?')}", flush=True)
                    except websockets.exceptions.ConnectionClosed as e:
                        print(f"[INFO] Connection closed: {e}", flush=True)
                        running = False
                        break
                    except Exception:
                        pass

            sender_task = asyncio.create_task(sender())
            receiver_task = asyncio.create_task(receiver())

            shutdown_task = asyncio.create_task(shutdown_event.wait())

            done, pending = await asyncio.wait(
                [capture_task, sender_task, receiver_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            running = False
            for task in pending:
                task.cancel()

            print("[INFO] Waiting for capture threads to terminate...", flush=True)
            await capture_task

            print("[INFO] Flushing and sending remaining audio queue...", flush=True)
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

            try:
                await ws.send(json.dumps({"type": "eof"}))
            except Exception:
                pass

            try:
                await ws.close()
            except Exception:
                pass

            print("[INFO] Waiting for server to finish processing and send final transcripts...", flush=True)
            try:
                await asyncio.wait_for(receiver_task, timeout=5.0)
            except asyncio.TimeoutError:
                receiver_task.cancel()
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
    finally:
        signal.signal(signal.SIGINT, old_handler)
        running = False
        pa.terminate()
        print(f"\n[DONE] Transcript saved to: {output_path}")


def list_devices():
    """Print all available audio devices and exit."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    print("=== sounddevice devices (mic candidates) ===")
    for i, d in enumerate(devices):
        ha_name = hostapis[d["hostapi"]]["name"] if d["hostapi"] < len(hostapis) else "?"
        print(f"  [{i}] {d['name']}  ({ha_name})  in={d['max_input_channels']} out={d['max_output_channels']}  {d['default_samplerate']:.0f} Hz")
    print(f"\nDefault input : {sd.default.device[0]}")
    print(f"Default output: {sd.default.device[1]}")

    print("\n=== pyaudiowpatch WASAPI loopback devices ===")
    pa = pyaudio.PyAudio()
    found = False
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d.get("isLoopbackDevice"):
            print(f"  [{i}] {d['name']}  in={d['maxInputChannels']}  {d['defaultSampleRate']:.0f} Hz")
            found = True
    if not found:
        print("  (none found)")
    pa.terminate()
    sys.exit(0)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Real-time dual-channel streaming transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  MY_NAME                - Your name (default: [ME], set in .env)
  STREAM_SERVER_URL      - WebSocket URL (default: ws://127.0.0.1:8000/ws/stream)
  STREAM_MIC_DEVICE      - Override mic device index
  STREAM_LOOPBACK_DEVICE - Override loopback device index
        """,
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output .txt path (default: stream_<timestamp>.txt)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all audio devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()

    try:
        asyncio.run(run_stream(args.output))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
