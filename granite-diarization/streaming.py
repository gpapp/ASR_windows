"""
Real-time dual-channel streaming transcription via WebSocket.
Uses Granite SAA for speaker-attributed ASR + ECAPA-TDNN for voiceprint matching.
"""

import json
import asyncio
from pathlib import Path

import torch
import numpy as np
from scipy.spatial.distance import cosine
import structlog

from fastapi import WebSocket, WebSocketDisconnect

from settings import get_settings
from model_state import state
from config import get
from transcriber import transcribe_saa_sync, parse_speaker_turns, clean_transcript
from speaker.audio import extract_fbank, generate_sliding_windows
from speaker.vad import run_vad_onnx_direct, split_at_energy_dips
from speaker.profiling import profile_speakers
from diarization.segment_ops import absorb_islands, eliminate_ghost_speakers

log = structlog.get_logger()

VOICEPRINTS_PATH = Path(__file__).parent / "voiceprints.json"


def _stream_load_voiceprints() -> dict:
    if VOICEPRINTS_PATH.exists():
        try:
            with open(VOICEPRINTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _stream_save_voiceprints(vp: dict):
    try:
        with open(VOICEPRINTS_PATH, "w", encoding="utf-8") as f:
            json.dump(vp, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning("voiceprint_save_failed", error=str(e))


def _estimate_embedding_from_audio(
    audio_1d: np.ndarray,
    sr: int = 16000,
) -> np.ndarray | None:
    if not state.embedding_session or len(audio_1d) < sr * 1.5:
        return None
    wav_t = torch.from_numpy(audio_1d).float().unsqueeze(0)
    try:
        windows, _ = generate_sliding_windows(wav_t, sr, window_sec=3.0, stride_sec=1.5)
        fbanks = []
        for w in windows:
            if w.shape[-1] / sr >= 1.5:
                if w.shape[-1] < 4800:
                    w = torch.nn.functional.pad(w, (0, 4800 - w.shape[-1]))
                fbanks.append(extract_fbank(w, sr))
        if not fbanks:
            return None
        max_len = max(fb.shape[1] for fb in fbanks)
        padded = []
        for fb in fbanks:
            if fb.shape[1] < max_len:
                fb = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
            padded.append(fb)
        batch = torch.stack(padded, dim=0)
        batch = (batch - batch.mean(dim=2, keepdim=True)).squeeze(1).numpy().astype(np.float32)
        embs = state.embedding_session.run(None, {
            state.embedding_session.get_inputs()[0].name: batch
        })[0]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-12)
        return embs.mean(axis=0)
    except Exception as e:
        log.warning("embedding_estimation_failed", error=str(e))
        return None


def _process_speaker_window(
    audio_1d: np.ndarray,
    voiceprints: dict,
    window_start: float,
    recording_ts: str,
    session_new_speakers: dict,
    sr: int = 16000,
) -> tuple[list[dict], dict]:
    """
    Process a single utterance with Granite SAA + voiceprint matching.

    Runs Granite SAA on the audio chunk, parses [Speaker N]: tags,
    then matches speaker segments against known voiceprints.

    Returns (transcript_messages, updated_session_new_speakers).
    """
    results = []

    if len(audio_1d) < sr * 0.5:
        return results, session_new_speakers

    # Run SAA on the full utterance
    try:
        if len(audio_1d) < sr * 3:
            padded = np.pad(audio_1d, (0, sr * 3 - len(audio_1d)))
        else:
            padded = audio_1d

        result = transcribe_saa_sync(padded)
    except Exception as e:
        log.warning("saa_failed", error=str(e))
        return results, session_new_speakers

    saa_text = result.get("text", "")
    turns = parse_speaker_turns(saa_text)

    if not turns:
        return results, session_new_speakers

    # Assign time boundaries proportional to text length
    dur = len(audio_1d) / sr
    total_text_len = sum(len(t["text"]) for t in turns) or 1
    ratio_so_far = 0.0

    for turn in turns:
        turn_ratio = len(turn["text"]) / total_text_len
        turn_start = window_start + ratio_so_far * dur
        turn_end = turn_start + turn_ratio * dur
        ratio_so_far += turn_ratio

        spk = turn["speaker"]
        text = turn["text"]

        # Try voiceprint match for this speaker
        matched_name = _match_to_voiceprint_or_session(
            audio_1d, turn_start, turn_end,
            spk, voiceprints, session_new_speakers,
        )

        final_speaker = matched_name or spk

        results.append({
            "type": "transcript",
            "channel": "speakers",
            "speaker": final_speaker,
            "text": clean_transcript(text),
            "start": round(turn_start, 2),
            "end": round(turn_end, 2),
            "confidence": 0.9,
        })

        if final_speaker in session_new_speakers:
            seg_audio = audio_1d[
                max(0, int(turn_start * sr) - int(window_start * sr)):
                min(len(audio_1d), int(turn_end * sr) - int(window_start * sr))
            ]
            session_new_speakers[final_speaker]["audio_fragments"].append(seg_audio)
            session_new_speakers[final_speaker]["total_sec"] += (turn_end - turn_start)
            session_new_speakers[final_speaker]["confidence_sum"] += 0.9
            session_new_speakers[final_speaker]["count"] += 1

    return results, session_new_speakers


def _match_to_voiceprint_or_session(
    audio_1d: np.ndarray,
    seg_start: float,
    seg_end: float,
    label: str,
    voiceprints: dict,
    session_new_speakers: dict,
    sr: int = 16000,
) -> str | None:
    """Match a speaker segment to known voiceprints or session speakers."""
    if not state.embedding_session:
        return None

    s = max(0, int(seg_start * sr))
    e = min(len(audio_1d), int(seg_end * sr))
    seg_audio = audio_1d[s:e]

    if len(seg_audio) < sr * 1.5:
        return None

    # Compute embedding for this segment
    emb = _estimate_embedding_from_audio(seg_audio, sr)
    if emb is None:
        return None

    # Try voiceprint match
    best_vp, best_vp_dist, second_vp_dist = None, 1.0, 1.0
    for vp_name, vp_data in voiceprints.items():
        if "embedding" not in vp_data:
            continue
        d = float(cosine(emb.tolist(), vp_data["embedding"]))
        if d < best_vp_dist:
            second_vp_dist = best_vp_dist
            best_vp_dist = d
            best_vp = vp_name
        elif d < second_vp_dist:
            second_vp_dist = d

    accept_thresh = get("matching", "accept_threshold", 0.35)
    clear_gap = get("matching", "clear_winner_gap", 0.02)

    if best_vp and best_vp_dist < accept_thresh and (len(voiceprints) <= 1 or (second_vp_dist - best_vp_dist) >= clear_gap):
        return best_vp

    # Try session speaker match (stable SPEAKER_N across utterances)
    best_sess, best_sess_dist = None, 1.0
    for sname, sdata in session_new_speakers.items():
        se = np.array(sdata["embedding"])
        d = float(1.0 - np.dot(emb, se))
        if d < best_sess_dist:
            best_sess_dist = d
            best_sess = sname

    clust_thresh = get("diarization", "default_threshold", 0.35)
    if best_sess and best_sess_dist < clust_thresh:
        n = session_new_speakers[best_sess].get("n_utterances", 1)
        old = np.array(session_new_speakers[best_sess]["embedding"])
        updated = (old * n + emb) / (n + 1)
        updated /= (np.linalg.norm(updated) + 1e-12)
        session_new_speakers[best_sess]["embedding"] = updated.tolist()
        session_new_speakers[best_sess]["n_utterances"] = n + 1
        return best_sess

    return None


def _persist_new_speakers(
    session_new_speakers: dict,
    voiceprints: dict,
    recording_ts: str,
    sr: int = 16000,
    min_speech_sec: float = 30.0,
    min_confidence: float = 0.8,
):
    persisted = []
    for spk_key, data in list(session_new_speakers.items()):
        if data["total_sec"] < min_speech_sec:
            continue
        avg_conf = data["confidence_sum"] / max(1, data["count"])
        if avg_conf < min_confidence:
            continue

        if not data["audio_fragments"]:
            continue
        all_audio = np.concatenate(data["audio_fragments"])
        dur = len(all_audio) / sr
        if dur < 5.0:
            continue

        emb = _estimate_embedding_from_audio(all_audio, sr)
        if emb is None:
            continue

        wav_t = torch.from_numpy(all_audio).float().unsqueeze(0)
        mock_segments = [{"start": 0.0, "end": dur, "speaker": spk_key}]
        prof = profile_speakers(wav_t, mock_segments, sr)
        speaker_prof = prof.get(spk_key, {})
        speaker_prof["embedding"] = emb.tolist() if isinstance(emb, np.ndarray) else emb

        existing_names = [n for n in voiceprints.keys() if n.startswith(f"{recording_ts}_SPEAKER")]
        next_n = len(existing_names) + 1
        persistent_name = f"{recording_ts}_SPEAKER_{next_n}"

        voiceprints[persistent_name] = speaker_prof
        persisted.append(persistent_name)

        del session_new_speakers[spk_key]

    if persisted:
        _stream_save_voiceprints(voiceprints)
        for name in persisted:
            log.info("new_speaker_persisted", name=name)


async def stream_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dual-channel streaming transcription.

    Client sends:
    1. JSON config: {"my_name": "...", "recording_ts": "YYYYMMDD_HHMMSS"}
    2. Binary PCM frames: 2-channel int16, 16kHz, interleaved [L=mic, R=speakers]

    Uses Granite SAA + voiceprint matching instead of VAD+embedding+clustering.
    """
    await websocket.accept()
    log.info("stream_connected")

    try:
        raw = await websocket.receive_json()
        my_name = raw.get("my_name", "[ME]")
        recording_ts = raw.get("recording_ts", "")
    except Exception:
        my_name = "[ME]"
        recording_ts = ""

    SR = 16000
    FRAME_SEC = 0.25
    FRAME_SAMPLES = int(FRAME_SEC * SR)
    FRAME_BYTES = FRAME_SAMPLES * 2 * 2
    SILENCE_THRESH = 0.002
    MAX_UTT_SEC = 15.0
    MIN_UTT_SEC = 0.1
    ASR_MIN = int(SR * 3)
    SILENCE_SAMPLES = int(SILENCE_THRESH * SR)
    MAX_UTT_SAMPLES = int(MAX_UTT_SEC * SR)

    class ChannelState:
        def __init__(self):
            self.utt_buf = np.array([], dtype=np.float32)
            self.in_speech = False
            self.silence_samples = 0
            self.offset = 0.0

    mic_ch = ChannelState()
    spk_ch = ChannelState()
    session_time = 0.0

    voiceprints = _stream_load_voiceprints()
    session_new_speakers: dict = {}

    def _vad_has_speech(audio: np.ndarray) -> bool:
        if not state.vad_session:
            return True
        try:
            ts = run_vad_onnx_direct(
                audio, state.vad_session,
                sample_rate=SR,
                threshold=get_settings().vad_threshold,
                min_speech_duration_ms=get_settings().vad_min_speech_duration_ms,
            )
            return bool(ts)
        except Exception:
            return True

    def _process_mic_utterance(audio: np.ndarray) -> list[dict]:
        """Simple ASR for the mic channel (speaker is always MY_NAME)."""
        asr_in = audio if len(audio) >= ASR_MIN else np.pad(audio, (0, ASR_MIN - len(audio)))
        try:
            result = transcribe_saa_sync(asr_in)
            text = result.get("text", "")
            text = clean_transcript(parse_speaker_turns_text_only(text))
        except Exception:
            text = ""

        if not text.strip():
            return []
        return [{"type": "transcript", "channel": "mic", "speaker": my_name,
                 "text": text, "confidence": 0.9}]

    def parse_speaker_turns_text_only(text: str) -> str:
        """Extract just the text from SAA output, stripping [Speaker N]: tags."""
        return ' '.join(t["text"] for t in parse_speaker_turns(text)).strip()

    async def flush_utterance(ch: ChannelState, channel: str):
        nonlocal session_new_speakers, voiceprints
        audio = ch.utt_buf.copy()
        ch.utt_buf = np.array([], dtype=np.float32)
        ch.in_speech = False
        ch.silence_samples = 0
        start_time = ch.offset

        if len(audio) < int(MIN_UTT_SEC * SR):
            return

        if channel == "mic":
            msgs = await asyncio.to_thread(_process_mic_utterance, audio)
            for msg in msgs:
                msg.setdefault("start", round(start_time, 2))
                msg.setdefault("end", round(start_time + len(audio) / SR, 2))
                try:
                    await websocket.send_json(msg)
                except Exception:
                    pass
        else:
            msgs, session_new_speakers = await asyncio.to_thread(
                _process_speaker_window,
                audio, voiceprints, start_time, recording_ts, session_new_speakers,
            )
            for msg in msgs:
                try:
                    await websocket.send_json(msg)
                except Exception:
                    pass
            await asyncio.to_thread(
                _persist_new_speakers,
                session_new_speakers, voiceprints, recording_ts,
            )

    async def process_chunk(ch: ChannelState, chunk: np.ndarray, channel: str):
        has_speech = await asyncio.to_thread(_vad_has_speech, chunk)
        if has_speech:
            if not ch.in_speech:
                ch.offset = session_time - len(ch.utt_buf) / SR
                ch.in_speech = True
            ch.utt_buf = np.concatenate([ch.utt_buf, chunk])
            ch.silence_samples = 0
            if len(ch.utt_buf) >= MAX_UTT_SAMPLES:
                await flush_utterance(ch, channel)
        else:
            if ch.in_speech:
                ch.silence_samples += len(chunk)
                ch.utt_buf = np.concatenate([ch.utt_buf, chunk])
                if ch.silence_samples >= SILENCE_SAMPLES:
                    trim = len(ch.utt_buf) - ch.silence_samples
                    ch.utt_buf = ch.utt_buf[:max(trim, 0)]
                    await flush_utterance(ch, channel)

    recv_buffer = b''

    def _iter_frames(data: bytes):
        nonlocal recv_buffer
        recv_buffer += data
        while len(recv_buffer) >= FRAME_BYTES:
            frame_bytes = recv_buffer[:FRAME_BYTES]
            recv_buffer = recv_buffer[FRAME_BYTES:]
            frame = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            frame = frame.reshape(-1, 2)
            yield frame

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=msg.get("code", 1000))

            if "bytes" in msg:
                data = msg["bytes"]
                for frame in _iter_frames(data):
                    session_time += len(frame) / SR
                    await process_chunk(mic_ch, frame[:, 0], "mic")
                    await process_chunk(spk_ch, frame[:, 1], "speakers")
            elif "text" in msg:
                try:
                    payload = json.loads(msg["text"])
                    if payload.get("type") == "eof":
                        log.info("stream_eof_received")
                        break
                except Exception:
                    pass

        if len(mic_ch.utt_buf) > 0:
            await flush_utterance(mic_ch, "mic")
        if len(spk_ch.utt_buf) > 0:
            await flush_utterance(spk_ch, "speakers")
        log.info("stream_completed_gracefully", total_seconds=round(session_time, 1))

    except WebSocketDisconnect:
        if len(mic_ch.utt_buf) > 0:
            await flush_utterance(mic_ch, "mic")
        if len(spk_ch.utt_buf) > 0:
            await flush_utterance(spk_ch, "speakers")
        log.info("stream_disconnected", total_seconds=round(session_time, 1))
    except Exception as e:
        log.error("stream_error", error=str(e))
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
