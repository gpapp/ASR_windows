"""
Real-time dual-channel streaming transcription via WebSocket.
"""

import json
import asyncio
from pathlib import Path

import torch
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
import structlog

from fastapi import WebSocket, WebSocketDisconnect

from settings import get_settings
from model_state import state
from config import get
from transcriber import transcribe_audio_sync, clean_transcript
from speaker.audio import extract_fbank, generate_sliding_windows
from speaker.vad import run_vad_onnx_direct, split_at_energy_dips
from speaker.profiling import profile_speakers
from diarization.segment_ops import absorb_islands, eliminate_ghost_speakers
from typing import Optional

log = structlog.get_logger()

VOICEPRINTS_PATH = Path(__file__).parent / "voiceprints.json"


# ============================================================================
# Voiceprint Management
# ============================================================================

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
    """Extract a single L2-normalised speaker embedding from an audio chunk."""
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
        # Ensure explicit float32 dtype for iGPU execution
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


# ============================================================================
# Speaker Window Processing
# ============================================================================

def _process_speaker_window(
    audio_1d: np.ndarray,
    voiceprints: dict,
    window_start: float,
    recording_ts: str,
    session_new_speakers: dict,
    speaker_kv_caches: dict[str, dict], # New parameter for persistent KV cache
    speaker_prefix_ids: dict[str, list[int]], # New parameter for token prefixing
    sr: int = 16000,
) -> tuple[list[dict], dict]:
    """
    Run VAD + energy-dip splitting + sliding-window diarization + ASR on a
    mono float32 chunk of arbitrary length (typically one VAD-bounded utterance).
    Returns (transcript_messages, updated_session_new_speakers).
    """
    results = []
    ASR_MIN = int(sr * 3.0)

    if len(audio_1d) < sr * 0.5:
        return results, session_new_speakers

    wav_t = torch.from_numpy(audio_1d).float().unsqueeze(0)

    # 1. VAD
    if not state.vad_session:
        return results, session_new_speakers

    try:
        speech_ts = run_vad_onnx_direct(
            audio_1d, state.vad_session,
            sample_rate=sr,
            threshold=get_settings().vad_threshold,
            min_speech_duration_ms=get_settings().vad_min_speech_duration_ms,
        )
    except Exception:
        return results, session_new_speakers

    if not speech_ts:
        return results, session_new_speakers

    # 1b. Energy-dip splitting
    speech_ts = split_at_energy_dips(speech_ts, audio_1d, sample_rate=sr)

    # 2. Sliding-window embedding extraction
    MIN_EMBED_DUR = 1.5
    all_fbanks = []
    all_seg_meta = []
    embeddable_idxs = []

    for ts in speech_ts:
        s = int(ts["start"] * sr)
        e = int(ts["end"] * sr)
        seg = wav_t[:, s:e]
        windows, starts = generate_sliding_windows(seg, sr, window_sec=2.0, stride_sec=0.75)
        for w, rel_start in zip(windows, starts):
            dur = w.shape[-1] / sr
            gs = ts["start"] + rel_start
            ge = gs + dur
            idx = len(all_seg_meta)
            all_seg_meta.append({"start": gs, "end": ge})
            if dur >= MIN_EMBED_DUR:
                if w.shape[-1] < 1600:
                    w = torch.nn.functional.pad(w, (0, 1600 - w.shape[-1]))
                all_fbanks.append(extract_fbank(w, sr))
                embeddable_idxs.append(idx)

    if not all_fbanks or not state.embedding_session:
        # Fallback: transcribe whole utterance as unknown speaker
        asr_audio = audio_1d if len(audio_1d) >= ASR_MIN else np.pad(audio_1d, (0, ASR_MIN - len(audio_1d)))
        result = transcribe_audio_sync(asr_audio)
        text = result.get("text", "")
        if text.strip():
            results.append({
                "type": "transcript",
                "channel": "speakers",
                "speaker": "SPEAKER_1",
                "text": clean_transcript(text),
                "start": round(window_start, 2),
                "end": round(window_start + len(audio_1d) / sr, 2),
                "confidence": 0.5,
            })
        return results, session_new_speakers

    # Batch embedding
    max_len = max(fb.shape[1] for fb in all_fbanks)
    padded = []
    for fb in all_fbanks:
        if fb.shape[1] < max_len:
            fb = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
        padded.append(fb)
    batch = torch.stack(padded, dim=0)
    batch = (batch - batch.mean(dim=2, keepdim=True)).squeeze(1).numpy().astype(np.float32)

    raw_embs = []
    bs = 32
    for i in range(0, len(batch), bs):
        audio_input = batch[i:i+bs].astype(np.float32)
        out = state.embedding_session.run(None, {"feats": audio_input})
        raw_embs.append(out[0])
    raw_embs = np.concatenate(raw_embs, axis=0)
    norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
    raw_embs = raw_embs / np.maximum(norms, 1e-12)

    # 3. Clustering
    settings = get_settings()
    if len(raw_embs) > 1:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=settings.diarization_threshold,
        )
        labels = clusterer.fit_predict(raw_embs)
    else:
        labels = np.array([0])

    # 4. Map raw clusters to speaker names (try voiceprint match)
    cluster_centroids = {}
    for cid in set(labels):
        mask = labels == cid
        mean_emb = raw_embs[mask].mean(axis=0)
        norm_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
        cluster_centroids[int(cid)] = norm_emb

    # Assign raw labels to segments
    for idx, label in zip(embeddable_idxs, labels):
        all_seg_meta[idx]["raw_label"] = int(label)

    # Assign short windows to nearest embeddable window
    emb_mids = np.array([
        (all_seg_meta[i]["start"] + all_seg_meta[i]["end"]) / 2
        for i in embeddable_idxs
    ])
    for i, seg in enumerate(all_seg_meta):
        if "raw_label" not in seg:
            mid = (seg["start"] + seg["end"]) / 2
            nearest = int(np.argmin(np.abs(emb_mids - mid)))
            seg["raw_label"] = all_seg_meta[embeddable_idxs[nearest]]["raw_label"]

    # Match clusters to voiceprints, then to session speakers
    accept_thresh = get("matching", "accept_threshold", 0.35)
    clear_gap = get("matching", "clear_winner_gap", 0.02)
    embed_only_t = get("matching", "embed_only_threshold", 0.16)
    embed_only_acc = get("matching", "embed_only_accept_threshold", 0.22)
    clust_thresh = get("diarization", "default_threshold", 0.35)

    speaker_map = {}
    next_spk_num = len(session_new_speakers) + 1

    for raw_id, centroid in sorted(cluster_centroids.items()):
        # --- voiceprint match ---
        best_vp, best_vp_dist, second_vp_dist = None, 1.0, 1.0
        for vp_name, vp_data in voiceprints.items():
            if "embedding" not in vp_data:
                continue
            d = float(cosine(centroid.tolist(), vp_data["embedding"]))
            if d < best_vp_dist:
                second_vp_dist = best_vp_dist
                best_vp_dist = d
                best_vp = vp_name
            elif d < second_vp_dist:
                second_vp_dist = d

        vp_gap = second_vp_dist - best_vp_dist
        if best_vp and best_vp_dist < accept_thresh and (len(voiceprints) == 1 or vp_gap >= clear_gap):
            speaker_map[raw_id] = best_vp
            continue
        if best_vp and best_vp_dist < embed_only_t and best_vp_dist < embed_only_acc:
            speaker_map[raw_id] = best_vp
            continue

        # --- session speaker match (stable SPEAKER_N across utterances) ---
        best_sess, best_sess_dist = None, 1.0
        for sname, sdata in session_new_speakers.items():
            se = np.array(sdata["embedding"])
            d = float(1.0 - np.dot(centroid, se))
            if d < best_sess_dist:
                best_sess_dist = d
                best_sess = sname

        if best_sess and best_sess_dist < clust_thresh:
            speaker_map[raw_id] = best_sess
            # Update running-average centroid
            n = session_new_speakers[best_sess].get("n_utterances", 1)
            old = np.array(session_new_speakers[best_sess]["embedding"])
            updated = (old * n + centroid) / (n + 1)
            updated /= (np.linalg.norm(updated) + 1e-12)
            session_new_speakers[best_sess]["embedding"] = updated.tolist()
            session_new_speakers[best_sess]["n_utterances"] = n + 1
        else:
            spk_key = f"SPEAKER_{next_spk_num}"
            next_spk_num += 1
            speaker_map[raw_id] = spk_key
            session_new_speakers[spk_key] = {
                "embedding": centroid.tolist(),
                "audio_fragments": [],
                "total_sec": 0.0,
                "confidence_sum": 0.0,
                "count": 0,
                "n_utterances": 1,
            }

    # 5. Merge contiguous same-speaker windows into segments
    merged = []
    all_seg_meta.sort(key=lambda x: x["start"])
    cur = None
    for seg in all_seg_meta:
        spk = speaker_map.get(seg.get("raw_label"), "SPEAKER_1")
        if cur is None:
            cur = {"start": seg["start"], "end": seg["end"], "speaker": spk}
        elif cur["speaker"] == spk and seg["start"] <= cur["end"] + 1.0:
            cur["end"] = max(cur["end"], seg["end"])
        else:
            merged.append(cur)
            cur = {"start": seg["start"], "end": seg["end"], "speaker": spk}
    if cur:
        merged.append(cur)

    # 5b. Island absorption + ghost-speaker elimination
    merged = absorb_islands(merged)
    merged = eliminate_ghost_speakers(merged, profiles=None)

    # 6. Transcribe each merged segment
    for seg in merged:
        s = max(0, int(seg["start"] * sr))
        e = min(len(audio_1d), int(seg["end"] * sr))
        if e - s < int(sr * 0.3):
            continue
        seg_audio = audio_1d[s:e]

        spk = seg["speaker"]
        current_kv_cache = speaker_kv_caches.get(spk)
        current_prefix_ids = speaker_prefix_ids.get(spk)

        dur = (e - s) / sr
        asr_audio = seg_audio if len(seg_audio) >= ASR_MIN else np.pad(seg_audio, (0, ASR_MIN - len(seg_audio)))
        try:
            result = transcribe_audio_sync(asr_audio, past_kv_cache_ort=current_kv_cache, prefix_ids=current_prefix_ids)
            text = result.get("text", "")
            token_count = result.get("tokens_generated", 0)
            conf_score = min(0.95, max(0.3, token_count / (dur * 10 + 1)))
            speaker_kv_caches[spk] = result.get("past_kv_cache_ort") # Store updated KV cache
            speaker_prefix_ids[spk] = result.get("last_token_ids") # Store token IDs for next utterance
        except Exception:
            text = ""
            conf_score = 0.0
        
        if not text.strip():
            continue

        spk = seg["speaker"]
        results.append({
            "type": "transcript",
            "channel": "speakers",
            "speaker": spk,
            "text": clean_transcript(text),
            "start": round(window_start + seg["start"], 2),
            "end": round(window_start + seg["end"], 2),
            "confidence": round(conf_score, 3),
        })

        # Accumulate audio for persistence
        if spk in session_new_speakers:
            session_new_speakers[spk]["audio_fragments"].append(seg_audio)
            session_new_speakers[spk]["total_sec"] += dur
            session_new_speakers[spk]["confidence_sum"] += conf_score
            session_new_speakers[spk]["count"] += 1

    return results, session_new_speakers


def _persist_new_speakers(
    session_new_speakers: dict,
    voiceprints: dict,
    recording_ts: str,
    sr: int = 16000,
    min_speech_sec: float = 30.0,
    min_confidence: float = 0.8,
):
    """
    Persist new speakers to voiceprints.json if they have enough data and confidence.
    Uses naming pattern: recording_timestamp_SPEAKER_x
    """
    persisted = []
    for spk_key, data in list(session_new_speakers.items()):
        if data["total_sec"] < min_speech_sec:
            continue
        avg_conf = data["confidence_sum"] / max(1, data["count"])
        if avg_conf < min_confidence:
            continue

        # Concatenate all audio fragments
        if not data["audio_fragments"]:
            continue
        all_audio = np.concatenate(data["audio_fragments"])
        dur = len(all_audio) / sr
        if dur < 5.0:
            continue

        # Estimate embedding
        emb = _estimate_embedding_from_audio(all_audio, sr)
        if emb is None:
            continue

        # Generate speaker profile
        wav_t = torch.from_numpy(all_audio).float().unsqueeze(0)
        mock_segments = [{"start": 0.0, "end": dur, "speaker": spk_key}]
        prof = profile_speakers(wav_t, mock_segments, sr)
        speaker_prof = prof.get(spk_key, {})
        speaker_prof["embedding"] = emb.tolist() if isinstance(emb, np.ndarray) else emb

        # Assign persistent name
        existing_names = [n for n in voiceprints.keys() if n.startswith(f"{recording_ts}_SPEAKER")]
        next_n = len(existing_names) + 1
        persistent_name = f"{recording_ts}_SPEAKER_{next_n}"

        voiceprints[persistent_name] = speaker_prof
        persisted.append(persistent_name)

        # Clean up session tracking
        del session_new_speakers[spk_key]

    if persisted:
        _stream_save_voiceprints(voiceprints)
        for name in persisted:
            log.info("new_speaker_persisted", name=name)


# ============================================================================
# WebSocket Endpoint
# ============================================================================

async def stream_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dual-channel streaming transcription.

    Client sends:
    1. JSON config: {"my_name": "...", "recording_ts": "YYYYMMDD_HHMMSS"}
    2. Binary PCM frames: 2-channel int16, 16kHz, interleaved [L=mic, R=speakers]

    VAD detects sentence boundaries; each utterance is processed with the full
    sliding-window diarization pipeline (energy-dip split → embed → cluster →
    voiceprint match → ASR).  SPEAKER_N labels are stable across the session.
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
    mic_kv_cache: Optional[dict] = None # KV cache for the mic channel
    mic_prefix_ids: Optional[list[int]] = None # Token prefix for mic channel
    speaker_kv_caches: dict[str, dict] = {} # KV caches for remote speakers
    speaker_prefix_ids: dict[str, list[int]] = {} # Token prefixes per speaker
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

    def _process_mic_utterance(audio: np.ndarray, past_kv_cache_mic: Optional[dict], prefix_ids_mic: Optional[list[int]]) -> tuple[list[dict], Optional[dict], Optional[list[int]]]:
        """Simple VAD-gated ASR for the mic channel (speaker is always MY_NAME)."""
        updated_kv_cache_mic = past_kv_cache_mic
        updated_prefix_ids_mic = prefix_ids_mic
        asr_in = audio if len(audio) >= ASR_MIN else np.pad(audio, (0, ASR_MIN - len(audio)))
        try:
            result = transcribe_audio_sync(asr_in, past_kv_cache_ort=past_kv_cache_mic, prefix_ids=prefix_ids_mic)
            text = clean_transcript(result.get("text", ""))
            updated_kv_cache_mic = result.get("past_kv_cache_ort")
            updated_prefix_ids_mic = result.get("last_token_ids")
        except Exception:
            text = ""

        if not text.strip():
            return [], updated_kv_cache_mic, updated_prefix_ids_mic
        return [{"type": "transcript", "channel": "mic", "speaker": my_name,
                 "text": text, "confidence": 0.9}], updated_kv_cache_mic, updated_prefix_ids_mic

    async def flush_utterance(ch: ChannelState, channel: str):
        nonlocal session_new_speakers, voiceprints, mic_kv_cache, mic_prefix_ids, speaker_kv_caches, speaker_prefix_ids
        audio = ch.utt_buf.copy()
        ch.utt_buf = np.array([], dtype=np.float32)
        ch.in_speech = False
        ch.silence_samples = 0
        start_time = ch.offset

        if len(audio) < int(MIN_UTT_SEC * SR):
            return

        if channel == "mic":
            msgs, updated_mic_kv_cache, updated_mic_prefix_ids = await asyncio.to_thread(_process_mic_utterance, audio, mic_kv_cache, mic_prefix_ids)
            mic_kv_cache = updated_mic_kv_cache # Update mic KV cache
            mic_prefix_ids = updated_mic_prefix_ids # Update mic prefix token IDs
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
                audio, voiceprints, start_time, recording_ts, session_new_speakers, speaker_kv_caches, speaker_prefix_ids,
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

        # Normal loop termination
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
