"""
Simplified hybrid diarization pipeline: Granite SAA + ECAPA-TDNN voiceprint matching.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import structlog

from settings import get_settings
from model_state import state
from config import get
from transcriber import transcribe_saa_sync, parse_speaker_turns, clean_transcript, _trim_prefix
from speaker.audio import extract_fbank, generate_sliding_windows
from speaker.vad import run_vad_onnx_direct, split_at_energy_dips
from speaker.profiling import profile_speakers
from speaker.embedding import extract_embedding, normalize_embedding
from speaker.matcher import match_clusters, compute_distance
from diarization.clustering import match_known_speakers_full
from diarization.segment_ops import collapse_same_speaker_segments, absorb_islands, eliminate_ghost_speakers

log = structlog.get_logger()

SAMPLE_RATE = 16000
CHUNK_SPEECH_MAX = 60
VAD_MIN_SILENCE = 0.5
VAD_MIN_SPEECH = 0.3
MIN_ASR_SAMPLES = SAMPLE_RATE


class Diarizer:
    """Hybrid diarizer: SAA for speaker turns + voiceprint matching via ECAPA-TDNN."""

    def __init__(self, model_state, settings):
        self.state = model_state
        self.settings = settings

    def run(self, req, queue, loop, wav_path: str):
        """
        Execute the full hybrid diarization pipeline.

        Yields progress messages via queue and returns (segments, profiles).
        """
        try:
            self._emit(queue, "progress", loop, step="loading_audio", completed=0, total=5)
            audio, sr = self._load_audio(wav_path)
            audio_duration = len(audio) / sr

            # 1. VAD for chunking
            self._emit(queue, "progress", loop, step="vad", completed=1, total=5)
            speech_segments = self._run_vad(audio, sr)
            if not speech_segments:
                self._emit_result(queue, [], {}, loop)
                return

            # 2. Group into chunks
            self._emit(queue, "progress", loop, step="chunking", completed=2, total=5)
            chunk_groups = self._group_segments(speech_segments, CHUNK_SPEECH_MAX * sr)

            # 3. SAA per chunk
            self._emit(queue, "progress", loop, step="saa_transcription", completed=3, total=5)
            all_segments, prefix_text = self._run_saa_chunks(audio, chunk_groups)

            if not all_segments:
                self._emit_result(queue, [], {}, loop)
                return

            # 4. Voiceprint matching
            self._emit(queue, "progress", loop, step="voiceprint_matching", completed=4, total=5)
            known_speakers = getattr(req, 'known_speakers', None) or {}
            all_segments = self._match_voiceprints(audio, all_segments, known_speakers)

            # 5. Profiling
            self._emit(queue, "progress", loop, step="profiling", completed=5, total=5)
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            mock_segments = self._segments_for_profiling(all_segments)
            profiles = profile_speakers(audio_tensor, mock_segments, sr)

            self._emit_result(queue, all_segments, profiles, loop)

        except Exception as e:
            log.error("diarization_failed", error=str(e))
            self._emit(queue, "type", loop, error=str(e))

    def _load_audio(self, wav_path: str):
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr

    def _run_vad(self, audio: np.ndarray, sr: int) -> list:
        """Run Silero VAD to find speech regions."""
        if not state.vad_session:
            return [(0, len(audio))]

        from speaker.vad import run_vad_onnx_direct
        try:
            ts = run_vad_onnx_direct(
                audio, state.vad_session,
                sample_rate=sr,
                threshold=self.settings.vad_threshold,
                min_speech_duration_ms=self.settings.vad_min_speech_duration_ms,
            )
        except Exception:
            return [(0, len(audio))]

        if not ts:
            return []

        ts = split_at_energy_dips(ts, audio, sample_rate=sr) if len(ts) > 1 else ts
        return [(int(s["start"] * sr), int(s["end"] * sr)) for s in ts]

    def _group_segments(self, segments: list, max_samples: int) -> list:
        """Group VAD segments into chunks capped at max_samples speech total."""
        groups = []
        current = []
        current_samples = 0

        for seg in segments:
            seg_len = seg[1] - seg[0]
            if current_samples + seg_len > max_samples and current:
                groups.append(current)
                current = []
                current_samples = 0
            current.append(seg)
            current_samples += seg_len

        if current:
            groups.append(current)
        return groups

    def _build_chunk_audio(self, audio: np.ndarray, group: list) -> np.ndarray:
        return np.concatenate([audio[s:e] for s, e in group])

    def _get_chunk_time_range(self, audio: np.ndarray, group: list) -> tuple:
        """Return (start_sec, end_sec) for a chunk based on original audio positions."""
        first = group[0][0]
        last = group[-1][1]
        return first / SAMPLE_RATE, last / SAMPLE_RATE

    def _run_saa_chunks(self, audio: np.ndarray, chunk_groups: list) -> tuple:
        """Process chunks with SAA, return (all_segments, last_prefix)."""
        all_segments = []
        prefix_text = ""

        for group in chunk_groups:
            chunk_audio = self._build_chunk_audio(audio, group)
            chunk_start, chunk_end = self._get_chunk_time_range(audio, group)

            if len(chunk_audio) < MIN_ASR_SAMPLES:
                continue

            result = transcribe_saa_sync(chunk_audio, prefix_text=prefix_text or None)

            saa_text = result.get("text", "")
            turns = parse_speaker_turns(saa_text)

            if not turns:
                continue

            total_text_len = sum(len(t["text"]) for t in turns) or 1
            chunk_dur = chunk_end - chunk_start
            ratio_so_far = 0.0

            for turn in turns:
                turn_ratio = len(turn["text"]) / total_text_len
                turn_start = chunk_start + ratio_so_far * chunk_dur
                turn_end = turn_start + turn_ratio * chunk_dur
                ratio_so_far += turn_ratio

                turn["start"] = turn_start
                turn["end"] = turn_end

            all_segments.extend(turns)
            prefix_text = _trim_prefix(saa_text)

        return all_segments, prefix_text

    def _match_voiceprints(self, audio: np.ndarray, segments: list, known_speakers: dict) -> list:
        """Per-speaker embedding + voiceprint matching, then re-label segments."""
        if not known_speakers or not state.embedding_session:
            return segments

        speaker_embeddings = {}

        unique_speakers = set(seg["speaker"] for seg in segments)
        for spk in unique_speakers:
            spk_audio = self._extract_speaker_audio(audio, segments, spk)
            if len(spk_audio) >= SAMPLE_RATE * 3:
                emb = self._compute_embedding(spk_audio)
                if emb is not None:
                    speaker_embeddings[spk] = emb

        if not speaker_embeddings:
            return segments

        match_info = match_known_speakers_full(speaker_embeddings, known_speakers)

        for spk_label, matched_name in match_info.items():
            for seg in segments:
                if seg["speaker"] == spk_label:
                    seg["speaker"] = matched_name

        return segments

    def _extract_speaker_audio(self, audio: np.ndarray, segments: list, speaker: str) -> np.ndarray:
        parts = []
        for seg in segments:
            if seg["speaker"] == speaker:
                s = max(0, int(seg["start"] * SAMPLE_RATE))
                e = min(len(audio), int(seg["end"] * SAMPLE_RATE))
                if e - s > 0:
                    parts.append(audio[s:e])
        if not parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(parts)

    def _compute_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        if not state.embedding_session or len(audio) < SAMPLE_RATE * 1.5:
            return None
        try:
            wav_t = torch.from_numpy(audio).float().unsqueeze(0)
            windows, _ = generate_sliding_windows(
                wav_t, SAMPLE_RATE, window_sec=3.0, stride_sec=1.5
            )
            fbanks = []
            for w in windows:
                dur = w.shape[-1] / SAMPLE_RATE
                if dur >= 1.5:
                    if w.shape[-1] < 4800:
                        w = torch.nn.functional.pad(w, (0, 4800 - w.shape[-1]))
                    fbanks.append(extract_fbank(w, SAMPLE_RATE))

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
            return embs.mean(axis=0).astype(np.float64)
        except Exception as e:
            log.warning("embedding_computation_failed", error=str(e))
            return None

    def _segments_for_profiling(self, segments: list) -> list:
        return [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
                for s in segments]

    def _emit(self, queue, msg_type, loop, **kwargs):
        """Send progress message to the asyncio queue (called from worker thread)."""
        import asyncio
        payload = json.dumps({"type": msg_type, **kwargs})
        try:
            if loop is not None and queue is not None:
                fut = asyncio.run_coroutine_threadsafe(queue.put(payload), loop)
                fut.result(timeout=5)
        except Exception:
            pass

    def _emit_result(self, queue, segments: list, profiles: dict, loop):
        """Send final result message."""
        self._emit(queue, "result", loop,
            segments=segments,
            profiles=profiles,
        )
