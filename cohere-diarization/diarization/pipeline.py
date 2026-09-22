"""
Full diarization pipeline: VAD → embedding → clustering → matching → refinement.
"""

import json
import asyncio
import torch
import numpy as np
import librosa
import structlog

from sklearn.cluster import AgglomerativeClustering

from settings import Settings
from model_state import ModelState
from config import get, is_debug
from api.schemas import DiarizePathsRequest
from speaker.audio import extract_fbank, generate_sliding_windows, refine_speaker_boundaries
from speaker.vad import run_vad_onnx_direct, split_at_energy_dips, merge_vad_sections
from speaker.profiling import profile_speakers, relabel_by_pitch
from .clustering import cap_clusters, greedy_merge_clusters, match_known_speakers_full
from .overlap import detect_overlaps, build_overlap_segments
from .segment_ops import collapse_same_speaker_segments, absorb_islands, eliminate_ghost_speakers

log = structlog.get_logger()


def _best_split(owners: list[int], weights: list[float]) -> int:
    """Best cut index k (before section k) minimising weighted disagreements.

    owners[i] = 0 for the left speaker, 1 for the right speaker.
    Cost of cut k: right-owned sections left of k plus left-owned sections
    right of k, weighted by embedding-distance margin (confident matches
    dominate noise). All-one-side (same speaker) collapses to k = n.
    """
    n = len(owners)
    if n == 0:
        return 0
    best_k, best_cost = 0, float("inf")
    for k in range(n + 1):
        cost = 0.0
        for i in range(k):
            if owners[i] == 1:
                cost += weights[i]
        for i in range(k, n):
            if owners[i] == 0:
                cost += weights[i]
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_k = k
    return best_k


def _gap_energy_cut(audio: np.ndarray, gap_start: float, gap_end: float,
                    sample_rate: int = 16000, frame_ms: float = 20.0,
                    dip_ratio: float = 0.35, min_dip_sec: float = 0.12) -> float:
    """Centre of the quietest pause inside a gap; midpoint as last resort."""
    mid = (gap_start + gap_end) / 2.0
    if audio is None or len(audio) == 0:
        return mid
    s = max(0, int(gap_start * sample_rate))
    e = min(len(audio), int(gap_end * sample_rate))
    frame = max(1, int(frame_ms / 1000 * sample_rate))
    if e - s < 2 * frame:
        return mid
    chunk = audio[s:e].astype(np.float64)
    n = len(chunk) // frame
    if n < 1:
        return mid
    rms = np.sqrt(np.mean(chunk[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    max_e = float(rms.max())
    if max_e < 1e-5:
        return mid
    quiet = rms < (max_e * dip_ratio)
    runs = []
    i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    min_frames = max(1, int(round(min_dip_sec * 1000 / frame_ms)))
    eligible = [r for r in runs if r[1] - r[0] >= min_frames]
    if eligible:
        centre = n / 2.0
        best = max(eligible, key=lambda r: (r[1] - r[0], -abs((r[0] + r[1]) / 2.0 - centre)))
        cut_frame = (best[0] + best[1]) // 2
    else:
        cut_frame = int(np.argmin(rms))
    cut = s + cut_frame * frame + frame // 2
    return float(min(max(cut / sample_rate, gap_start), gap_end))


class Diarizer:
    """
    Full speaker diarization pipeline.
    
    Processes audio through VAD, embedding extraction, clustering, voiceprint matching,
    and boundary refinement to produce timestamped speaker segments.
    """
    
    def __init__(self, state: ModelState, settings: Settings):
        self.state = state
        self.settings = settings
        self.MIN_EMBED_DURATION = 1.5  # seconds
    
    def run(
        self,
        req: DiarizePathsRequest,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        resolved_path: str,
    ) -> None:
        """
        Run full diarization pipeline in a thread.
        
        Streams progress as NDJSON to queue, ends with final result or error.
        """
        try:
            waveform, sr = librosa.load(str(resolved_path), sr=16000, mono=True)
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
            
            if not self.state.vad_session or not self.state.embedding_session:
                log.error("diarization_models_missing", reason="VAD session or embedding session is missing")
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "error": "Diarization models not fully loaded"
                }))
                loop.call_soon_threadsafe(queue.put_nowait, None)
                return
            
            # Pipeline stages
            speech_ts = self._run_vad(waveform_tensor, req, queue, loop)
            if not speech_ts:
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "type": "result", "segments": [], "profiles": {}
                }))
                loop.call_soon_threadsafe(queue.put_nowait, None)
                return
            
            all_fbanks, all_segments_meta, embeddable_indices = self._extract_features(
                waveform_tensor, speech_ts, queue, loop
            )
            
            if not all_fbanks:
                # Fallback: single speaker
                fallback_segments = [
                    {"start": ts["start"], "end": ts["end"], "speaker": "SPEAKER1"}
                    for ts in speech_ts
                ]
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "type": "result", "segments": fallback_segments, "profiles": {}
                }))
                loop.call_soon_threadsafe(queue.put_nowait, None)
                return
            
            raw_embeddings = self._extract_embeddings(all_fbanks, queue, loop)
            
            long_labels, cluster_centroids = self._cluster_embeddings(
                raw_embeddings, req, queue, loop
            )
            
            # Assign labels to all segments (including short ones)
            self._assign_labels_to_segments(
                all_segments_meta, embeddable_indices, long_labels
            )

            # --- OVERLAP: detect overlapping speech windows ---
            proximity_ratio = get("overlap", "proximity_ratio", 0.15)
            min_distance = get("overlap", "min_distance", 0.30)
            overlap_meta_indices = detect_overlaps(
                raw_embeddings, cluster_centroids, embeddable_indices,
                all_segments_meta, proximity_ratio, min_distance
            )

            # Map clusters to speaker names (produces OVERLAP segments when detected)
            merged_segments, speaker_map = self._map_to_speakers(
                all_segments_meta, cluster_centroids, req
            )

            # ── Split: process single-speaker segments separately ──────
            ov_segments = [s for s in merged_segments if s.get("speaker") == "OVERLAP"]
            non_ov = [s for s in merged_segments if s.get("speaker") != "OVERLAP"]

            non_ov = absorb_islands(non_ov)

            non_ov = self._refine_boundaries(
                non_ov, all_segments_meta, embeddable_indices,
                raw_embeddings, waveform_tensor
            )

            profiles = profile_speakers(waveform_tensor, non_ov, sample_rate=16000)
            non_ov, profiles, pitch_remap = relabel_by_pitch(non_ov, profiles)

            # Inject cluster centroid embeddings into every speaker's profile
            # speaker_map: raw_cluster_int → initial_speaker_name (e.g. SPEAKER1)
            # pitch_remap: initial_speaker_name → final_speaker_name (e.g. SPEAKER1 → SPEAKER3)
            centroid_emb_map = {}
            for raw_cluster, init_name in speaker_map.items():
                centroid_emb_map[init_name] = cluster_centroids[raw_cluster].tolist()
            for init_name, final_name in pitch_remap.items():
                if init_name in centroid_emb_map:
                    if final_name not in profiles:
                        profiles[final_name] = {}
                    profiles[final_name]["embedding"] = centroid_emb_map[init_name]

            # Voiceprint matching
            if req.known_speakers:
                non_ov, profiles = match_known_speakers_full(
                    non_ov, all_segments_meta, embeddable_indices,
                    raw_embeddings, cluster_centroids, profiles, req.known_speakers, req
                )

            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "progress", "step": "Clustering", "completed": 1, "total": 1
            }))

            # Ghost-speaker elimination (single-speaker segments only)
            non_ov = eliminate_ghost_speakers(non_ov, profiles=profiles)

            # Exact turn-boundary refinement: uncollapsed VAD sections
            # attributed to the adjacent speaker via voiceprints.
            try:
                non_ov = self._refine_turn_boundaries_exact(
                    non_ov, waveform_tensor, profiles, req.known_speakers,
                )
            except Exception as e:
                log.warning("exact_boundary_refinement_failed", error=str(e))

            # ── Recombine with overlap segments ────────────────────────
            merged_segments = sorted(non_ov + ov_segments, key=lambda x: x["start"])

            # Resolve time conflicts (boundary refinement may have shifted
            # single-speaker edges into overlap regions)
            resolved = []
            for seg in merged_segments:
                if resolved and seg["start"] < resolved[-1]["end"]:
                    mid = (seg["start"] + resolved[-1]["end"]) / 2.0
                    resolved[-1]["end"] = mid
                    seg["start"] = mid
                if seg["end"] > seg["start"]:
                    resolved.append(seg)
            merged_segments = resolved

            # Compute confidence scores
            segment_confidences = self._compute_confidence(
                merged_segments, all_segments_meta, cluster_centroids,
                raw_embeddings, embeddable_indices, long_labels, req
            )

            # Format final response
            final_data = self._format_response(merged_segments, segment_confidences)
            
            unique_speakers = list(set(s["speaker"] for s in final_data))
            log.info("diarization_complete", unique_speakers=unique_speakers, num_segments=len(final_data))
            
            msg = {"type": "result", "segments": final_data, "profiles": profiles}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(msg))
            loop.call_soon_threadsafe(queue.put_nowait, None)
            
        except Exception as e:
            log.error("diarization_failed", error=str(e), exc_info=True)
            err_msg = {"error": f"Diarization processing failed: {str(e)}"}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(err_msg))
            loop.call_soon_threadsafe(queue.put_nowait, None)
    
    def _run_vad(self, waveform_tensor, req, queue, loop):
        """Run Voice Activity Detection."""
        loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
            "type": "progress", "step": "Voice Activity Detection", "completed": 0, "total": 1
        }))
        
        vad_thresh_val = req.vad_threshold if req.vad_threshold is not None else self.settings.vad_threshold
        vad_min_dur_val = req.vad_min_speech_duration_ms if req.vad_min_speech_duration_ms is not None else self.settings.vad_min_speech_duration_ms
        
        waveform_np = waveform_tensor.squeeze(0).numpy()
        
        log.info("vad_using_onnx_direct", duration=len(waveform_np) / 16000)
        # Raw uncollapsed sections first — kept for exact turn-boundary
        # refinement later; the pipeline itself uses the fused/dip-split copy.
        raw_sections = run_vad_onnx_direct(
            waveform_np,
            self.state.vad_session,
            sample_rate=16000,
            threshold=vad_thresh_val,
            min_speech_duration_ms=vad_min_dur_val,
            merge_close=False,
        )
        self._raw_vad_sections = raw_sections
        speech_ts = merge_vad_sections(raw_sections, max_gap_sec=0.1)

        # Post-process: Split long segments at local energy dips to avoid cross-speaker window contamination
        speech_ts = split_at_energy_dips(speech_ts, waveform_np, sample_rate=16000)

        return speech_ts
    
    def _extract_features(self, waveform_tensor, speech_ts, queue, loop):
        """Extract fbank features from speech segments."""
        loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
            "type": "progress", "step": "Feature Extraction", "completed": 0, "total": len(speech_ts)
        }))
        
        all_fbanks = []
        all_segments_meta = []
        embeddable_indices = []
        
        for i, ts in enumerate(speech_ts):
            start_sample = int(ts['start'] * 16000)
            end_sample = int(ts['end'] * 16000)
            segment_wav = waveform_tensor[:, start_sample:end_sample]
            
            windows, start_times = generate_sliding_windows(segment_wav, 16000, window_sec=2.0, stride_sec=1.2)
            
            for w, rel_start in zip(windows, start_times):
                chunk_duration = w.shape[-1] / 16000
                global_start = ts['start'] + rel_start
                global_end = global_start + chunk_duration
                meta_idx = len(all_segments_meta)
                all_segments_meta.append({"start": global_start, "end": global_end})
                
                if chunk_duration >= self.MIN_EMBED_DURATION:
                    if w.shape[-1] < 1600:
                        w = torch.nn.functional.pad(w, (0, 1600 - w.shape[-1]))
                    all_fbanks.append(extract_fbank(w, 16000))
                    embeddable_indices.append(meta_idx)
            
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "progress", "step": "Feature Extraction", "completed": i + 1, "total": len(speech_ts)
            }))
        
        return all_fbanks, all_segments_meta, embeddable_indices
    
    def _extract_embeddings(self, all_fbanks, queue, loop):
        """Extract speaker embeddings via ONNX with memory cache."""
        loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
            "type": "progress", "step": "Embedding Extraction", "completed": 0, "total": 1
        }))
        
        import hashlib
        
        # Hash each unpadded filterbank to uniquely identify the audio chunk
        fb_hashes = [hashlib.md5(fb.numpy().tobytes()).hexdigest() for fb in all_fbanks]
        
        # Check cache for existing embeddings
        cached_embeddings = {}
        miss_indices = []
        
        for idx, h in enumerate(fb_hashes):
            cached = self.state.embedding_cache.get(h)
            if cached is not None:
                cached_embeddings[idx] = cached
            else:
                miss_indices.append(idx)
        
        # If we have cache misses, compute them via ONNX in batch
        if miss_indices:
            miss_fbanks = [all_fbanks[idx] for idx in miss_indices]
            
            # Apply CMN per sub-segment (critical for ECAPA-TDNN)
            max_len = max(fb.shape[1] for fb in miss_fbanks)
            padded_fbanks = []
            for fb in miss_fbanks:
                if fb.shape[1] < max_len:
                    fb_padded = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
                else:
                    fb_padded = fb
                padded_fbanks.append(fb_padded)
            
            batch = torch.stack(padded_fbanks, dim=0)  # [N_miss, 1, max_len, 80]
            cmn_batch = batch - batch.mean(dim=2, keepdim=True)  # CMN on all at once
            
            # Ensure explicit float32 dtype for iGPU execution
            batch_fbanks = cmn_batch.squeeze(1).numpy().astype(np.float32)  # [N_miss, max_len, 80]
            
            computed_embeddings = []
            batch_size = 32
            for i in range(0, len(batch_fbanks), batch_size):
                audio_input = batch_fbanks[i:i+batch_size].astype(np.float32)
                out = self.state.embedding_session.run(None, {"feats": audio_input})
                computed_embeddings.append(out[0])
            
            computed_embeddings = np.concatenate(computed_embeddings, axis=0)
            
            # Store newly computed embeddings in the global cache
            for local_idx, idx in enumerate(miss_indices):
                emb = computed_embeddings[local_idx]
                h = fb_hashes[idx]
                self.state.embedding_cache.put(h, emb)
                cached_embeddings[idx] = emb
        
        # Reconstruct the full sequence of raw embeddings from cache/computed
        raw_embeddings = np.array([cached_embeddings[idx] for idx in range(len(all_fbanks))])
        
        # Perform L2-normalization
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        raw_embeddings = raw_embeddings / np.maximum(norms, 1e-12)
        
        # Log cache efficiency
        hits = len(all_fbanks) - len(miss_indices)
        log.info("embedding_cache_status", total=len(all_fbanks), hits=hits, misses=len(miss_indices))
        
        loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
            "type": "progress", "step": "Embedding Extraction", "completed": 1, "total": 1
        }))
        
        return raw_embeddings
    
    def _cluster_embeddings(self, raw_embeddings, req, queue, loop):
        """Cluster embeddings into speakers."""
        loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
            "type": "progress", "step": "Clustering", "completed": 0, "total": 1
        }))
        
        n_clusters_val = req.num_speakers
        # Only use threshold if num_speakers is not specified
        if n_clusters_val is not None:
            dist_thresh_val = None  # Force exact number of clusters
        else:
            dist_thresh_val = req.diarization_threshold if req.diarization_threshold is not None else self.settings.diarization_threshold
        
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters_val,
            metric="cosine",
            linkage="average",
            distance_threshold=dist_thresh_val
        )
        
        if len(raw_embeddings) > 1:
            long_labels = clusterer.fit_predict(raw_embeddings)
        else:
            long_labels = np.array([0])
        
        # Cap at 15 clusters
        long_labels = cap_clusters(raw_embeddings, long_labels, max_clusters=15)
        
        # Greedy merge (skip if user forced exact speaker count)
        n_clusters = len(set(int(l) for l in long_labels))
        if n_clusters > 1 and n_clusters_val is None:
            merge_threshold = get("diarization", "merge_threshold", 0.25)
            long_labels, cluster_centroids = greedy_merge_clusters(
                raw_embeddings, long_labels, merge_threshold
            )
        else:
            # Compute centroids without merging
            cluster_centroids = {}
            for cluster_id in set(long_labels):
                mask = (long_labels == cluster_id)
                mean_emb = raw_embeddings[mask].mean(axis=0)
                norm_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
                cluster_centroids[int(cluster_id)] = norm_emb
        
        return long_labels, cluster_centroids
    
    def _assign_labels_to_segments(self, all_segments_meta, embeddable_indices, long_labels):
        """Assign cluster labels to all segments (including short ones)."""
        # Assign cluster labels to embeddable windows
        for idx, label in zip(embeddable_indices, long_labels):
            all_segments_meta[idx]["speaker_raw"] = int(label)
        
        # Assign short windows to the nearest embeddable window by midpoint
        emb_mids = np.array([
            (all_segments_meta[i]["start"] + all_segments_meta[i]["end"]) / 2
            for i in embeddable_indices
        ])
        for i, seg in enumerate(all_segments_meta):
            if "speaker_raw" not in seg:
                mid = (seg["start"] + seg["end"]) / 2
                nearest = int(np.argmin(np.abs(emb_mids - mid)))
                seg["speaker_raw"] = all_segments_meta[embeddable_indices[nearest]]["speaker_raw"]
    
    def _map_to_speakers(self, all_segments_meta, cluster_centroids, req):
        """Map raw cluster IDs to speaker names with overlap awareness.

        After mapping raw cluster IDs → SPEAKER1… labels, merges contiguous
        same-*merge-key* windows into segments.  Overlap windows (flagged by
        ``detect_overlaps``) produce ``OVERLAP`` segments carrying a
        ``"speakers": [A, B]`` list.
        """
        # Map raw cluster int -> SPEAKER1, SPEAKER2, … in first-appearance order
        speaker_map: dict[int, str] = {}
        for seg in sorted(all_segments_meta, key=lambda x: x["start"]):
            raw = seg["speaker_raw"]
            if raw not in speaker_map:
                speaker_map[raw] = f"SPEAKER{len(speaker_map) + 1}"
            seg["speaker"] = speaker_map[raw]

        # Resolve overlap-speaker raw IDs → speaker names
        for seg in all_segments_meta:
            if seg.get("is_overlap", False):
                overs_raw = seg.get("overlap_speakers_raw")
                if overs_raw is not None and len(overs_raw) == 2:
                    seg["overlap_speakers"] = sorted([
                        speaker_map[overs_raw[0]],
                        speaker_map[overs_raw[1]],
                    ])

        max_gap = get("overlap", "max_speaker_gap", 1.0)
        min_dur = get("overlap", "min_duration_sec", 0.3)
        merged_segments = build_overlap_segments(all_segments_meta, max_gap, min_dur)

        return merged_segments, speaker_map
    
    def _refine_boundaries(self, merged_segments, all_segments_meta, embeddable_indices,
                          raw_embeddings, waveform_tensor):
        """Refine speaker boundaries with fine-grained re-embedding."""
        if len(merged_segments) < 2 or not self.state.embedding_session:
            return merged_segments
        
        # Build per-named-speaker centroids
        spk_emb_lists: dict[str, list] = {}
        for k, meta_idx in enumerate(embeddable_indices):
            spk = all_segments_meta[meta_idx].get("speaker")
            if spk:
                spk_emb_lists.setdefault(spk, []).append(raw_embeddings[k])
        
        named_centroids_np: dict[str, np.ndarray] = {}
        for spk, embs in spk_emb_lists.items():
            avg = np.mean(embs, axis=0)
            named_centroids_np[spk] = avg / (np.linalg.norm(avg) + 1e-12)
        
        if named_centroids_np:
            merged_segments = refine_speaker_boundaries(
                merged_segments,
                waveform_tensor,
                self.state.embedding_session,
                named_centroids_np,
                sample_rate=16000,
            )
            log.debug("boundary_refinement_done", n_segments=len(merged_segments))

        return merged_segments

    def _refine_turn_boundaries_exact(self, segments, waveform_tensor,
                                      profiles, known_speakers=None):
        """Exact turn boundaries from uncollapsed VAD sections + voiceprints.

        For every gap between consecutive single-speaker segments: collect the
        raw VAD sections spanning it, embed each one, attribute it to the
        better-matching adjacent speaker (known voiceprint, else that
        speaker's cluster-centroid embedding from profiles), find the
        ownership split, then apply ONE shared cut per gap at the exact start
        of the first section owned by the incoming speaker (gap end when the
        gap belongs entirely to the left speaker). Sections owned by the left
        speaker extend its end; the right speaker's start meets the same cut,
        so the pair stays contiguous — no overlap, no uncovered gap.

        Fallback chain per gap: energy-dip centre → midpoint. Requires raw
        VAD sections (captured in _run_vad) and the embedding session.
        """
        if not get("boundary_refine", "enabled", True):
            return segments
        if len(segments) < 2 or not self.state.embedding_session:
            return segments
        segments = sorted(segments, key=lambda s: s["start"])

        min_gap = float(get("boundary_refine", "min_gap_sec", 0.01))
        min_section = float(get("boundary_refine", "min_section_sec", 0.3))
        dip_ratio = float(get("boundary_refine", "dip_ratio", 0.35))
        min_dip = float(get("boundary_refine", "min_dip_sec", 0.12))

        gaps = []
        for i in range(len(segments) - 1):
            g0 = float(segments[i]["end"])
            g1 = float(segments[i + 1]["start"])
            if g1 - g0 > min_gap:
                gaps.append((i, g0, g1))
        if not gaps:
            return segments

        raw = getattr(self, "_raw_vad_sections", None)
        if raw is None and self.state.vad_session is not None:
            waveform_np = waveform_tensor.squeeze(0).numpy()
            raw = run_vad_onnx_direct(
                waveform_np, self.state.vad_session,
                sample_rate=16000, threshold=self.settings.vad_threshold,
                min_speech_duration_ms=self.settings.vad_min_speech_duration_ms,
                merge_close=False,
            )
        if raw is None:
            raw = []
        raw = sorted(raw, key=lambda s: s["start"])
        # Reference embeddings per speaker: known voiceprint → profiles centroid.
        def _ref(name):
            if known_speakers:
                entry = known_speakers.get(name) or known_speakers.get(str(name).strip("[]"))
                if entry and entry.get("embedding") is not None:
                    v = np.asarray(entry["embedding"], dtype=np.float64).ravel()
                    nrm = float(np.linalg.norm(v))
                    if nrm > 1e-8:
                        return v / nrm
            if profiles:
                entry = profiles.get(name)
                if entry and entry.get("embedding") is not None:
                    v = np.asarray(entry["embedding"], dtype=np.float64).ravel()
                    nrm = float(np.linalg.norm(v))
                    if nrm > 1e-8:
                        return v / nrm
            return None

        # Collect gap sections (only when both refs exist).
        gap_sections: dict[int, list] = {}
        emb_jobs: list[tuple[int, dict]] = []  # (gap_idx, section)
        for gi, (i, g0, g1) in enumerate(gaps):
            ref_l = _ref(segments[i]["speaker"])
            ref_r = _ref(segments[i + 1]["speaker"])
            if ref_l is None or ref_r is None:
                continue
            secs = [
                dict(s) for s in raw
                if s["end"] > g0 and s["start"] < g1
                and (s["end"] - s["start"]) >= min_section
            ]
            if secs:
                gap_sections[gi] = secs
                for s in secs:
                    emb_jobs.append((gi, s))

        # Embed all gap sections in one batched ONNX call (with cache).
        embeddings: dict[int, list] = {}  # id(section dict) -> np vector
        if emb_jobs:
            import hashlib
            waveform_np = waveform_tensor.squeeze(0).numpy()
            fbanks, job_idx = [], []
            for j, (gi, s) in enumerate(emb_jobs):
                a = max(0, int(s["start"] * 16000))
                b = min(len(waveform_np), int(s["end"] * 16000))
                if b - a < int(min_section * 16000):
                    continue
                chunk = torch.from_numpy(waveform_np[a:b]).unsqueeze(0).float()
                fb = extract_fbank(chunk, 16000)  # [1, T, 80]
                fb = fb - fb.mean(dim=1, keepdim=True)  # CMN
                fbanks.append(fb)
                job_idx.append(j)
            if fbanks:
                hashes = [hashlib.md5(fb.numpy().tobytes()).hexdigest() for fb in fbanks]
                cached = {}
                misses = []
                for idx, h in enumerate(hashes):
                    hit = self.state.embedding_cache.get(h)
                    if hit is not None:
                        cached[idx] = hit
                    else:
                        misses.append(idx)
                if misses:
                    miss_fb = [fbanks[idx] for idx in misses]
                    max_len = max(fb.shape[1] for fb in miss_fb)
                    padded = []
                    for fb in miss_fb:
                        if fb.shape[1] < max_len:
                            fb = torch.nn.functional.pad(fb, (0, 0, 0, max_len - fb.shape[1]))
                        padded.append(fb.squeeze(0))
                    batch = torch.stack(padded).numpy().astype(np.float32)
                    input_name = self.state.embedding_session.get_inputs()[0].name
                    out = self.state.embedding_session.run(None, {input_name: batch})[0]
                    for li, idx in enumerate(misses):
                        emb = out[li]
                        self.state.embedding_cache.put(hashes[idx], emb)
                        cached[idx] = emb
                for idx in range(len(fbanks)):
                    emb = np.asarray(cached[idx], dtype=np.float64).ravel()
                    nrm = float(np.linalg.norm(emb))
                    if nrm > 1e-8:
                        emb = emb / nrm
                    _, s = emb_jobs[job_idx[idx]]
                    embeddings[id(s)] = emb

        # Apply one shared cut per gap.
        waveform_np = waveform_tensor.squeeze(0).numpy()
        refined = 0
        for gi, (i, g0, g1) in enumerate(gaps):
            left, right = segments[i], segments[i + 1]
            secs = gap_sections.get(gi, [])
            ref_l = _ref(left["speaker"])
            ref_r = _ref(right["speaker"])
            owners, weights, valid = [], [], []
            if secs and ref_l is not None and ref_r is not None:
                for s in secs:
                    emb = embeddings.get(id(s))
                    if emb is None:
                        continue
                    sa = float(np.dot(emb, ref_l))
                    sb = float(np.dot(emb, ref_r))
                    owners.append(0 if sa >= sb else 1)
                    weights.append(abs(sa - sb) + 1e-3)
                    valid.append(s)
            if valid:
                k = _best_split(owners, weights)
                if k < len(valid):
                    cut = float(valid[k]["start"])
                    cut = min(max(cut, g0), g1)
                else:
                    cut = g1
                n_left = sum(1 for o in owners if o == 0)
                log.info("boundary_refinement", gap_index=i,
                         gap=round(g0, 3), gap_end=round(g1, 3),
                         sections=len(valid), left_owned=n_left,
                         right_owned=len(valid) - n_left, cut=round(cut, 3))
            else:
                cut = _gap_energy_cut(
                    waveform_np, g0, g1,
                    dip_ratio=dip_ratio, min_dip_sec=min_dip,
                )
                log.debug("boundary_fallback_energy", gap_index=i,
                          gap=round(g0, 3), gap_end=round(g1, 3),
                          cut=round(cut, 3))
            left["end"] = float(cut)
            right["start"] = float(cut)
            refined += 1

        if refined:
            log.info("exact_boundary_refinement_done", gaps_closed=refined,
                     total_gaps=len(gaps))
        return segments
    
    def _compute_confidence(self, merged_segments, all_segments_meta, cluster_centroids,
                           raw_embeddings, embeddable_indices, long_labels, req):
        """Compute confidence scores for each segment."""
        from scipy.spatial.distance import cosine
        
        # First compute per-window distances to centroids
        if cluster_centroids and raw_embeddings is not None:
            centroid_arrays = {cid: np.array(centroid) if isinstance(centroid, list) else centroid
                             for cid, centroid in cluster_centroids.items()}
            
            # For each embeddable window, store distances to all centroids
            for idx, (meta_idx, label) in enumerate(zip(embeddable_indices, long_labels)):
                if meta_idx < len(all_segments_meta) and idx < len(raw_embeddings):
                    emb = raw_embeddings[idx]
                    all_dists = {}
                    for cid, centroid in centroid_arrays.items():
                        dist = cosine(emb, centroid)
                        all_dists[cid] = dist
                    
                    # Get sorted distances to measure clarity
                    sorted_dists = sorted(all_dists.values())
                    assigned_dist = all_dists.get(int(label), 1.0)
                    
                    # Clarity: ratio of assigned distance to 2nd closest
                    if len(sorted_dists) > 1:
                        second_closest = sorted_dists[1]
                        clarity = assigned_dist / second_closest if second_closest > 0 else 0.0
                    else:
                        clarity = 0.0  # Only one cluster
                    
                    all_segments_meta[meta_idx]["assigned_dist"] = assigned_dist
                    all_segments_meta[meta_idx]["clarity"] = min(1.0, clarity)
                    all_segments_meta[meta_idx]["all_dists"] = all_dists
        
        # Compute segment-level confidence
        segment_confidences = []
        for seg in merged_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            
            clarities = []
            assigned_dists = []
            speaker_consistency = {}
            
            for meta in all_segments_meta:
                if meta["start"] >= seg_start - 0.1 and meta["end"] <= seg_end + 0.1:
                    if "speaker_raw" in meta:
                        raw_spk = meta["speaker_raw"]
                        speaker_consistency[raw_spk] = speaker_consistency.get(raw_spk, 0) + 1
                        
                        if "clarity" in meta:
                            clarities.append(meta["clarity"])
                        if "assigned_dist" in meta:
                            assigned_dists.append(meta["assigned_dist"])
            
            # Confidence components
            clarity_conf = 0.5
            if clarities:
                avg_clarity = sum(clarities) / len(clarities)
                clarity_conf = max(0, 1 - avg_clarity)
            
            dist_conf = 0.5
            if assigned_dists:
                avg_dist = sum(assigned_dists) / len(assigned_dists)
                dist_conf = max(0, 1 - (avg_dist / 0.5))
            
            consistency_conf = 0.5
            if speaker_consistency:
                max_count = max(speaker_consistency.values())
                total = sum(speaker_consistency.values())
                consistency_conf = max_count / total if total > 0 else 0.5
            
            seg_dur = seg_end - seg_start
            dur_conf = min(1.0, seg_dur / 5.0)
            
            final_conf = (
                clarity_conf * 0.3 +
                dist_conf * 0.3 +
                consistency_conf * 0.25 +
                dur_conf * 0.15
            )
            
            # Boost if known speaker matching
            if req.known_speakers and seg["speaker"] in req.known_speakers:
                final_conf = min(1.0, final_conf + 0.15)
            
            segment_confidences.append(final_conf)
        
        return segment_confidences
    
    def _format_response(self, merged_segments, segment_confidences):
        """Format segments as final JSON response."""
        final_data = []
        for i, seg in enumerate(merged_segments):
            if seg["end"] <= seg["start"]:
                continue
            seg_data = {
                "start": round(float(seg["start"]), 3),
                "end": round(float(seg["end"]), 3),
                "speaker": seg["speaker"],
                "confidence": round(float(segment_confidences[i]), 3) if i < len(segment_confidences) else 0.5
            }
            # Add overlap speakers list if present
            if seg.get("speaker") == "OVERLAP" and seg.get("speakers"):
                seg_data["speakers"] = seg["speakers"]
            # Add alternatives if present
            if "alternatives" in seg and seg["alternatives"]:
                seg_data["alternatives"] = seg["alternatives"]
            final_data.append(seg_data)
        
        # Ensure sorted by start time
        final_data.sort(key=lambda x: x["start"])
        
        return final_data
