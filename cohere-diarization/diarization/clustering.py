"""
Clustering operations: greedy merge, speaker matching, cluster capping.
"""

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
import structlog

from config import get, is_debug
from speaker.matcher import match_clusters
from .segment_ops import merge_profiles

log = structlog.get_logger()


def cap_clusters(raw_embeddings: np.ndarray, long_labels: np.ndarray, max_clusters: int = 15) -> np.ndarray:
    """Cap number of clusters to prevent over-segmentation.
    
    Args:
        raw_embeddings: Embeddings array [N, D]
        long_labels: Initial cluster labels
        max_clusters: Maximum allowed clusters
        
    Returns:
        Updated cluster labels
    """
    n_clusters = len(set(int(l) for l in long_labels))
    
    if n_clusters > max_clusters:
        log.warning("cluster_cap_exceeded", initial=n_clusters, capped=max_clusters)
        # Re-cluster with exact number
        clusterer = AgglomerativeClustering(
            n_clusters=max_clusters,
            metric="cosine",
            linkage="average"
        )
        if len(raw_embeddings) > 1:
            long_labels = clusterer.fit_predict(raw_embeddings)
    
    return long_labels


def greedy_merge_clusters(
    raw_embeddings: np.ndarray,
    long_labels: np.ndarray,
    merge_threshold: float = 0.25
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Greedily merge clusters below distance threshold.
    
    Args:
        raw_embeddings: Embeddings array [N, D]
        long_labels: Cluster labels
        merge_threshold: Cosine distance threshold for merging
        
    Returns:
        (updated_labels, cluster_centroids)
    """
    cluster_ids = sorted(set(int(l) for l in long_labels))
    cluster_avgs = {}
    for cid in cluster_ids:
        mask = long_labels == cid
        cluster_avgs[cid] = np.mean(raw_embeddings[mask], axis=0)
    
    changed = True
    while changed:
        changed = False
        ids = sorted(cluster_avgs.keys())
        for i_idx in range(len(ids)):
            for j_idx in range(i_idx + 1, len(ids)):
                id_i, id_j = ids[i_idx], ids[j_idx]
                if id_i not in cluster_avgs:
                    continue
                if id_j not in cluster_avgs:  # Already merged
                    continue
                dist = cosine_distances(
                    [cluster_avgs[id_i]], [cluster_avgs[id_j]]
                )[0][0]
                if dist < merge_threshold:
                    if is_debug():
                        log.debug("merging_clusters", from_cluster=id_j, to_cluster=id_i, distance=dist)
                    # Merge j into i
                    long_labels[long_labels == id_j] = id_i
                    # Update average embedding
                    mask_i = long_labels == id_i
                    cluster_avgs[id_i] = np.mean(raw_embeddings[mask_i], axis=0)
                    del cluster_avgs[id_j]
                    changed = True
                    break
            if changed:
                break
    
    # Compute normalized centroid vector for each remaining cluster
    cluster_centroids = {}
    for cluster_id in set(long_labels):
        mask = (long_labels == cluster_id)
        mean_emb = raw_embeddings[mask].mean(axis=0)
        norm_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
        cluster_centroids[int(cluster_id)] = norm_emb
    
    return long_labels, cluster_centroids


def match_known_speakers_simple(
    cluster_centroids: dict[int, np.ndarray],
    known_speakers: dict[str, dict],
    match_thresh: float = 0.03,
    close_match_thresh: float = 0.1,
) -> tuple[dict[int, str], dict]:
    """Simple embedding-only matching for streaming context.
    
    Used by streaming pipeline where we don't have full profiles yet.
    
    Args:
        cluster_centroids: Raw cluster ID -> centroid
        known_speakers: Known speaker profiles
        match_thresh: Distance threshold for accepting a match
        close_match_thresh: Threshold for merging clusters that match same speaker
        
    Returns:
        (raw_to_name_map, merged_cluster_centroids)
    """
    speaker_map: dict[int, str] = {}
    unknown_idx = 1
    
    # First pass: match all clusters
    cluster_matches = {}
    for raw_spk, centroid in cluster_centroids.items():
        best_match = None
        best_dist = float('inf')
        
        for known_name, known_prof in known_speakers.items():
            if "embedding" not in known_prof:
                continue
            emb_dist = cosine(centroid, known_prof["embedding"])
            
            if emb_dist < best_dist:
                best_dist = emb_dist
                best_match = known_name
        
        cluster_matches[raw_spk] = {"name": best_match, "dist": best_dist, "centroid": centroid}
    
    # Second pass: detect closely matching clusters to same known speaker
    merged_clusters = set()
    for raw_spk1, match1 in cluster_matches.items():
        if raw_spk1 in merged_clusters:
            continue
        if not match1["name"] or match1["dist"] > close_match_thresh:
            continue
        
        for raw_spk2, match2 in cluster_matches.items():
            if raw_spk2 == raw_spk1 or raw_spk2 in merged_clusters:
                continue
            if not match2["name"] or match2["dist"] > close_match_thresh:
                continue
            
            # Both closely match known speakers - check if they're the same
            if match1["name"] == match2["name"]:
                # Same known speaker, very close matches - check distance between clusters
                dist_between = cosine(match1["centroid"], match2["centroid"])
                if dist_between < 0.2:
                    if is_debug():
                        log.debug("merging_similar_clusters", cluster1=raw_spk1, cluster2=raw_spk2, matched_to=match1["name"])
                    merged_clusters.add(raw_spk2)
    
    # Assign final names
    for raw_spk, match_info in cluster_matches.items():
        if raw_spk in merged_clusters:
            continue
        
        best_match = match_info["name"]
        best_dist = match_info["dist"]
        
        if best_match and best_dist <= match_thresh:
            speaker_map[raw_spk] = best_match
        else:
            speaker_map[raw_spk] = f"SPEAKER{unknown_idx}"
            unknown_idx += 1
    
    return speaker_map, merged_clusters


def match_known_speakers_full(
    merged_segments: list[dict],
    all_segments_meta: list[dict],
    embeddable_indices: list[int],
    raw_embeddings: np.ndarray,
    cluster_centroids: dict[int, np.ndarray],
    profiles: dict,
    known_speakers: dict[str, dict],
    req,
) -> tuple[list[dict], dict]:
    """Full voiceprint matching with profiles, pitch, and energy.
    
    Used by batch diarization endpoint where we have complete profiles.
    
    Args:
        merged_segments: Segments after initial clustering
        all_segments_meta: Window-level metadata
        embeddable_indices: Indices of embeddable windows
        raw_embeddings: Window embeddings
        cluster_centroids: Raw cluster centroids
        profiles: Speaker profiles (pitch, energy, etc.)
        known_speakers: Known speaker voiceprints
        req: Request object for threshold overrides
        
    Returns:
        (updated_segments, updated_profiles)
    """
    # Compute centroid for each final speaker
    speaker_centroids = {}
    for seg in merged_segments:
        spk = seg["speaker"]
        if spk not in speaker_centroids:
            speaker_centroids[spk] = []
        
        # Find embedding for this segment's time range
        seg_start = seg["start"]
        seg_end = seg["end"]
        for i, meta in enumerate(all_segments_meta):
            if meta["start"] >= seg_start - 0.1 and meta["end"] <= seg_end + 0.1:
                if "speaker_raw" in meta:
                    raw_id = meta["speaker_raw"]
                    if raw_id in cluster_centroids:
                        speaker_centroids[spk].append(np.array(cluster_centroids[raw_id]))
    
    # Match each speaker to known voiceprints
    match_thresh = get("matching", "accept_threshold", 0.35)
    
    if is_debug():
        log.debug("speaker_matching_start", cluster_speakers=list(speaker_centroids.keys()), 
                  known_speakers=list(known_speakers.keys()) if known_speakers else None)
    
    # Build clusters dict for matcher module
    clusters = {}
    all_cluster_features = {}
    for spk, emb_list in speaker_centroids.items():
        if not emb_list:
            continue
        
        centroid = np.mean(emb_list, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        # Get cluster features from profiles
        prof = profiles.get(spk, {})
        clusters[spk] = {
            "embedding": centroid.tolist(),
            "pitch_hz": prof.get("pitch_hz", 0) or 0,
            "energy_rms": prof.get("energy_rms", 0) or 0,
        }
        
        # Collect spectral and MFCC features
        features = {"spectral_centroid": prof.get("spectral_centroid", 0) or 0,
                    "spectral_rolloff": prof.get("spectral_rolloff", 0) or 0}
        for i in range(13):
            features[f"mfcc{i}_mean"] = prof.get(f"mfcc{i}_mean", 0) or 0
            features[f"mfcc{i}_std"] = prof.get(f"mfcc{i}_std", 0) or 0
        all_cluster_features[spk] = features
    
    # Use matcher module for distance computation and matching
    match_results = match_clusters(clusters, known_speakers, 
                                   all_cluster_features=all_cluster_features)
    
    # Build all_matches for post-processing
    all_matches = {}  # spk -> [(name, distance, confidence), ...]
    debug_matches = {}
    
    for spk, result in match_results.items():
        matches_list = []
        debug_distances = {}
        for name, dist_info in result.get("distances", {}).items():
            conf = dist_info.get("confidence", 0)
            matches_list.append((name, dist_info.get("combined", 1.0), conf))
            debug_distances[name] = dist_info
        matches_list.sort(key=lambda x: x[1])
        all_matches[spk] = matches_list
        debug_matches[spk] = debug_distances
        
        if is_debug() and matches_list:
            cluster_pitch = clusters[spk].get("pitch_hz", 0)
            cluster_energy = clusters[spk].get("energy_rms", 0)
            log.debug("cluster_match_distances", cluster=spk, pitch=cluster_pitch, 
                     energy=cluster_energy, distances=debug_distances)
    
    # Apply best match if distance below threshold, store alternatives
    matching_cfg = get("matching", default={})
    gap_threshold = matching_cfg.get("clear_winner_gap", 0.02)
    embed_only_thresh = matching_cfg.get("embed_only_threshold", 0.16)
    
    for spk, emb_list in speaker_centroids.items():
        if not emb_list or spk not in all_matches:
            continue
        
        matches = all_matches[spk]
        if not matches:
            continue
        
        best_match, best_dist, best_conf = matches[0]
        
        # Check if it's a clear winner
        clear_winner = True
        
        if len(matches) > 1:
            second_dist = matches[1][1]
            gap = second_dist - best_dist
            if gap < gap_threshold:
                # When gap is small, prefer speaker with more training data
                first_dur = known_speakers.get(matches[0][0], {}).get("segments_sec") or known_speakers.get(matches[0][0], {}).get("total_speech_sec", 0)
                second_dur = known_speakers.get(matches[1][0], {}).get("segments_sec") or known_speakers.get(matches[1][0], {}).get("total_speech_sec", 0)
                
                # If best has significantly more data, use it despite small gap
                if first_dur > second_dur * 2 and best_dist < embed_only_thresh:
                    clear_winner = True
                    if is_debug():
                        log.debug("preferring_more_training_data", winner=matches[0][0], 
                                 first_dur=first_dur, second_dur=second_dur)
                else:
                    clear_winner = False
        
        if is_debug():
            log.debug("match_decision", cluster=spk, best_match=best_match, best_dist=best_dist, 
                     clear_winner=clear_winner, accepted=best_dist <= match_thresh and clear_winner)
        
        # Store alternatives in each segment
        alternatives = []
        conf_thresh = matching_cfg.get("confidence_threshold", 0.3)
        for name, dist, conf in matches[1:4]:  # Top 3 alternatives
            if conf >= conf_thresh:
                alternatives.append({"speaker": name, "confidence": round(conf, 2)})
        
        if best_match and best_dist <= match_thresh and clear_winner:
            # Update all segments with this speaker
            for seg in merged_segments:
                if seg["speaker"] == spk:
                    seg["speaker"] = best_match
                    seg["alternatives"] = alternatives
                    # Store debug for logging
                    seg["debug"] = {"from_cluster": spk, "confidence": best_conf, "distances": debug_matches[spk]}
            # Update profiles
            profiles[best_match] = profiles.pop(spk)
            profiles[best_match]["matched_from"] = spk
            profiles[best_match]["match_confidence"] = best_conf
    
    # Post-match merging: combine clusters that both matched to the same speaker
    speaker_to_clusters = {}
    for spk, matches_list in all_matches.items():
        if not matches_list:
            continue
        best_match, best_dist, _ = matches_list[0]
        if best_match and best_dist <= match_thresh:
            if best_match not in speaker_to_clusters:
                speaker_to_clusters[best_match] = []
            speaker_to_clusters[best_match].append(spk)
    
    # Merge clusters with same speaker match
    for speaker, cluster_list in speaker_to_clusters.items():
        if len(cluster_list) > 1:
            if is_debug():
                log.debug("merging_matched_clusters", clusters=cluster_list, matched_to=speaker)
            primary = cluster_list[0]
            for extra in cluster_list[1:]:
                # Update all segments from extra cluster to primary speaker
                for seg in merged_segments:
                    if seg["speaker"] == extra:
                        seg["speaker"] = speaker
                merge_profiles(profiles, primary, extra)
    
    # Additional merge: clusters with similar distance profiles
    cluster_ids = list(all_matches.keys())
    merged_already = set()
    for i in range(len(cluster_ids)):
        if cluster_ids[i] in merged_already:
            continue
        for j in range(i + 1, len(cluster_ids)):
            if cluster_ids[j] in merged_already:
                continue
            # Compare distance vectors
            matches_i = all_matches[cluster_ids[i]]
            matches_j = all_matches[cluster_ids[j]]
            if not matches_i or not matches_j:
                continue
            # Build distance vectors for same speakers
            dist_vec_i = {m[0]: m[1] for m in matches_i}
            dist_vec_j = {m[0]: m[1] for m in matches_j}
            common_speakers = set(dist_vec_i.keys()) & set(dist_vec_j.keys())
            if len(common_speakers) >= 3:
                # Check if distances are similar (within 0.05)
                max_diff = max(abs(dist_vec_i[s] - dist_vec_j[s]) for s in common_speakers)
                if max_diff < 0.05:
                    if is_debug():
                        log.debug("merging_similar_profiles", cluster_i=cluster_ids[i], 
                                 cluster_j=cluster_ids[j], max_diff=max_diff)
                    # Merge j into i
                    target_spk = cluster_ids[i]
                    for seg in merged_segments:
                        if seg["speaker"] == cluster_ids[j]:
                            seg["speaker"] = target_spk
                    merge_profiles(profiles, target_spk, cluster_ids[j])
                    merged_already.add(cluster_ids[j])
    
    return merged_segments, profiles


def collapse_unknown_speakers_second_pass(
    segments: list,
    audio_np: np.ndarray,
    sample_rate: int,
    known_speakers: dict,
    profiles: dict,
    embedding_session=None,
    cfg: dict = None,
) -> tuple:
    """Second-pass re-identification and consolidation of unknown speakers.

    1. Gathers all audio across the entire call for each speaker.
    2. Computes aggregated high-SNR embeddings.
    3. Blends in-call references with DB voiceprints for confirmed known speakers.
    4. Matches remaining unknown 'Speaker N' clusters against known targets with larger fit margin.
    5. Cross-matches and merges duplicate unknown speakers.
    6. Fuses adjacent same-speaker segments.
    """
    from speaker.embedding import extract_embedding
    from speaker.matcher import find_best_match, is_clear_winner
    from diarization.segment_ops import merge_profiles, collapse_same_speaker_segments
    from config import get as cfg_get

    if not segments or audio_np is None or len(audio_np) == 0:
        return segments, profiles

    cfg = cfg or {}
    sec_cfg = cfg.get("second_pass", {})
    if not sec_cfg and cfg_get:
        sec_cfg = cfg_get("second_pass", {})

    if sec_cfg and not sec_cfg.get("enabled", True):
        return segments, profiles

    accept_thresh = sec_cfg.get("accept_threshold", 0.38)
    unknown_merge_thresh = sec_cfg.get("unknown_merge_threshold", 0.25)
    min_speaker_dur = sec_cfg.get("min_speaker_duration_sec", 1.0)

    all_speakers = set(seg.get("speaker") for seg in segments if seg.get("speaker"))
    unknown_speakers = [
        s for s in all_speakers
        if (s.startswith("Speaker ") or s.startswith("SPEAKER ")) and s != "OVERLAP"
    ]
    if not unknown_speakers:
        return segments, profiles

    # 1. Gather audio and compute aggregated embeddings for each speaker
    spk_embeddings: dict = {}
    for spk in all_speakers:
        if spk == "OVERLAP":
            continue
        audio_chunks = []
        for seg in segments:
            if seg.get("speaker") != spk:
                continue
            s_sec = seg.get("start", 0.0)
            e_sec = seg.get("end", 0.0)
            if e_sec - s_sec < 0.2:
                continue
            s_idx = max(0, int(s_sec * sample_rate))
            e_idx = min(len(audio_np), int(e_sec * sample_rate))
            if e_idx > s_idx:
                audio_chunks.append(audio_np[s_idx:e_idx])

        if not audio_chunks:
            continue

        spk_audio = np.concatenate(audio_chunks)
        dur = len(spk_audio) / sample_rate
        if dur >= min_speaker_dur:
            try:
                emb = extract_embedding(spk_audio, sample_rate, embedding_session=embedding_session)
                if emb is not None and len(emb) > 0:
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        spk_embeddings[spk] = emb / norm
            except Exception as e:
                logger.warning("Second pass embedding failed for %s: %s", spk, e)

    # 2. Build reference voiceprints (blending DB voiceprints + in-call centroids)
    reference_targets = {}
    if known_speakers:
        for name, vp in known_speakers.items():
            ref_vp = dict(vp)
            db_emb = np.array(vp.get("embedding", []), dtype=np.float32)
            if name in spk_embeddings and len(db_emb) > 0:
                in_call_emb = spk_embeddings[name]
                blended = 0.5 * db_emb + 0.5 * in_call_emb
                norm = np.linalg.norm(blended)
                if norm > 0:
                    blended = blended / norm
                ref_vp["embedding"] = blended.tolist()
            reference_targets[name] = ref_vp

    # 3. Match unknown clusters against reference targets
    resolved_unknowns = set()
    if reference_targets:
        for spk in list(unknown_speakers):
            if spk not in spk_embeddings:
                continue
            emb = spk_embeddings[spk]
            prof = profiles.get(spk, {})
            pitch = prof.get("pitch_hz", 0.0) or 0.0
            energy = prof.get("energy_rms", 0.0) or 0.0
            features = {
                k: v for k, v in prof.items()
                if k.startswith("mfcc") or k.startswith("spectral")
            }

            best_name, best_dist, second_dist, all_dists = find_best_match(
                emb.tolist(), pitch, energy, reference_targets, cfg, features
            )

            if not best_name or best_dist > accept_thresh:
                continue

            matches = [(n, d["combined"], d["confidence"]) for n, d in all_dists.items()]
            matches.sort(key=lambda x: x[1])
            if not is_clear_winner(matches, reference_targets, cfg):
                continue

            # Remap segments
            for seg in segments:
                if seg.get("speaker") == spk:
                    seg["speaker"] = best_name
            merge_profiles(profiles, best_name, spk)
            resolved_unknowns.add(spk)
            logger.info(
                "Second pass: collapsed unknown %s -> %s (dist=%.3f, conf=%.2f)",
                spk, best_name, best_dist, all_dists[best_name].get("confidence", 0.0)
            )

    # 4. Cross-match and merge duplicate unknown speakers
    remaining_unknowns = [s for s in unknown_speakers if s not in resolved_unknowns]
    merged_unknowns = set()
    for i in range(len(remaining_unknowns)):
        u1 = remaining_unknowns[i]
        if u1 in merged_unknowns or u1 not in spk_embeddings:
            continue
        for j in range(i + 1, len(remaining_unknowns)):
            u2 = remaining_unknowns[j]
            if u2 in merged_unknowns or u2 not in spk_embeddings:
                continue

            cos_dist = 1.0 - float(np.dot(spk_embeddings[u1], spk_embeddings[u2]))
            if cos_dist < unknown_merge_thresh:
                dur1 = profiles.get(u1, {}).get("total_speech_sec", 0.0)
                dur2 = profiles.get(u2, {}).get("total_speech_sec", 0.0)
                primary, secondary = (u1, u2) if dur1 >= dur2 else (u2, u1)

                for seg in segments:
                    if seg.get("speaker") == secondary:
                        seg["speaker"] = primary
                merge_profiles(profiles, primary, secondary)
                merged_unknowns.add(secondary)
                logger.info(
                    "Second pass: merged duplicate unknowns %s -> %s (dist=%.3f)",
                    secondary, primary, cos_dist
                )

    segments = collapse_same_speaker_segments(segments, max_gap=0.5)
    return segments, profiles
