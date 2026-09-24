"""Speaker matching - match clusters to known voiceprints."""
import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get


def compute_distance(
    cluster_emb: List[float],
    cluster_pitch: float,
    cluster_energy: float,
    voiceprint: Dict,
    cfg: Dict = None,
    cluster_features: Dict = None,
    is_known_speaker: bool = False
) -> Dict[str, float]:
    """
    Compute distance between a cluster and a voiceprint.
    
    Returns:
    - emb_dist: cosine distance (0-1)
    - pitch_dist: normalized pitch distance (0-1)
    - combined: weighted combination
    - confidence: 0-1 confidence score
    """
    if cfg is None:
        cfg = get("weights")
    
    w_emb = cfg.get("embedding", 0.9) if isinstance(cfg, dict) else 0.9
    w_pitch = cfg.get("pitch", 0.1) if isinstance(cfg, dict) else 0.1
    
    norm = cfg.get("normalization", {}) if isinstance(cfg, dict) else {}
    pitch_per_unit = norm.get("pitch_hz_per_unit", 50)
    conf_max_dist = norm.get("confidence_max_distance", 0.5)
    
    emb_dist = cosine(cluster_emb, voiceprint.get("embedding", []))
    
    known_pitch = voiceprint.get("pitch_hz", 0) or 0
    if cluster_pitch > 0 and known_pitch > 0:
        pitch_dist = abs(cluster_pitch - known_pitch) / pitch_per_unit
    else:
        pitch_dist = 0.0
    
    total_weight = w_emb + w_pitch
    combined = (
        w_emb * emb_dist +
        w_pitch * min(pitch_dist, 1.0)
    ) / (total_weight if total_weight > 0 else 1.0)
    
    if is_known_speaker:
        bias = cfg.get("known_speaker_margin_bias", 0.0) if isinstance(cfg, dict) else 0.0
        if not bias:
            from config import get as cfg_get
            bias = cfg_get("second_pass", "known_speaker_margin_bias", 0.0) if cfg_get else 0.0
        if bias > 0:
            combined = max(0.0, combined - bias)

    confidence = max(0.0, 1.0 - (combined / conf_max_dist))
    
    return {
        "emb_dist": round(float(emb_dist), 4),
        "pitch_dist": round(float(pitch_dist), 4),
        "combined": round(float(combined), 4),
        "confidence": round(float(confidence), 4)
    }


def find_best_match(
    cluster_emb: List[float],
    cluster_pitch: float,
    cluster_energy: float,
    voiceprints: Dict[str, Dict],
    cfg: Dict = None,
    cluster_features: Dict = None
) -> Tuple[Optional[str], float, float, Dict[str, Dict]]:
    """
    Find best matching voiceprint for a cluster.
    
    Returns: (best_name, best_dist, best_conf, all_distances)
    """
    if cfg is None:
        cfg = get("matching")
    
    matches = []
    distances = {}
    
    for name, voiceprint in voiceprints.items():
        if "embedding" not in voiceprint:
            continue
        
        is_known = not (name.startswith("Speaker ") or name.startswith("SPEAKER ") or name == "OVERLAP")
        dist_info = compute_distance(
            cluster_emb, cluster_pitch, cluster_energy, voiceprint,
            cfg=cfg,
            cluster_features=cluster_features,
            is_known_speaker=is_known
        )
        
        matches.append((name, dist_info["combined"], dist_info["confidence"]))
        distances[name] = dist_info
    
    # Sort by distance ascending
    matches.sort(key=lambda x: x[1])
    
    if not matches:
        return None, float('inf'), 0.0, {}
    
    best_name, best_dist, best_conf = matches[0]
    return best_name, best_dist, best_conf, distances


def is_clear_winner(matches: List[Tuple], voiceprints: Dict, cfg: Dict = None) -> bool:
    """
    Check if best match is a clear winner (gap > threshold).
    
    matches: [(name, dist, conf), ...] sorted by distance
    """
    if cfg is None:
        cfg = get("matching")
    
    gap_threshold = cfg.get("clear_winner_gap", 0.02)
    embed_only_thresh = cfg.get("embed_only_threshold", 0.16)
    
    if len(matches) < 2:
        return True
    
    best_dist = matches[0][1]
    second_dist = matches[1][1]
    gap = second_dist - best_dist
    
    if gap >= gap_threshold:
        return True
    
    # Gap is small - check if best has significantly more training data
    first_dur = voiceprints.get(matches[0][0], {}).get("segments_sec") or voiceprints.get(matches[0][0], {}).get("total_speech_sec", 0)
    second_dur = voiceprints.get(matches[1][0], {}).get("segments_sec") or voiceprints.get(matches[1][0], {}).get("total_speech_sec", 0)
    
    if first_dur > second_dur * 2 and best_dist < embed_only_thresh:
        return True
    
    return False


def match_clusters(
    clusters: Dict[str, Dict],  # {cluster_id: {"embedding": [], "pitch_hz": float, "energy_rms": float}}
    voiceprints: Dict[str, Dict],
    cfg: Dict = None,
    all_cluster_features: Dict[str, Dict] = None
) -> Dict[str, Dict]:
    """
    Match all clusters to known voiceprints.
    
    Returns: {cluster_id: {"name": str, "confidence": float, "distances": dict}}
    """
    if cfg is None:
        cfg = get("matching")
    
    if all_cluster_features is None:
        all_cluster_features = {}
    
    accept_threshold = cfg.get("accept_threshold", 0.35)
    embed_only_threshold = cfg.get("embed_only_threshold", 0.16)
    embed_only_accept = cfg.get("embed_only_accept_threshold", 0.22)
    
    results = {}
    
    for cluster_id, cluster_data in clusters.items():
        emb = cluster_data.get("embedding", [])
        pitch = cluster_data.get("pitch_hz", 0) or 0
        energy = cluster_data.get("energy_rms", 0) or 0
        features = all_cluster_features.get(cluster_id, {})
        
        if not emb:
            continue
        
        best_name, best_dist, best_conf, all_distances = find_best_match(
            emb, pitch, energy, voiceprints, cluster_features=features
        )
        
        # Build matches list for clear_winner check
        matches = [(name, d["combined"], d["confidence"]) for name, d in all_distances.items()]
        matches.sort(key=lambda x: x[1])
        
        clear_winner = is_clear_winner(matches, voiceprints, cfg)
        
        is_strong_embed = all_distances.get(best_name, {}).get("emb_dist", 1.0) < embed_only_threshold
        matched = (
            best_name is not None and
            (is_strong_embed or best_dist <= accept_threshold) and
            clear_winner
        )
        
        results[cluster_id] = {
            "name": best_name if matched else None,
            "confidence": best_conf if matched else 0.0,
            "distances": all_distances,
            "best_distance": best_dist,
            "clear_winner": clear_winner,
            "matched": matched
        }
    
    return results


def merge_matched_clusters(
    results: Dict[str, Dict],
    clusters: Dict[str, Dict]
) -> Dict[str, str]:
    """
    Post-process: merge clusters that matched to the same speaker.
    
    Returns: {raw_cluster_id: final_speaker_name}
    """
    speaker_to_clusters = {}
    
    for cluster_id, result in results.items():
        if not result.get("matched") or not result.get("name"):
            continue
        
        name = result["name"]
        if name not in speaker_to_clusters:
            speaker_to_clusters[name] = []
        speaker_to_clusters[name].append(cluster_id)
    
    # For speakers with multiple clusters, keep the best match
    final_map = {}
    unknown_idx = 1
    
    for cluster_id, result in results.items():
        if result.get("matched") and result.get("name"):
            final_map[cluster_id] = result["name"]
        else:
            final_map[cluster_id] = f"SPEAKER{unknown_idx}"
            unknown_idx += 1
    
    return final_map