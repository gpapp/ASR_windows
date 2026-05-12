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
    cfg: Dict = None
) -> Dict[str, float]:
    """
    Compute distance between a cluster and a voiceprint.
    
    Returns dict with:
    - emb_dist: cosine distance of embeddings
    - pitch_dist: normalized pitch distance (0-1)
    - energy_dist: normalized energy distance (0-1)
    - combined: weighted combination
    - confidence: 0-1 confidence score
    """
    if cfg is None:
        cfg = get("weights")
    
    weights = cfg.get("embedding", 0.7), cfg.get("pitch", 0.2), cfg.get("energy", 0.1)
    norm = cfg.get("normalization", {})
    pitch_per_unit = norm.get("pitch_hz_per_unit", 50)
    energy_per_unit = norm.get("energy_rms_per_unit", 0.05)
    conf_max_dist = norm.get("confidence_max_distance", 0.5)
    
    # Embedding distance
    emb_dist = cosine(cluster_emb, voiceprint.get("embedding", []))
    
    # Pitch distance (normalized)
    known_pitch = voiceprint.get("pitch_hz", 0) or 0
    if cluster_pitch > 0 and known_pitch > 0:
        pitch_dist = abs(cluster_pitch - known_pitch) / pitch_per_unit
    else:
        pitch_dist = 0.5
    
    # Energy distance (normalized)
    known_energy = voiceprint.get("energy_rms", 0) or 0
    if cluster_energy > 0 and known_energy > 0:
        energy_dist = abs(cluster_energy - known_energy) / energy_per_unit
    else:
        energy_dist = 0.5
    
    # Combined distance - always use weighted (simpler than dual-path)
    combined = (
        weights[0] * emb_dist +
        weights[1] * min(pitch_dist, 1.0) +
        weights[2] * min(energy_dist, 1.0)
    )
    
    # Confidence (higher is better)
    confidence = max(0, 1 - (combined / conf_max_dist))
    
    return {
        "emb_dist": round(float(emb_dist), 3),
        "pitch_dist": round(pitch_dist, 3),
        "energy_dist": round(energy_dist, 3),
        "combined": round(float(combined), 3),
        "confidence": round(confidence, 3)
    }


def find_best_match(
    cluster_emb: List[float],
    cluster_pitch: float,
    cluster_energy: float,
    voiceprints: Dict[str, Dict],
    cfg: Dict = None
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
        
        dist_info = compute_distance(
            cluster_emb, cluster_pitch, cluster_energy, voiceprint
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
    first_dur = voiceprints.get(matches[0][0], {}).get("total_speech_sec", 0)
    second_dur = voiceprints.get(matches[1][0], {}).get("total_speech_sec", 0)
    
    if first_dur > second_dur * 2 and best_dist < embed_only_thresh:
        return True
    
    return False


def match_clusters(
    clusters: Dict[str, Dict],  # {cluster_id: {"embedding": [], "pitch_hz": float, "energy_rms": float}}
    voiceprints: Dict[str, Dict],
    cfg: Dict = None
) -> Dict[str, Dict]:
    """
    Match all clusters to known voiceprints.
    
    Returns: {cluster_id: {"name": str, "confidence": float, "distances": dict}}
    """
    if cfg is None:
        cfg = get("matching")
    
    accept_threshold = cfg.get("accept_threshold", 0.35)
    embed_only_threshold = cfg.get("embed_only_threshold", 0.16)
    embed_only_accept = cfg.get("embed_only_accept_threshold", 0.22)
    
    results = {}
    
    for cluster_id, cluster_data in clusters.items():
        emb = cluster_data.get("embedding", [])
        pitch = cluster_data.get("pitch_hz", 0) or 0
        energy = cluster_data.get("energy_rms", 0) or 0
        
        if not emb:
            continue
        
        best_name, best_dist, best_conf, all_distances = find_best_match(
            emb, pitch, energy, voiceprints
        )
        
        # Build matches list for clear_winner check
        matches = [(name, d["combined"], d["confidence"]) for name, d in all_distances.items()]
        matches.sort(key=lambda x: x[1])
        
        clear_winner = is_clear_winner(matches, voiceprints, cfg)
        
        # Determine threshold based on embedding distance
        # If embedding alone is good enough, use lower threshold
        if all_distances.get(best_name, {}).get("emb_dist", 1.0) < embed_only_threshold:
            effective_threshold = embed_only_accept
        else:
            effective_threshold = accept_threshold
        
        matched = (
            best_name is not None and
            best_dist <= effective_threshold and
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