"""Speaker matching - match clusters to known voiceprints."""
import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get


def _compute_spectral_distance(cluster: Dict, voiceprint: Dict, norm_cfg: Dict) -> float:
    """Compute normalized spectral feature distance."""
    dist = 0.0
    count = 0
    
    # Spectral centroid distance
    cluster_centroid = cluster.get("spectral_centroid", 0)
    vp_centroid = voiceprint.get("spectral_centroid", 0)
    if cluster_centroid > 0 and vp_centroid > 0:
        centroid_per_unit = norm_cfg.get("spectral_centroid_per_unit", 500)
        dist += abs(cluster_centroid - vp_centroid) / centroid_per_unit
        count += 1
    
    # Spectral rolloff distance
    cluster_rolloff = cluster.get("spectral_rolloff", 0)
    vp_rolloff = voiceprint.get("spectral_rolloff", 0)
    if cluster_rolloff > 0 and vp_rolloff > 0:
        rolloff_per_unit = norm_cfg.get("spectral_rolloff_per_unit", 1000)
        dist += abs(cluster_rolloff - vp_rolloff) / rolloff_per_unit
        count += 1
    
    return dist / count if count > 0 else 0.5


def _compute_mfcc_distance(cluster: Dict, voiceprint: Dict, norm_cfg: Dict) -> float:
    """Compute MFCC feature distance (first 13 coefficients)."""
    dist = 0.0
    count = 0
    
    for i in range(13):
        cluster_mean = cluster.get(f"mfcc{i}_mean", 0)
        vp_mean = voiceprint.get(f"mfcc{i}_mean", 0)
        cluster_std = cluster.get(f"mfcc{i}_std", 0)
        vp_std = voiceprint.get(f"mfcc{i}_std", 0)
        
        if cluster_mean != 0 or vp_mean != 0:
            # Distance between means, normalized by typical MFCC range
            mean_per_unit = norm_cfg.get(f"mfcc{i}_mean_per_unit", 10)
            mean_dist = abs(cluster_mean - vp_mean) / mean_per_unit
            
            # Also compare std deviation
            std_per_unit = norm_cfg.get(f"mfcc{i}_std_per_unit", 5)
            std_dist = abs(cluster_std - vp_std) / std_per_unit
            
            dist += (mean_dist + std_dist) / 2
            count += 1
    
    return dist / count if count > 0 else 0.5


def compute_distance(
    cluster_emb: List[float],
    cluster_pitch: float,
    cluster_energy: float,
    voiceprint: Dict,
    cfg: Dict = None,
    cluster_features: Dict = None
) -> Dict[str, float]:
    """
    Compute distance between a cluster and a voiceprint.
    
    Returns dict with:
    - emb_dist: cosine distance of embeddings
    - pitch_dist: normalized pitch distance (0-1)
    - energy_dist: normalized energy distance (0-1)
    - spectral_dist: normalized spectral feature distance (0-1)
    - mfcc_dist: normalized MFCC feature distance (0-1)
    - combined: weighted combination
    - confidence: 0-1 confidence score
    """
    if cfg is None:
        cfg = get("weights")
    
    if cluster_features is None:
        cluster_features = {}
    
    weights = cfg.get("embedding", 0.7), cfg.get("pitch", 0.2), cfg.get("energy", 0.1)
    norm = cfg.get("normalization", {})
    pitch_per_unit = norm.get("pitch_hz_per_unit", 50)
    energy_per_unit = norm.get("energy_rms_per_unit", 0.05)
    conf_max_dist = norm.get("confidence_max_distance", 0.5)
    
    emb_dist = cosine(cluster_emb, voiceprint.get("embedding", []))
    
    known_pitch = voiceprint.get("pitch_hz", 0) or 0
    if cluster_pitch > 0 and known_pitch > 0:
        pitch_dist = abs(cluster_pitch - known_pitch) / pitch_per_unit
    else:
        pitch_dist = 0.5
    
    known_energy = voiceprint.get("energy_rms", 0) or 0
    if cluster_energy > 0 and known_energy > 0:
        energy_dist = abs(cluster_energy - known_energy) / energy_per_unit
    else:
        energy_dist = 0.5
    
    spectral_dist = _compute_spectral_distance(cluster_features, voiceprint, norm)
    mfcc_dist = _compute_mfcc_distance(cluster_features, voiceprint, norm)
    
    # Get extended weights if available
    spectral_weight = cfg.get("spectral", 0.0)
    mfcc_weight = cfg.get("mfcc", 0.0)
    
    # Combined distance - use extended weights if available
    if spectral_weight > 0 or mfcc_weight > 0:
        total_weight = weights[0] + weights[1] + weights[2] + spectral_weight + mfcc_weight
        combined = (
            weights[0] * emb_dist +
            weights[1] * min(pitch_dist, 1.0) +
            weights[2] * min(energy_dist, 1.0) +
            spectral_weight * min(spectral_dist, 1.0) +
            mfcc_weight * min(mfcc_dist, 1.0)
        ) / total_weight
    else:
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
        "spectral_dist": round(spectral_dist, 3),
        "mfcc_dist": round(mfcc_dist, 3),
        "combined": round(float(combined), 3),
        "confidence": round(confidence, 3)
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
        
        dist_info = compute_distance(
            cluster_emb, cluster_pitch, cluster_energy, voiceprint,
            cluster_features=cluster_features
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