"""
Segment post-processing operations.
"""

from config import is_debug


def collapse_same_speaker_segments(segments: list, max_gap: float = 0.0) -> list:
    """Collapse adjacent segments with the same speaker.
    
    Args:
        segments: List of segment dicts with 'speaker', 'start', 'end' keys
        max_gap: Max time gap between segments to still consider them adjacent (default 0 = contiguous)
    """
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x["start"])
    collapsed = [segments[0].copy()]
    for seg in segments[1:]:
        prev = collapsed[-1]
        if seg["speaker"] == prev["speaker"] and seg["start"] - prev["end"] <= max_gap:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            collapsed.append(seg.copy())
    return collapsed


def absorb_islands(segments: list, min_island_dur: float = 1.0) -> list:
    """Absorb short segments sandwiched between the same speaker on both sides.

    OVERLAP segments are never absorbed — they represent genuine multi-speaker
    activity even when brief.
    """
    changed = True
    while changed:
        changed = False
        for i in range(1, len(segments) - 1):
            seg = segments[i]
            if seg.get("speaker") == "OVERLAP":
                continue
            dur = seg["end"] - seg["start"]
            prev_spk = segments[i - 1]["speaker"]
            next_spk = segments[i + 1]["speaker"]
            if dur < min_island_dur and prev_spk == next_spk and seg["speaker"] != prev_spk:
                seg["speaker"] = prev_spk
                changed = True
        if changed:
            segments = collapse_same_speaker_segments(segments, max_gap=1.0)
    return segments


def eliminate_ghost_speakers(
    segments: list,
    profiles: dict | None = None,
    ghost_threshold_sec: float = 10.0,
) -> list:
    """Reassign speakers whose total speech is below the ghost threshold.

    OVERLAP segments are never reassigned. Their duration is counted toward
    both speakers in the overlap pair when computing per-speaker totals,
    preventing a speaker that appears mainly in overlaps from being incorrectly
    classified as a ghost.
    """
    speaker_total: dict[str, float] = {}
    for seg in segments:
        spk = seg["speaker"]
        dur = seg["end"] - seg["start"]
        if spk == "OVERLAP":
            for s in seg.get("speakers", []):
                speaker_total[s] = speaker_total.get(s, 0.0) + dur
        else:
            speaker_total[spk] = speaker_total.get(spk, 0.0) + dur

    ghost_speakers = {spk for spk, total in speaker_total.items() if total < ghost_threshold_sec}
    if not ghost_speakers:
        return segments

    if is_debug():
        print(f"GHOST SPEAKERS (< {ghost_threshold_sec}s total): {ghost_speakers}")

    for seg in segments:
        if seg["speaker"] not in ghost_speakers:
            continue

        reassigned = False
        for alt in seg.get("alternatives", []):
            alt_spk = alt["speaker"]
            if alt_spk not in ghost_speakers:
                if is_debug():
                    print(f"  GHOST REASSIGN {seg['start']:.1f}-{seg['end']:.1f}: "
                          f"{seg['speaker']} -> {alt_spk} (alt conf={alt['confidence']:.2f})")
                seg["speaker"] = alt_spk
                reassigned = True
                break

        if not reassigned:
            seg_mid = (seg["start"] + seg["end"]) / 2
            best_neighbor, best_dist_t = None, float("inf")
            for other in segments:
                if other is seg or other["speaker"] in ghost_speakers:
                    continue
                d = abs((other["start"] + other["end"]) / 2 - seg_mid)
                if d < best_dist_t:
                    best_dist_t = d
                    best_neighbor = other["speaker"]
            if best_neighbor:
                if is_debug():
                    print(f"  GHOST NEIGHBOUR {seg['start']:.1f}-{seg['end']:.1f}: "
                          f"{seg['speaker']} -> {best_neighbor}")
                seg["speaker"] = best_neighbor

    segments = collapse_same_speaker_segments(segments, max_gap=1.0)

    if profiles is not None:
        for ghost in ghost_speakers:
            profiles.pop(ghost, None)

    return segments


def merge_profiles(profiles: dict, target: str, source: str):
    """Merge source profile into target profile, deleting source."""
    if source not in profiles:
        return
    if target not in profiles:
        profiles[target] = profiles.pop(source, {})
        return
    profiles[target]["speech_sec"] = profiles[target].get("speech_sec", 0) + profiles[source].get("speech_sec", 0)
    profiles[target]["segment_count"] = profiles[target].get("segment_count", 0) + profiles[source].get("segment_count", 0)
    del profiles[source]
