"""Speaker module for embedding extraction, matching, audio processing, VAD, and profiling."""
from .embedding import extract_embedding, normalize_embedding, compute_pitch, compute_energy
from .audio import extract_fbank, generate_sliding_windows, refine_speaker_boundaries
from .vad import run_vad_chunked, run_vad_onnx, split_at_energy_dips
from .profiling import profile_speakers, relabel_by_pitch
from .matcher import match_clusters, merge_matched_clusters

__all__ = [
    # embedding
    "extract_embedding",
    "normalize_embedding",
    "compute_pitch",
    "compute_energy",
    # audio
    "extract_fbank",
    "generate_sliding_windows",
    "refine_speaker_boundaries",
    # vad
    "run_vad_chunked",
    "run_vad_onnx",
    "split_at_energy_dips",
    # profiling
    "profile_speakers",
    "relabel_by_pitch",
    # matcher
    "match_clusters",
    "merge_matched_clusters",
]
