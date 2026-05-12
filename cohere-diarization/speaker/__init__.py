"""Speaker module for embedding extraction and matching."""
from .embedding import extract_embedding, normalize_embedding, compute_pitch, compute_energy

__all__ = [
    "extract_embedding",
    "normalize_embedding",
    "compute_pitch",
    "compute_energy",
]