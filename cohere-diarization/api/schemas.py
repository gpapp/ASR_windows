"""
Pydantic request and response models for API endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DiarizeResult(BaseModel):
    start: float
    end: float
    speaker: str
    speakers: Optional[list[str]] = None


class DiarizeResponse(BaseModel):
    segments: list[DiarizeResult]
    total_time_sec: float
    error: Optional[str] = None


class DiarizePathsRequest(BaseModel):
    wav_path: str = Field(..., description="Audio file path")
    num_speakers: Optional[int] = Field(None, description="Exact number of speakers (if known)")
    diarization_threshold: Optional[float] = Field(None, description="Distance threshold for clustering (overrides server default)")
    vad_threshold: Optional[float] = Field(None, description="VAD speech probability cutoff (0.0-1.0)")
    vad_min_speech_duration_ms: Optional[int] = Field(None, description="VAD minimum speech chunk length (ms)")
    known_speakers: Optional[dict[str, dict]] = Field(None, description="Map of known speaker names to their profiles")


class TranscribePathsRequest(BaseModel):
    """Request model for path-based transcription."""
    wav_paths: list[str] = Field(..., max_length=10, description="List of audio file paths")
    language: str = Field(default="en", pattern=r"^[a-z]{2}$", description="ISO 639-1 language code")
    
    @field_validator("wav_paths")
    @classmethod
    def validate_paths(cls, v):
        if not v:
            raise ValueError("At least one path required")
        return v


class TimedSegment(BaseModel):
    """A transcription segment with timing info for speaker alignment."""
    start: float
    end: float
    text: str


class TranscribeResult(BaseModel):
    """Single transcription result."""
    text: str
    audio_duration_sec: float
    inference_time_sec: float
    tokens_generated: int
    segments: Optional[list[TimedSegment]] = None
    error: Optional[str] = None


class TranscribeResponse(BaseModel):
    """Transcription response model."""
    results: list[TranscribeResult]
    total_time_sec: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_status: str
    version: str = "1.0.0"
