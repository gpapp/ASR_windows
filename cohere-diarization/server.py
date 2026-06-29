"""
Production-ready ASR transcription server using ONNX Runtime.

Features:
- Secure file upload and validated path access
- Async request handling with thread pool for inference
- Request validation and size limits
- Timeout protection
- KV cache pooling for memory efficiency
- Structured logging
- Optional API key authentication
- Rate limiting
"""

import os
import gc
import time
import asyncio
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, cast

import numpy as np
import librosa
import structlog

from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from settings import get_settings
from model_state import state, executor as global_executor
from model_loader import load_models
from transcriber import transcribe_audio_async, clean_transcript, _compute_mel_spectrogram_fast
from diarization import Diarizer
from streaming import stream_transcribe
from api.schemas import (
    DiarizePathsRequest, TranscribePathsRequest, TranscribeResult,
    TranscribeResponse, HealthResponse, TimedSegment
)
from api.security import verify_api_key, validate_path_security
from api.middleware import apply_middleware
from api.exceptions import register_exception_handlers, PathSecurityError

log = structlog.get_logger()


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global global_executor
    
    settings = get_settings()
          
    global_executor = ThreadPoolExecutor(max_workers=settings.workers)
    
    # Assign to model_state module global
    import model_state
    model_state.executor = global_executor
    
    # Load models
    log.info("starting_server", host=settings.host, port=settings.port)
    load_models(settings)
    
    yield
    
    # Cleanup
    log.info("shutting_down")
    if global_executor:
        global_executor.shutdown(wait=True)

    state.status = "shutdown"


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Transcription Server",
    description="Production ASR transcription service using Cohere model",
    version="1.0.0",
    lifespan=lifespan
)

# Apply middleware and exception handlers
apply_middleware(app)
register_exception_handlers(app)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="online" if state.is_ready else "degraded",
        model_status=state.status
    )


async def _transcribe_chunks(
    audio: np.ndarray,
    language: str,
    settings,
    full_mel_spectrogram: np.ndarray,
    chunk_sec: int = 120
) -> tuple[list[TimedSegment], float, int]:
    """Transcribe audio in chunks, returning timed segments for speaker alignment.

    Larger chunks amortize encoder overhead. Default 120s — works well with q4
    quantized encoder that has GPU memory headroom.
    """
    import gc

    segments: list[TimedSegment] = []
    total_inference = 0.0
    total_tokens = 0

    chunk_samples = chunk_sec * 16000
    chunk_frames = chunk_sec * 100  # 10ms per frame

    for start in range(0, len(audio), chunk_samples):
        chunk_start_sec = start / 16000
        start_frame = start // 160
        end_frame = min(start_frame + chunk_frames, full_mel_spectrogram.shape[1])
        chunk_mel_spectrogram = full_mel_spectrogram[:, start_frame:end_frame, :]

        result = await transcribe_audio_async(
            None,
            language,
            settings.request_timeout,
            mel_spectrogram=chunk_mel_spectrogram
        )

        chunk_end_sec = chunk_start_sec + (end_frame - start_frame) * 0.01
        chunk_text = result.get("text", "")
        if chunk_text:
            segments.append(TimedSegment(
                start=round(chunk_start_sec, 3),
                end=round(chunk_end_sec, 3),
                text=chunk_text
            ))

        total_inference += result.get("inference_time_sec", 0.0)
        total_tokens += result.get("tokens_generated", 0)

        chunk_mel_spectrogram = None
        result = None
        gc.collect()

    return segments, total_inference, total_tokens


async def delayed_shutdown():
    await asyncio.sleep(1.0)
    os._exit(0)


@app.post("/shutdown")
async def shutdown(_: str = Depends(verify_api_key)):
    """Shutdown the server."""
    log.info("shutdown_requested")
    asyncio.create_task(delayed_shutdown())
    return {"status": "shutting down"}


@app.post("/diarize/path")
async def diarize_path_endpoint(
    req: DiarizePathsRequest,
    settings = Depends(get_settings),
    _: str = Depends(verify_api_key)
):
    """
    Streams Pyannote diarization progress as NDJSON, then yields the final segments.
    """
    if not state.vad_session:
        return JSONResponse(
            status_code=400,
            content={"error": "Diarization not enabled or pipeline not loaded"}
        )
    
    if not state.embedding_session:
        return JSONResponse(
            status_code=400,
            content={"error": "Diarization not enabled or pipeline not loaded"}
        )

    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    try:
        resolved = validate_path_security(req.wav_path, settings)
    except PathSecurityError as e:
        return JSONResponse(status_code=403, content={"error": f"Access denied: {e.message}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Start diarization in thread
    diarizer = Diarizer(state, settings)
    task = asyncio.create_task(
        asyncio.to_thread(diarizer.run, req, queue, loop, str(resolved))
    )

    async def event_generator():
        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield msg + "\n"

                # Stop if we hit a terminal message
                if '"type": "result"' in msg or '"error"' in msg:
                    break
        finally:
            # Free diarization memory after streaming completes
            state.embedding_cache.clear()
            gc.collect()

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/transcribe/upload", response_model=TranscribeResponse)
async def transcribe_upload(
    files: list[UploadFile] = File(..., description="Audio files to transcribe"),
    language: str = "en",
    _: str = Depends(verify_api_key)
):
    """
    Transcribe uploaded audio files.
    
    Accepts multiple audio files and returns transcriptions.
    Supports WAV, MP3, MP4, M4A, FLAC, OGG formats.
    """
    settings = get_settings()
    start_time = time.perf_counter()
    
    if len(files) > settings.max_batch_size:
        return JSONResponse(
            status_code=400, 
            content={"error": f"Too many files. Max: {settings.max_batch_size}"}
        )
      
    results = []
    
    for file in files:
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(
                suffix=Path(file.filename or "audio.wav").suffix,
                delete=True
            ) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp.flush()
                
                # Load audio
                load_start = time.perf_counter()
                audio, _ = cast(
                    Tuple[np.ndarray, float],
                    await asyncio.to_thread(
                        lambda: librosa.load(tmp.name, sr=16000, mono=True)
                    )
                )  # type: ignore[assignment]
                audio_load_sec = time.perf_counter() - load_start

                log.debug(
                    "audio_loaded",
                    filename=file.filename,
                    duration_sec=len(audio) / 16000,
                    load_time_sec=audio_load_sec
                )

                audio_float = audio.astype(np.float32)
                audio_float[1:] -= 0.97 * audio_float[:-1]

                mel_start = time.perf_counter()
                full_mel = _compute_mel_spectrogram_fast(audio_float)
                mel_sec = time.perf_counter() - mel_start

                log.debug(
                    "mel_spectrogram_computed",
                    filename=file.filename,
                    mel_time_sec=mel_sec,
                    mel_frames=full_mel.shape[1]
                )

                full_mel_spectrogram = full_mel.T[np.newaxis, :, :].astype(np.float32)

            # Process audio in chunks using the precomputed mel spectrogram
            timed_segments = []
            total_inference = 0
            total_tokens = 0

            timed_segments, total_inference, total_tokens = await _transcribe_chunks(
                audio,
                language,
                settings,
                full_mel_spectrogram
            )

            full_text = " ".join(s.text for s in timed_segments)

            result = {
                "text": clean_transcript(full_text.strip()),
                "audio_duration_sec": len(audio) / 16000,
                "inference_time_sec": total_inference,
                "tokens_generated": total_tokens
            }
            results.append(TranscribeResult(
               text=result["text"],
               audio_duration_sec=result["audio_duration_sec"],
               inference_time_sec=result["inference_time_sec"],
               tokens_generated=result["tokens_generated"],
               segments=timed_segments
            ))
                
        except Exception as e:
            log.error("transcription_failed", filename=file.filename, error=str(e))
            results.append(TranscribeResult(
                text="",
                audio_duration_sec=0.0,
                inference_time_sec=0.0,
                tokens_generated=0,
                error=str(e)
            ))
        
        # Clear per-file memory to prevent accumulation across multiple files
        if "audio" in dir():
            audio = None
        if "full_mel_spectrogram" in dir():
            full_mel_spectrogram = None
        if "full_mel" in dir():
            full_mel = None
        gc.collect()
    
    return TranscribeResponse(
        results=results,
        total_time_sec=time.perf_counter() - start_time
    )


@app.post("/transcribe/paths", response_model=TranscribeResponse)
async def transcribe_paths(
    req: TranscribePathsRequest,
    settings = Depends(get_settings),
    _: str = Depends(verify_api_key)
):
    """
    Transcribe audio files by path.
    
    Paths must be accessible by the server and within allowed directories
    if TRANSCRIBE_ALLOWED_AUDIO_DIR is set.
    """
    start_time = time.perf_counter()
    
    results = []
    
    for path in req.wav_paths:
        try:
            # Validate path
            resolved = validate_path_security(path, settings)
            
            # Load audio
            load_start = time.perf_counter()
            audio, _ = cast(
                Tuple[np.ndarray, float],
                await asyncio.to_thread(
                    lambda: librosa.load(str(resolved), sr=16000, mono=True)
                )
            )  # type: ignore[assignment]
            audio_load_sec = time.perf_counter() - load_start

            log.debug(
                "audio_loaded",
                path=str(resolved),
                duration_sec=len(audio) / 16000,
                load_time_sec=audio_load_sec
            )

            audio_float = audio.astype(np.float32)
            audio_float[1:] -= 0.97 * audio_float[:-1]

            mel_start = time.perf_counter()
            full_mel = _compute_mel_spectrogram_fast(audio_float)
            mel_sec = time.perf_counter() - mel_start

            log.debug(
                "mel_spectrogram_computed",
                path=str(resolved),
                mel_time_sec=mel_sec,
                mel_frames=full_mel.shape[1]
            )

            full_mel_spectrogram = full_mel.T[np.newaxis, :, :].astype(np.float32)
            
            # Process audio in chunks using the precomputed mel spectrogram
            timed_segments = []
            total_inference = 0
            total_tokens = 0

            timed_segments, total_inference, total_tokens = await _transcribe_chunks(
                audio,
                req.language,
                settings,
                full_mel_spectrogram
            )

            full_text = " ".join(s.text for s in timed_segments)

            result = {
                "text": clean_transcript(full_text.strip()),
                "audio_duration_sec": len(audio) / 16000,
                "inference_time_sec": total_inference,
                "tokens_generated": total_tokens
            }

            results.append(TranscribeResult(
                text=result["text"],
                audio_duration_sec=result["audio_duration_sec"],
                inference_time_sec=result["inference_time_sec"],
                tokens_generated=result["tokens_generated"],
                segments=timed_segments
            ))
            
            # Clear per-file memory to prevent accumulation across multiple files
            audio = None
            full_mel_spectrogram = None
            full_mel = None
            gc.collect()
            
        except PathSecurityError as e:
            log.warning("path_security_error", path=path, error=str(e))
            results.append(TranscribeResult(
                text="",
                audio_duration_sec=0.0,
                inference_time_sec=0.0,
                tokens_generated=0,
                error=f"Access denied: {e.message}"
            ))
            
        except Exception as e:
            log.error("transcription_failed", path=path, error=str(e))
            results.append(TranscribeResult(
                text="",
                audio_duration_sec=0.0,
                inference_time_sec=0.0,
                tokens_generated=0,
                error=str(e)
            ))
        
    return TranscribeResponse(
        results=results,
        total_time_sec=time.perf_counter() - start_time
    )


# WebSocket endpoint for streaming
app.add_api_websocket_route("/ws/stream", stream_transcribe)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="debug",
        access_log=True,
    )
