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
import time
import asyncio
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import librosa
import structlog

from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from settings import get_settings
from model_state import state, executor as global_executor
from model_loader import load_models
from transcriber import transcribe_audio_async, clean_transcript, TARGET_SAMPLES
from diarization import Diarizer
from streaming import stream_transcribe
from api.schemas import (
    DiarizePathsRequest, TranscribePathsRequest, TranscribeResult,
    TranscribeResponse, HealthResponse
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
       
    # Initialize thread pool
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
        model_status=state.status,
        device=state.device
    )


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
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield msg + "\n"

            # Stop if we hit a terminal message
            if '"type": "result"' in msg or '"error"' in msg:
                break

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
                audio, _ = await asyncio.to_thread(
                    librosa.load, tmp.name, sr=16000, mono=True
                )
                
                # Process audio in 30s chunks (model expects fixed-length input)
                full_text = ""
                total_duration = 0
                total_inference = 0
                total_tokens = 0
                
                chunk_samples = TARGET_SAMPLES  # 30s at 16kHz
                for start in range(0, len(audio), chunk_samples):
                    chunk = audio[start:start + chunk_samples]
                    
                    # Pad short final chunk to exactly 30s
                    if len(chunk) < chunk_samples:
                        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
                    
                    result = await transcribe_audio_async(
                        chunk, 
                        language, 
                        settings.request_timeout
                    )
                    
                    full_text += result["text"] + " "
                    total_duration += result["audio_duration_sec"]
                    total_inference += result["inference_time_sec"]
                    total_tokens += result["tokens_generated"]
                
                result = {
                    "text": clean_transcript(full_text.strip()),
                    "audio_duration_sec": len(audio) / 16000,
                    "inference_time_sec": total_inference,
                    "tokens_generated": total_tokens
                }
                
                results.append(TranscribeResult(
                    text=result["text"],
                    audio_duration_sec=str(result["audio_duration_sec"]),
                    inference_time_sec=str(result["inference_time_sec"]),
                    tokens_generated=str(result["tokens_generated"])
                ))
                
        except Exception as e:
            log.error("transcription_failed", filename=file.filename, error=str(e))
            results.append(TranscribeResult(
                text="",
                audio_duration_sec="0",
                inference_time_sec="0",
                tokens_generated="0",
                error=str(e)
            ))
    
    return TranscribeResponse(
        results=results,
        total_time_sec=str(time.perf_counter() - start_time)
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
            audio, _ = await asyncio.to_thread(
                librosa.load, str(resolved), sr=16000, mono=True
            )
            
            # Process audio in 30s chunks
            full_text = ""
            total_duration = 0
            total_inference = 0
            total_tokens = 0
            
            chunk_samples = TARGET_SAMPLES  # 30s at 16kHz
            for start in range(0, len(audio), chunk_samples):
                chunk = audio[start:start + chunk_samples]
                
                # Pad short final chunk to exactly 30s
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
                
                result = await transcribe_audio_async(
                    chunk, 
                    req.language, 
                    settings.request_timeout
                )
                
                full_text += result["text"] + " "
                total_duration += result["audio_duration_sec"]
                total_inference += result["inference_time_sec"]
                total_tokens += result["tokens_generated"]
            
            result = {
                "text": clean_transcript(full_text.strip()),
                "audio_duration_sec": len(audio) / 16000,
                "inference_time_sec": total_inference,
                "tokens_generated": total_tokens
            }
            
            results.append(TranscribeResult(
                text=result["text"],
                audio_duration_sec=str(result["audio_duration_sec"]),
                inference_time_sec=str(result["inference_time_sec"]),
                tokens_generated=str(result["tokens_generated"])
            ))
            
        except PathSecurityError as e:
            log.warning("path_security_error", path=path, error=str(e))
            results.append(TranscribeResult(
                text="",
                audio_duration_sec="0",
                inference_time_sec="0",
                tokens_generated="0",
                error=f"Access denied: {e.message}"
            ))
            
        except Exception as e:
            log.error("transcription_failed", path=path, error=str(e))
            results.append(TranscribeResult(
                text="",
                audio_duration_sec="0",
                inference_time_sec="0",
                tokens_generated="0",
                error=str(e)
            ))
        
    return TranscribeResponse(
        results=results,
        total_time_sec=str(time.perf_counter() - start_time)
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
