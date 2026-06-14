"""
Production-ready ASR transcription server using Granite Speech + ECAPA-TDNN.

Features:
- Secure file upload and validated path access
- Async request handling with thread pool for inference
- Request validation and size limits
- Timeout protection
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
from typing import Tuple, cast

import numpy as np
import librosa
import structlog

from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from settings import get_settings
from model_state import state, executor as global_executor
from model_loader import load_models
from transcriber import transcribe_audio_async, transcribe_saa_async
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_executor

    settings = get_settings()

    global_executor = ThreadPoolExecutor(max_workers=settings.workers)

    import model_state
    model_state.executor = global_executor

    log.info("starting_server", host=settings.host, port=settings.port)
    load_models(settings)

    yield

    log.info("shutting_down")
    if global_executor:
        global_executor.shutdown(wait=True)

    state.status = "shutdown"


app = FastAPI(
    title="Granite Transcription Server",
    description="ASR transcription service using Granite Speech + ECAPA-TDNN voiceprint matching",
    version="1.0.0",
    lifespan=lifespan
)

apply_middleware(app)
register_exception_handlers(app)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="online" if state.is_ready else "degraded",
        model_status=state.status
    )


async def delayed_shutdown():
    await asyncio.sleep(1.0)
    os._exit(0)


@app.post("/shutdown")
async def shutdown(_: str = Depends(verify_api_key)):
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
    Run hybrid diarization: Granite SAA for speaker turns + ECAPA-TDNN voiceprint matching.
    Streams progress as NDJSON, then yields final segments.
    """
    if not state.vad_session:
        return JSONResponse(
            status_code=400,
            content={"error": "VAD model not loaded"}
        )

    if not state.embedding_session:
        return JSONResponse(
            status_code=400,
            content={"error": "Embedding model not loaded"}
        )

    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    try:
        resolved = validate_path_security(req.wav_path, settings)
    except PathSecurityError as e:
        return JSONResponse(status_code=403, content={"error": f"Access denied: {e.message}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

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

            if '"type": "result"' in msg or '"error"' in msg:
                break

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/transcribe/upload", response_model=TranscribeResponse)
async def transcribe_upload(
    files: list[UploadFile] = File(..., description="Audio files to transcribe"),
    _: str = Depends(verify_api_key)
):
    """
    Transcribe uploaded audio files using plain Granite ASR (no speaker attribution).
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
            with tempfile.NamedTemporaryFile(
                suffix=Path(file.filename or "audio.wav").suffix,
                delete=True
            ) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp.flush()

                audio, _ = cast(
                    Tuple[np.ndarray, float],
                    await asyncio.to_thread(
                        lambda: librosa.load(tmp.name, sr=16000, mono=True)
                    )
                )

            audio_float = audio.astype(np.float32)
            result = await transcribe_audio_async(audio_float, timeout_sec=settings.request_timeout)

            results.append(TranscribeResult(
                text=result.get("text", ""),
                audio_duration_sec=result.get("audio_duration_sec", len(audio) / 16000),
                inference_time_sec=result.get("inference_time_sec", 0.0),
                tokens_generated=0,
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
    Transcribe audio files by path using plain Granite ASR.
    Paths must be accessible by the server and within allowed directories
    if TRANSCRIBE_ALLOWED_AUDIO_DIR is set.
    """
    start_time = time.perf_counter()

    results = []

    for path in req.wav_paths:
        try:
            resolved = validate_path_security(path, settings)

            audio, _ = cast(
                Tuple[np.ndarray, float],
                await asyncio.to_thread(
                    lambda: librosa.load(str(resolved), sr=16000, mono=True)
                )
            )

            audio_float = audio.astype(np.float32)
            result = await transcribe_audio_async(audio_float, timeout_sec=settings.request_timeout)

            results.append(TranscribeResult(
                text=result.get("text", ""),
                audio_duration_sec=result.get("audio_duration_sec", len(audio) / 16000),
                inference_time_sec=result.get("inference_time_sec", 0.0),
                tokens_generated=0,
            ))

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


@app.post("/transcribe/saa")
async def transcribe_saa_endpoint(
    req: DiarizePathsRequest,
    settings = Depends(get_settings),
    _: str = Depends(verify_api_key)
):
    """
    Pure Granite SAA transcription (speaker-attributed) without voiceprint matching.
    Returns text with [Speaker N]: tags directly from the model.
    """
    try:
        resolved = validate_path_security(req.wav_path, settings)
    except PathSecurityError as e:
        return JSONResponse(status_code=403, content={"error": f"Access denied: {e.message}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    try:
        audio, sr = cast(
            Tuple[np.ndarray, float],
            await asyncio.to_thread(
                lambda: librosa.load(str(resolved), sr=16000, mono=True)
            )
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to load audio: {str(e)}"})

    audio_float = audio.astype(np.float32)
    result = await transcribe_saa_async(audio_float, timeout_sec=300 if len(audio_float) / 16000 > 60 else settings.request_timeout)

    return {
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "audio_duration_sec": result.get("audio_duration_sec", 0),
        "inference_time_sec": result.get("inference_time_sec", 0),
    }


# WebSocket endpoint for streaming
app.add_api_websocket_route("/ws/stream", stream_transcribe)


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
