"""
Production-ready ASR transcription server using Nemotron streaming ONNX model.

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

from fastapi import FastAPI, File, UploadFile, Depends, WebSocket, WebSocketDisconnect
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
    title="Nemotron Transcription Server",
    description="Production ASR transcription service using Nemotron streaming model",
    version="1.0.0",
    lifespan=lifespan
)

apply_middleware(app)
register_exception_handlers(app)


@app.get("/health", response_model=HealthResponse)
async def health():
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
    """
    Transcribe audio in 120s chunks using Nemotron streaming API.
    Returns timed segments with ~560ms granularity from streaming chunk positions.
    """
    import gc

    segments: list[TimedSegment] = []
    total_inference = 0.0
    total_tokens = 0
    chunk_samples = chunk_sec * 16000

    for start in range(0, len(audio), chunk_samples):
        chunk_start_sec = start / 16000
        audio_chunk = audio[start:start + chunk_samples]

        result = await transcribe_audio_async(
            audio_chunk, language, settings.request_timeout,
            use_chunked=True
        )

        for ct in result.get("chunk_texts", []):
            offset_start = chunk_start_sec + ct["start"]
            offset_end = chunk_start_sec + ct["end"]
            if ct["text"]:
                segments.append(TimedSegment(
                    start=round(offset_start, 3),
                    end=round(offset_end, 3),
                    text=ct["text"]
                ))

        total_inference += result.get("inference_time_sec", 0.0)
        total_tokens += result.get("tokens_generated", 0)

        audio_chunk = None
        result = None
        gc.collect()

    return segments, total_inference, total_tokens


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
                if '"type": "result"' in msg or '"error"' in msg:
                    break
        finally:
            state.embedding_cache.clear()
            gc.collect()

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/transcribe/upload", response_model=TranscribeResponse)
async def transcribe_upload(
    files: list[UploadFile] = File(..., description="Audio files to transcribe"),
    language: str = "en",
    _: str = Depends(verify_api_key)
):
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

                load_start = time.perf_counter()
                audio, _ = cast(
                    Tuple[np.ndarray, float],
                    await asyncio.to_thread(
                        lambda: librosa.load(tmp.name, sr=16000, mono=True)
                    )
                )
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

            timed_segments, total_inference, total_tokens = await _transcribe_chunks(
                audio, language, settings, full_mel_spectrogram
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
    start_time = time.perf_counter()
    results = []

    for path in req.wav_paths:
        try:
            resolved = validate_path_security(path, settings)

            load_start = time.perf_counter()
            audio, _ = cast(
                Tuple[np.ndarray, float],
                await asyncio.to_thread(
                    lambda: librosa.load(str(resolved), sr=16000, mono=True)
                )
            )
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

            timed_segments, total_inference, total_tokens = await _transcribe_chunks(
                audio, req.language, settings, full_mel_spectrogram
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


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """Transcribe a file with per-chunk progress and stream speaker-attributed segments.

    Client sends:
        {"wav_path": "...", "diarize_segments": [...], "language": "en", "api_key": "..."}
    Server streams:
        {"type": "progress", "chunk": N, "total_chunks": M, ...}
        {"type": "segment", "speaker": "...", "start": ..., "end": ..., "text": "..."}
        {"type": "done"}
    """
    await websocket.accept()
    settings = get_settings()
    import gc

    try:
        data = await websocket.receive_json()
        wav_path = data.get("wav_path", "")
        language = data.get("language", "en")
        diarize_segments = data.get("diarize_segments", [])

        api_key = data.get("api_key")
        if settings.api_keys:
            if api_key not in settings.api_key_set:
                await websocket.send_json({"type": "error", "message": "Invalid API key"})
                await websocket.close()
                return

        resolved = validate_path_security(wav_path, settings)

        audio, _ = librosa.load(str(resolved), sr=16000, mono=True)

        total_segments = len(diarize_segments)

        for idx, ds in enumerate(diarize_segments):
            start_sample = int(ds["start"] * 16000)
            end_sample = int(ds["end"] * 16000)
            seg_audio = audio[start_sample:end_sample]

            result = await transcribe_audio_async(
                seg_audio, language, settings.request_timeout,
                use_chunked=False
            )

            text = result.get("text", "").strip()
            if text:
                await websocket.send_json({
                    "type": "segment",
                    "speaker": ds["speaker"],
                    "start": round(ds["start"], 3),
                    "end": round(ds["end"], 3),
                    "text": text,
                    "ds_idx": idx
                })

            await websocket.send_json({
                "type": "progress",
                "chunk": idx + 1,
                "total_chunks": total_segments,
                "segment_duration_sec": round(ds["end"] - ds["start"], 2),
                "inference_time_sec": round(result.get("inference_time_sec", 0.0), 3)
            })

            seg_audio = None
            result = None
            gc.collect()

        await websocket.send_json({"type": "done"})

    except PathSecurityError as e:
        await websocket.send_json({"type": "error", "message": f"Access denied: {e.message}"})
    except WebSocketDisconnect:
        log.warn("ws_transcribe_disconnected")
    except Exception as e:
        log.error("ws_transcribe_error", error=str(e))
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# WebSocket endpoint for streaming
app.add_api_websocket_route("/ws/stream", stream_transcribe)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False,
    )
