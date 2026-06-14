"""
Custom exceptions and FastAPI error handlers.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
import structlog

log = structlog.get_logger()


# ============================================================================
# Custom Exceptions
# ============================================================================

class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class TimeoutError(TranscriptionError):
    """Inference timeout error."""
    pass


class AudioValidationError(TranscriptionError):
    """Invalid audio input error."""
    pass


class PathSecurityError(TranscriptionError):
    """Path access security error."""
    pass


# ============================================================================
# Exception Handlers
# ============================================================================

async def transcription_error_handler(request: Request, exc: TranscriptionError):
    """Handle transcription-specific errors."""
    return JSONResponse(
        status_code=400 if isinstance(exc, (AudioValidationError, PathSecurityError)) else 500,
        content={
            "error": exc.message,
            "type": type(exc).__name__,
            "details": exc.details
        }
    )


async def general_error_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    log.exception("unhandled_error", path=request.url.path)
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


def register_exception_handlers(app):
    """Register all exception handlers with FastAPI app."""
    app.add_exception_handler(TranscriptionError, transcription_error_handler)
    app.add_exception_handler(Exception, general_error_handler)
