"""
Middleware configuration: CORS, rate limiting, and request logging.
"""

import time
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog

from settings import get_settings

log = structlog.get_logger()


async def request_middleware(request: Request, call_next):
    """Global request middleware for logging and size limits."""
    settings = get_settings()
    request_id = request.headers.get("X-Request-ID", str(time.time_ns()))
    
    # Check request size
    content_length = request.headers.get("content-length")
    max_size = settings.max_request_size_mb * 1024 * 1024
    if content_length and int(content_length) > max_size:
        return JSONResponse(
            status_code=413,
            content={"error": f"Request too large. Max: {settings.max_request_size_mb}MB"}
        )
    
    # Log request
    start = time.perf_counter()
    
    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        
        log.info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=f"{elapsed*1000:.1f}"
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed*1000:.1f}ms"
        return response
        
    except Exception as e:
        log.exception("request_error", request_id=request_id, error=str(e))
        raise


def apply_middleware(app):
    """Apply all middleware to FastAPI app."""
    settings = get_settings()
    
    # Request logging middleware
    app.middleware("http")(request_middleware)
    
    # CORS middleware
    if settings.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins.split(","),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Rate limiting (optional)
    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.util import get_remote_address
        from slowapi.errors import RateLimitExceeded
        
        if settings.enable_rate_limit:
            limiter = Limiter(key_func=get_remote_address)
            app.state.limiter = limiter
            app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
            log.info("rate_limiting_enabled", limit=settings.rate_limit)
    except ImportError:
        log.warning("rate_limiting_unavailable", reason="slowapi not installed")
