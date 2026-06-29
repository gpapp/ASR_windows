"""
Security: API key authentication and path validation.
"""

from typing import Optional
from pathlib import Path

from fastapi import Security, HTTPException, Depends
from fastapi.security import APIKeyHeader

from settings import Settings, get_settings
from api.exceptions import PathSecurityError

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    settings: Settings = Depends(get_settings)
) -> Optional[str]:
    """Verify API key if authentication is enabled."""
    if not settings.api_key_set:
        return None  # No auth required
    
    if not api_key or api_key not in settings.api_key_set:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key"
        )
    return api_key


def validate_path_security(path: str, settings: Settings) -> Path:
    """Validate that a path is allowed to be accessed."""
    resolved = Path(path).resolve()
    
    if not resolved.exists():
        raise PathSecurityError(f"File not found: {path}")
    
    if settings.allowed_audio_dir:
        allowed = settings.allowed_audio_dir.resolve()
        if not str(resolved).startswith(str(allowed)):
            raise PathSecurityError(
                f"Path not allowed. Must be under: {allowed}",
                details={"path": path}
            )
    
    # Check file extension
    allowed_extensions = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".webm"}
    if resolved.suffix.lower() not in allowed_extensions:
        raise PathSecurityError(
            f"File type not allowed: {resolved.suffix}",
            details={"path": path, "allowed": list(allowed_extensions)}
        )
    
    return resolved
