"""API key authentication and tiered rate limiting."""

import hashlib

from fastapi import HTTPException, Request

from app.core.config import get_settings


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def verify_api_key(request: Request) -> dict:
    """
    Check X-API-Key header. Returns tier info.
    Raises 401 if key is invalid (when auth is enabled).
    """
    settings = get_settings()
    if not settings.api_keys_set:
        # Auth disabled — anonymous access
        return {"tier": "anonymous", "limit": settings.rate_limit_per_minute}

    raw_key = request.headers.get("X-API-Key")
    if not raw_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required.")

    hashed = hash_api_key(raw_key)
    if hashed not in settings.api_keys_set:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    return {"tier": "authenticated", "limit": settings.rate_limit_per_minute * 5}
