"""API key validation endpoint."""
from fastapi import APIRouter, Header, Request

from app.core.middleware import verify_api_key

router = APIRouter()


@router.get("/keys/validate", summary="Validate API Key")
async def validate_key(
    request: Request, x_api_key: str = Header(None)  # noqa: B008
):
    key_info = await verify_api_key(request)
    return {"valid": True, "tier": key_info["tier"]}
