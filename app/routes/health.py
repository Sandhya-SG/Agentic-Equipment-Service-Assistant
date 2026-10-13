import hmac
import os

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse

from app.services.graph_runner import is_ready

router = APIRouter(tags=["health"])

OPEN_ENVIRONMENTS = ("development", "dev", "local", "test")


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "backend",
        "version": "0.1.0",
    }


@router.get("/ready")
async def readiness_check():
    """Ready only when the agents have loaded, so Kubernetes sends traffic to working pods only."""
    if not is_ready():
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "service": "backend", "reason": "agents not loaded"},
        )
    return {
        "status": "ready",
        "service": "backend",
    }


@router.get("/version")
async def version_info(x_version_token: str | None = Header(default=None)):
    """What this backend was built from: commit, models, prompt hashes and the manuals behind the index.

    It describes the internal configuration, so it is protected: with `VERSION_TOKEN` set the caller must send it in
    the `X-Version-Token` header. With no token it is open only in development (`ENVIRONMENT` development, dev, local
    or test, and the default when unset); in any deployed environment (kubernetes, production, ...) it is off (404).
    """
    token = os.getenv("VERSION_TOKEN", "")
    if token:
        if not x_version_token or not hmac.compare_digest(x_version_token, token):
            raise HTTPException(status_code=401, detail="a valid X-Version-Token is required")
    elif os.getenv("ENVIRONMENT", "development").lower() not in OPEN_ENVIRONMENTS:
        raise HTTPException(status_code=404, detail="not found")
    from asa.versioning import build_info

    return build_info()
