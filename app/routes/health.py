from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.services.graph_runner import is_ready

router = APIRouter(tags=["health"])


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
