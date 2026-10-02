from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.health import router as health_router
from app.schemas.request import ChatRequest

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Starter backend for the AEM AI assistant.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "status": "running",
        "environment": settings.environment,
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return {
        "response": f"Echo: {request.message}",
        "conversation_id": request.conversation_id or "demo-session",
        "status": "ok",
    }
