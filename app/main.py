from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routes.health import router as health_router
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.services.chat_service import handle_chat

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Starter backend for the AEM AI assistant.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
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


# Plain `def` so FastAPI runs it in a threadpool: the model call blocks.
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = handle_chat(request.message, request.conversation_id or "demo-session")
    status_code = {"blocked": 400, "unavailable": 503}.get(result.status, 200)
    if status_code == 200:
        return result
    return JSONResponse(status_code=status_code, content=result.model_dump())
