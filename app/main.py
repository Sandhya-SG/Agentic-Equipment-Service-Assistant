import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routes.health import router as health_router
from app.routes.reviews import router as reviews_router
from app.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.services import rate_limit, review_queue
from app.services.chat_service import handle_chat
from app.services.graph_runner import warm_up
from asa.components import logging_sub
from asa.components.llm import read_ledger, spend_cap
from asa.components.tracing import flush_tracing
from asa.ingestion.metadata import DOCUMENT_REGISTRY

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the embedding model and index now so the first request is not slow.
    if not warm_up():
        logger.warning("agents not ready: chat will answer 'unavailable' until the cause is fixed")
    yield
    flush_tracing()  # send any buffered traces before shutdown


app = FastAPI(
    lifespan=lifespan,
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
app.include_router(reviews_router)


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "status": "running",
        "environment": settings.environment,
    }


@app.get("/api/equipment")
async def equipment():
    """Equipment the assistant has manuals for, for the UI's selector."""
    return [{"id": d["equipment_model"], "label": d["title"]} for d in DOCUMENT_REGISTRY.values()]


# Plain `def` so FastAPI runs it in a threadpool: the agent run blocks.
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request):
    # Every chat request can cost model calls, so a client that sends too many is turned away before any work is done.
    peer = http_request.client.host if http_request.client else None
    wait = rate_limit.check(rate_limit.client_key(peer, http_request.headers.get("x-forwarded-for")))
    if wait:
        seconds = max(1, int(wait + 0.999))
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(seconds)},
            content={
                "status": "unavailable",
                "error": "rate_limited",
                "response": f"Too many requests. Please wait {seconds} seconds and try again.",
                "retry_after_s": seconds,
            },
        )
    # Without a conversation_id every request gets its own, so a pending clarification is never shared between
    # callers. (A fixed fallback id made unrelated API clients continue each other's clarification.) A client that
    # wants a multi-turn conversation sends the conversation_id returned in the first response.
    conversation_id = request.conversation_id or f"anon-{uuid.uuid4().hex[:12]}"
    result = handle_chat(request.message, conversation_id, request.equipment_model, request.clarification_state)
    status_code = {"blocked": 400, "unavailable": 503}.get(result.status, 200)
    if status_code == 200:
        return result
    return JSONResponse(status_code=status_code, content=result.model_dump())


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    """Record whether an answer was helpful, linked to its run so confidence can be checked later."""
    if not logging_sub.run_exists(request.run_id):
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return FeedbackResponse(recorded=logging_sub.log_feedback(request.run_id, request.helpful, request.comment or ""))


@app.get("/api/monitoring")
def monitoring(
    limit: int = Query(200, ge=1, le=5000, description="Only the most recent runs"),
    verify: bool = Query(True, description="Also re-walk the audit trail and check its hash chain"),
):
    """Operational numbers, alerts and the audit-trail integrity check. Aggregates only, no text."""
    summary = logging_sub.monitoring_summary(limit)
    return {
        "summary": summary,
        "alerts": logging_sub.check_alerts(summary),
        "audit_integrity": logging_sub.verify_audit_integrity() if verify else None,
        "review": review_queue.summary(),
        "spend": _spend(),
    }


def _spend() -> dict:
    """This pod's estimated OpenAI spend against its cap (an estimate, not the bill; each pod keeps its own ledger)."""
    ledger, cap = read_ledger(), spend_cap()
    return {
        "estimated_usd": round(ledger["estimated_usd"], 4),
        "calls": ledger["calls"],
        "cap_usd": cap,
        "cap_reached": cap is not None and ledger["estimated_usd"] >= cap,
    }
