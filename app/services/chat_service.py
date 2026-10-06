"""Chat pipeline: input guardrail -> LangGraph agents -> deterministic safety check.

Guardrails come from asa.guardrails and are deterministic (pattern-based), so the
decision to block or escalate is predictable and testable, not left to the model.
The agents (request, planning, agentic RAG, diagnostic, safety) answer only from the
equipment manual selected by the user; this module wraps them with the input guard,
PII redaction before text leaves for the model provider, the audit trail and a final
rule-based hazard check that can only add caution, never remove it.
"""

import logging
import time

from app.schemas.response import ChatResponse, SafetyInfo
from app.services.graph_runner import run_graph
from asa.components import logging_sub
from asa.guardrails.injection import scan_user_input
from asa.guardrails.safety_rules import check_text

logger = logging.getLogger(__name__)

BLOCKED_MESSAGE = "This request was blocked because it looks like an attempt to override the assistant's instructions."
UNAVAILABLE_MESSAGE = (
    "The assistant is temporarily unavailable. Please try again shortly or contact a qualified engineer."
)
ESCALATION_MESSAGE = (
    "This task involves hazards that are not field-serviceable ({hazards}). "
    "Do not proceed on your own. Escalate to a qualified engineer."
)
HAZARD_WARNING = (
    "Hazards detected ({hazards}). Follow lockout/tagout and use the required PPE. A human must review before acting."
)
MAX_SOURCES = 5


def _hazard_safety(hazards: list[str], ppe: list[str]) -> SafetyInfo:
    return SafetyInfo(
        hazards=hazards,
        ppe_required=ppe,
        requires_human_review=True,
        warning=HAZARD_WARNING.format(hazards=", ".join(hazards)),
    )


def _sources(chunks) -> list[str]:
    """Readable, de-duplicated citations for the passages the answer was built from."""
    seen: list[str] = []
    for chunk in chunks or []:
        label = f"{chunk.source_file}, p.{chunk.page}, {chunk.section_title}"
        if label not in seen:
            seen.append(label)
    return seen[:MAX_SOURCES]


def _agent_status(state: dict) -> str:
    if state.get("request_status") == "CLARIFY":
        return "clarification"
    if state.get("safety_verdict") == "halt":
        return "halted"
    if state.get("escalated"):
        return "escalated"
    return "ok"


def handle_chat(message: str, conversation_id: str, equipment_model: str | None = None) -> ChatResponse:
    started = time.monotonic()

    scan = scan_user_input(message)
    if scan.flagged:
        # Log categories and length only, never the raw message.
        logger.warning("chat blocked: injection categories=%s length=%d", scan.categories, len(message))
        return ChatResponse(
            response=BLOCKED_MESSAGE,
            conversation_id=conversation_id,
            status="blocked",
            safety=SafetyInfo(requires_human_review=True),
        )

    # Redact emails, card numbers and secrets before the text goes to the model provider.
    safe_message = logging_sub._redact(message)
    run_id = logging_sub.start_run(safe_message)

    try:
        state = run_graph(safe_message, equipment_model)
    except Exception as exc:
        # OpenAI error, missing index, bad key... Never leak details to the user.
        logger.error("chat failed: run_id=%s error=%s: %s", run_id, type(exc).__name__, exc)
        return ChatResponse(
            response=UNAVAILABLE_MESSAGE, conversation_id=conversation_id, status="unavailable", run_id=run_id
        )

    status = _agent_status(state)
    answer = state.get("final_answer") or state.get("clarification_question") or ""
    if not answer:
        logger.error("chat produced no answer: run_id=%s", run_id)
        return ChatResponse(
            response=UNAVAILABLE_MESSAGE, conversation_id=conversation_id, status="unavailable", run_id=run_id
        )

    # Rule-based check on the question and the answer. It can only add caution.
    query_check = check_text(message)
    answer_check = check_text(answer)
    hazards = sorted(set(state.get("hazards") or []) | set(query_check.hazards) | set(answer_check.hazards))
    ppe = sorted(set(state.get("ppe_required") or []) | set(query_check.ppe_required) | set(answer_check.ppe_required))
    # Only the user's own request can force withholding. The answer is quoted from the manual,
    # whose safety sections naturally mention high voltage; there it only adds warnings and PPE.
    escalate_only = query_check.escalate_only

    safety = SafetyInfo()
    if status == "clarification":
        pass
    elif hazards and escalate_only and status == "ok":
        # Withhold the procedure: unsafe guidance must not be actionable.
        status = "escalated"
        answer = ESCALATION_MESSAGE.format(hazards=", ".join(query_check.hazards))
        safety = _hazard_safety(hazards, ppe)
    elif hazards:
        safety = _hazard_safety(hazards, ppe)

    if status in ("halted", "escalated"):
        safety.requires_human_review = True

    latency_ms = int((time.monotonic() - started) * 1000)
    logging_sub.end_run(run_id, {**state, "final_answer": answer, "escalated": status == "escalated"})
    logging_sub.emit_metrics(run_id, state, latency_ms / 1000)
    logger.info(
        "chat handled: run_id=%s status=%s specialist=%s hazards=%s latency_ms=%d",
        run_id,
        status,
        state.get("current_step"),
        hazards,
        latency_ms,
    )
    return ChatResponse(
        response=answer,
        conversation_id=conversation_id,
        status=status,
        safety=safety,
        sources=[] if status == "clarification" else _sources(state.get("retrieved_chunks")),
        specialist=state.get("current_step"),
        escalation_reason=state.get("escalation_reason") or None,
        run_id=run_id,
    )
