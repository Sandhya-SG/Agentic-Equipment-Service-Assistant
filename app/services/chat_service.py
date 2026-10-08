"""Chat pipeline: input guardrail -> LangGraph agents -> deterministic safety check.

Guardrails come from asa.guardrails and are deterministic (pattern-based), so the
decision to block or escalate is predictable and testable, not left to the model.
The agents (request, planning, agentic RAG, diagnostic, safety) answer only from the
equipment manual selected by the user; this module wraps them with the input guard,
PII redaction before text leaves for the model provider, the audit trail and a final
rule-based hazard check that can only add caution, never remove it.

Order of one request: guard -> redact -> agents (traced step by step) -> hazard check ->
explanation (optional layer) -> audit trail, metrics and trace scores -> response.
Every outcome, including blocked and unavailable, is written to the audit trail.
"""

import hashlib
import logging
import time

from app.schemas.response import ChatResponse, SafetyInfo, TraceStep
from app.services.graph_runner import run_graph
from asa.components import logging_sub
from asa.components.tracing import record_outcome, tag_run, trace_request
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


def _trace_steps(state: dict) -> list[TraceStep]:
    """The agents that ran, in order, with their duration (from the traced graph run)."""
    steps = []
    for event in state.get("trace") or []:
        detail = dict(getattr(event, "detail", None) or {})
        steps.append(
            TraceStep(
                agent=getattr(event, "agent", "unknown"), duration_ms=int(detail.pop("duration_ms", 0)), detail=detail
            )
        )
    return steps


def _explain(state: dict, status: str) -> dict | None:
    """Call the Explainability layer if it exists. It is built by another team member.

    Contract: `asa.components.explanation.explain(state, status) -> dict`, a pure function with no
    logging and no outside calls. It returns at least `confidence` (0 to 1) and `confidence_level`.
    Until it exists, or if it fails, the response simply has no explanation: the answer, its sources
    and every safety decision are unaffected, and the audit trail records that none was produced.
    """
    try:
        from asa.components import explanation

        explain = getattr(explanation, "explain", None)
        if explain is None:
            return None
        result = explain(state, status)
        return result if isinstance(result, dict) else None
    except Exception as exc:
        logger.warning("explanation failed: %s", type(exc).__name__)
        return None


def _confidence(explanation: dict | None, state: dict) -> float:
    try:
        value = (explanation or {}).get("confidence", state.get("confidence", 0.0))
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def handle_chat(message: str, conversation_id: str, equipment_model: str | None = None) -> ChatResponse:
    """Run one chat request inside a Langfuse trace (a no-op when tracing is off)."""
    with trace_request(session_id=conversation_id, equipment_model=equipment_model):
        response = _handle_chat(message, conversation_id, equipment_model)
        record_outcome(
            status=response.status,
            hazard_count=len(response.safety.hazards),
            source_count=len(response.sources),
            escalated=response.status in ("escalated", "halted"),
            confidence=(response.explanation or {}).get("confidence"),
        )
        return response


def _unavailable(run_id: str, conversation_id: str, equipment_model: str | None, started: float, error_type: str):
    """The assistant could not answer: tell the user, and still record the outcome."""
    logging_sub.end_run(
        run_id,
        {"final_answer": "", "status": "unavailable", "equipment_model": equipment_model, "error_type": error_type},
    )
    logging_sub.emit_metrics(
        run_id, {"status": "unavailable", "equipment_model": equipment_model}, time.monotonic() - started
    )
    return ChatResponse(
        response=UNAVAILABLE_MESSAGE, conversation_id=conversation_id, status="unavailable", run_id=run_id
    )


def _handle_chat(message: str, conversation_id: str, equipment_model: str | None = None) -> ChatResponse:
    started = time.monotonic()

    scan = scan_user_input(message)
    if scan.flagged:
        # Log categories and length only, never the raw message.
        logger.warning("chat blocked: injection categories=%s length=%d", scan.categories, len(message))
        run_id = logging_sub.log_blocked(conversation_id, scan.categories, len(message))
        return ChatResponse(
            response=BLOCKED_MESSAGE,
            conversation_id=conversation_id,
            status="blocked",
            safety=SafetyInfo(requires_human_review=True),
            run_id=run_id,
        )

    # Redact emails, card numbers and secrets before the text goes to the model provider.
    safe_message = logging_sub._redact(message)
    run_id = logging_sub.start_run(safe_message)

    try:
        with tag_run(run_id):
            state = run_graph(safe_message, equipment_model)
    except Exception as exc:
        # OpenAI error, missing index, bad key... Never leak details to the user.
        logger.error("chat failed: run_id=%s error=%s: %s", run_id, type(exc).__name__, exc)
        return _unavailable(run_id, conversation_id, equipment_model, started, type(exc).__name__)

    status = _agent_status(state)
    answer = state.get("final_answer") or state.get("clarification_question") or ""
    if not answer:
        logger.error("chat produced no answer: run_id=%s", run_id)
        return _unavailable(run_id, conversation_id, equipment_model, started, "EmptyAnswer")

    # Rule-based check on the question and the answer. It can only add caution.
    query_check = check_text(message)
    answer_check = check_text(answer)
    rule_hazards = sorted(set(query_check.hazards) | set(answer_check.hazards))
    hazards = sorted(set(state.get("hazards") or []) | set(rule_hazards))
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

    sources = [] if status == "clarification" else _sources(state.get("retrieved_chunks"))
    trace = _trace_steps(state)
    # Explain what the user is shown: after a withholding the shown answer is the escalation message.
    explanation = _explain({**state, "final_answer": answer}, status)
    confidence = _confidence(explanation, state)
    level = (explanation or {}).get("confidence_level") or (
        logging_sub.confidence_level(confidence) if confidence > 0 else None
    )

    latency_s = time.monotonic() - started
    escalation_reason = state.get("escalation_reason") or None
    audit = {
        "final_answer": answer,
        "escalated": status == "escalated",
        "safety_verdict": state.get("safety_verdict"),
        "confidence": confidence,
        "confidence_level": level,
        "explained": explanation is not None,
        "status": status,
        "specialist": state.get("current_step"),
        "equipment_model": equipment_model,
        "source_refs": sources,
        "hazard_categories": rule_hazards,
        "hazard_count": len(safety.hazards),
        "ppe_required": safety.ppe_required,
        "requires_human_review": safety.requires_human_review,
        "escalation_reason": escalation_reason,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "trace_summary": [step.agent for step in trace],
    }
    if state.get("trace"):
        logging_sub.log_trace(run_id, state["trace"])
    logging_sub.end_run(run_id, audit)
    logging_sub.emit_metrics(
        run_id,
        {
            **audit,
            "source_count": len(sources),
            "retrieved_chunks": state.get("retrieved_chunks"),
            "retry_count": state.get("retry_count", 0),
        },
        latency_s,
    )
    logger.info(
        "chat handled: run_id=%s status=%s specialist=%s hazards=%s latency_ms=%d",
        run_id,
        status,
        state.get("current_step"),
        hazards,
        int(latency_s * 1000),
    )
    return ChatResponse(
        response=answer,
        conversation_id=conversation_id,
        status=status,
        safety=safety,
        sources=sources,
        specialist=state.get("current_step"),
        escalation_reason=escalation_reason,
        run_id=run_id,
        trace=trace,
        explanation=explanation,
    )
