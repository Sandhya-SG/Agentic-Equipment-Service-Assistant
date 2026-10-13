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
from app.services.clarification_store import (
    PendingClarification,
    clear_pending,
    get_pending,
    set_pending,
)
from app.services.graph_runner import run_graph
from asa.agents.clarification import (
    resolve_clarification,
)
from asa.components import logging_sub
from asa.components.tracing import observation, record_outcome, tag_run, trace_request
from asa.graph.state import CONFIDENCE_FLOOR, MAX_CLARIFICATIONS, Citation
from asa.guardrails.injection import scan_user_input
from asa.guardrails.safety_rules import check_defeat_request, check_text
from asa.guardrails.schema_guard import guard_output

logger = logging.getLogger(__name__)

BLOCKED_MESSAGE = "This request was blocked because it looks like an attempt to override the assistant's instructions."
DEFEAT_HALT_MESSAGE = (
    "Safety decision: HALT\n\n"
    "Bypassing, defeating or disabling a safety control ({control}) is not permitted. "
    "Keep all safety controls in place. If a safety control is preventing the work, "
    "stop and contact a qualified engineer."
)
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
LOW_CONFIDENCE_WARNING = "Low evidence confidence ({confidence:.2f}): verify with a qualified engineer before acting."
MAX_SOURCES = 5


def _hazard_labels(hazards: list[str]) -> str:
    """Short, readable hazard names for a warning: "Electrical Hazards, high voltage".

    The Safety Agent lists documented hazards as full manual sentences ("Electrical Hazards:
    Hazardous voltages may exist ...") and the rules use identifiers ("high_voltage"). The
    warning names each hazard once; the details stay in the answer with their sources.
    """
    labels: list[str] = []
    for hazard in hazards:
        label = " ".join(str(hazard).split(":", 1)[0].replace("_", " ").split())[:40]
        if label and label.lower() not in {existing.lower() for existing in labels}:
            labels.append(label)
    return ", ".join(labels)


def _hazard_safety(hazards: list[str], ppe: list[str]) -> SafetyInfo:
    return SafetyInfo(
        hazards=hazards,
        ppe_required=ppe,
        requires_human_review=True,
        warning=HAZARD_WARNING.format(hazards=_hazard_labels(hazards)),
    )


def _source_label(source_file, page, section) -> str:
    label = f"{source_file}, p.{page}"
    return label if not section or section == "Unknown" else f"{label}, {section}"


def _sources(chunks, explanation: dict | None = None) -> list[str]:
    """Readable, de-duplicated references for the passages the answer relies on.

    These are the passages the explanation layer verified as supporting the answer. When there are
    none (no explanation, or nothing could be quoted) the retrieved passages are listed instead, so
    the user still sees where the assistant looked.
    """
    citations = [c for c in (explanation or {}).get("citations") or [] if isinstance(c, dict)]
    if citations:
        refs = [(c.get("source_file", ""), c.get("page", 0), c.get("section")) for c in citations]
    else:
        refs = [(chunk.source_file, chunk.page, chunk.section_title) for chunk in chunks or []]
    seen: list[str] = []
    for ref in refs:
        label = _source_label(*ref)
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


def _is_standalone_request(message: str, equipment_model: str | None) -> bool:
    """True when a message sent while a clarification is pending is a complete question on its own.

    Engineers often answer a clarification with a new question instead. Such a message is answered
    as a fresh request rather than merged into the earlier one. The Request Agent judges it; if it
    cannot (no key, network error), the message is treated as a clarification answer, as before.
    """
    try:
        from asa.agents.request import request_node

        result = request_node({"raw_query": message, "equipment_model": equipment_model})
        return result.get("request_status") == "READY"
    except Exception as exc:
        logger.warning("standalone check failed, treating the message as a clarification: %s", type(exc).__name__)
        return False


def _flag_low_confidence(safety: SafetyInfo, status: str, explanation: dict | None, confidence: float) -> None:
    """Below CONFIDENCE_FLOOR an answer is still shown, but marked for human review with a warning.

    Like the hazard rules it can only add caution: it never changes the answer or the status. It
    applies only when the explanation layer produced a confidence (otherwise there is none to judge).
    """
    if status != "ok" or explanation is None or confidence >= CONFIDENCE_FLOOR:
        return
    warning = LOW_CONFIDENCE_WARNING.format(confidence=confidence)
    safety.requires_human_review = True
    safety.warning = f"{safety.warning} {warning}" if safety.warning else warning


def _record_explanation(state: dict, explanation: dict | None, confidence: float) -> None:
    """Fill the "Explanation Module writes" fields of the shared state (asa.graph.state.AgentState).

    The explanation layer is a pure function and never changes the state, so the pipeline records its
    result: the confidence, and every citation as a `Citation` pointing to the exact manual text.
    """
    if explanation is None:
        return
    state["confidence"] = confidence
    state["citations"] = [
        Citation(
            doc_id=str(c.get("doc_id", "")),
            section_id=str(c.get("section_id", "")),
            revision=str(c.get("revision", "")),
            quote_span=str(c.get("quote_span") or c.get("quote", "")),
        )
        for c in explanation.get("citations") or []
        if isinstance(c, dict)
    ]


def _confidence(explanation: dict | None, state: dict) -> float:
    try:
        value = (explanation or {}).get("confidence", state.get("confidence", 0.0))
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def handle_chat(message: str, conversation_id: str, equipment_model: str | None = None) -> ChatResponse:
    """Run one chat request inside a Langfuse trace (a no-op when tracing is off).

    The trace input is the redacted question (the raw text never leaves this service) and the
    output is what the user is shown; both are masked unless LANGFUSE_CAPTURE_CONTENT=true.
    """
    with trace_request(session_id=conversation_id, equipment_model=equipment_model) as root:
        root.update(input={"message": logging_sub._redact(message), "equipment_model": equipment_model})
        response = _handle_chat(message, conversation_id, equipment_model)
        root.update(
            output={
                "status": response.status,
                "response": response.response,
                "specialist": response.specialist,
                "sources": response.sources,
            },
        )
        record_outcome(
            status=response.status,
            hazard_count=len(response.safety.hazards),
            source_count=len(response.sources),
            escalated=response.status in ("escalated", "halted"),
            confidence=(response.explanation or {}).get("confidence"),
            specialist=response.specialist,
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


def _halt_defeat_request(
    control: str, safe_message: str, conversation_id: str, equipment_model: str | None, started: float
) -> ChatResponse:
    """A request to defeat a safety control is halted before any agent or model runs, and recorded."""
    run_id = logging_sub.start_run(safe_message)
    clear_pending(conversation_id)
    answer = DEFEAT_HALT_MESSAGE.format(control=control)
    audit = {
        "final_answer": answer,
        "status": "halted",
        "safety_verdict": "halt",
        "escalated": False,
        "requires_human_review": True,
        "equipment_model": equipment_model,
        "hazard_categories": ["safety_control_defeat"],
        "escalation_reason": None,
        "explained": False,
        "output_guard": {"action": "none", "findings": []},
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "trace_summary": [],
    }
    logging_sub.end_run(run_id, audit)
    logging_sub.emit_metrics(run_id, audit, time.monotonic() - started)
    logger.warning("chat halted: request to defeat a safety control (%s) run_id=%s", control, run_id)
    return ChatResponse(
        response=answer,
        conversation_id=conversation_id,
        status="halted",
        safety=SafetyInfo(requires_human_review=True),
        run_id=run_id,
    )


def _handle_chat(message: str, conversation_id: str, equipment_model: str | None = None) -> ChatResponse:
    started = time.monotonic()

    with observation("detect-prompt-injection", as_type="guardrail") as guard:
        scan = scan_user_input(message)
        guard.update(output={"flagged": scan.flagged, "categories": list(scan.categories)})
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
    with observation("redact-pii", as_type="guardrail") as guard:
        safe_message = logging_sub._redact(message)
        guard.update(output={"redacted": safe_message != message})

    # Deterministic and final: no agent or model gets to decide whether to help defeat a safety control.
    with observation("detect-safety-control-defeat", as_type="guardrail") as guard:
        defeated_control = check_defeat_request(message)
        guard.update(output={"flagged": defeated_control is not None, "categories": [defeated_control or "none"]})
    if defeated_control:
        return _halt_defeat_request(defeated_control, safe_message, conversation_id, equipment_model, started)

    pending = get_pending(conversation_id)

    clarification_response = None
    resolved_query = None
    clarification_count = 0

    if pending is not None:
        # Equipment identity must remain stable across the
        # clarification exchange.
        if pending.equipment_model and equipment_model and pending.equipment_model != equipment_model:
            clear_pending(conversation_id)

            pending = None

        elif _is_standalone_request(safe_message, equipment_model or pending.equipment_model):
            # A complete new question, not an answer to the clarification: start afresh.
            clear_pending(conversation_id)

            pending = None

        else:
            clarification_response = safe_message

            clarification_count = pending.clarification_count + 1

            resolved_query = resolve_clarification(
                original_query=pending.original_query,
                clarification_response=clarification_response,
            )

    run_id = logging_sub.start_run(safe_message)

    try:
        with tag_run(run_id):
            state = run_graph(
                safe_message,
                equipment_model,
                clarification_response=clarification_response,
                resolved_query=resolved_query,
                clarification_count=clarification_count,
            )
    except Exception as exc:
        # OpenAI error, missing index, bad key... Never leak details to the user.
        logger.error("chat failed: run_id=%s error=%s: %s", run_id, type(exc).__name__, exc)
        return _unavailable(run_id, conversation_id, equipment_model, started, type(exc).__name__)

    status = _agent_status(state)

    if status == "clarification" and clarification_count >= MAX_CLARIFICATIONS:
        status = "escalated"

        state["escalated"] = True

        state["escalation_reason"] = (
            "The request remained ambiguous after the maximum number of clarification attempts."
        )

        state["final_answer"] = (
            "I still do not have enough information to "
            "route this equipment-service request reliably. "
            "Please escalate the request to a qualified "
            "engineer for further clarification."
        )

        clear_pending(conversation_id)

    elif status == "clarification":
        original_query = pending.original_query if pending is not None else safe_message

        set_pending(
            conversation_id,
            PendingClarification(
                original_query=original_query,
                clarification_question=(state.get("clarification_question") or ""),
                equipment_model=equipment_model,
                clarification_count=clarification_count,
            ),
        )

    else:
        clear_pending(conversation_id)

    answer = state.get("final_answer") or state.get("clarification_question") or ""
    if not answer:
        logger.error("chat produced no answer: run_id=%s", run_id)
        return _unavailable(run_id, conversation_id, equipment_model, started, "EmptyAnswer")

    # Rule-based check on the question and the answer. It can only add caution.
    with observation("check-hazard-rules", as_type="guardrail") as guard:
        query_check = check_text(message)
        answer_check = check_text(answer)
        guard.update(
            output={
                "query_hazards": list(query_check.hazards),
                "answer_hazards": list(answer_check.hazards),
                "escalate_only": query_check.escalate_only,
            }
        )
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
        answer = ESCALATION_MESSAGE.format(hazards=_hazard_labels(query_check.hazards))
        safety = _hazard_safety(hazards, ppe)
    elif hazards:
        safety = _hazard_safety(hazards, ppe)

    # Response firewall: the last check on what leaves the system. Secrets, personal data that is not in the
    # manual, executable markup and over-long text are cleaned; a leaked prompt or internal structure is withheld
    # and the request goes to a human. Findings are categories only, never the text itself.
    with observation("check-response", as_type="guardrail") as guard:
        output = guard_output(answer, [chunk.text for chunk in state.get("retrieved_chunks") or []])
        guard.update(output={"action": output.action, "findings": list(output.findings)})
    if output.action != "none":
        logger.warning(
            "chat answer %s by the response firewall: findings=%s run_id=%s",
            output.action,
            list(output.findings),
            run_id,
        )
        answer = output.text
    if output.action == "withheld" and status in ("ok", "clarification"):
        status = "escalated"

    if status in ("halted", "escalated"):
        safety.requires_human_review = True

    trace = _trace_steps(state)
    # Explain what the user is shown: after a withholding the shown answer is the escalation message.
    explanation = _explain({**state, "final_answer": answer}, status)
    confidence = _confidence(explanation, state)
    _record_explanation(state, explanation, confidence)
    _flag_low_confidence(safety, status, explanation, confidence)
    sources = [] if status == "clarification" else _sources(state.get("retrieved_chunks"), explanation)
    level = (explanation or {}).get("confidence_level") or (
        logging_sub.confidence_level(confidence) if confidence > 0 else None
    )

    latency_s = time.monotonic() - started
    escalation_reason = state.get("escalation_reason") or None
    if output.action == "withheld":
        escalation_reason = escalation_reason or "The answer was withheld by the response firewall."
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
        "output_guard": {"action": output.action, "findings": list(output.findings)},
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
