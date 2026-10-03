"""Chat service: runs a request through the LangGraph workflow and shapes the API response.

The workflow itself (guards, retrieval, model call, safety checks) lives in
asa.graph. This module only connects it to the model client and maps the final
graph state to the response schema.
"""

import logging

from app.schemas.response import ChatResponse, SafetyInfo
from app.services.llm_client import generate_with_usage
from asa.graph.orchestrator import build_graph, run
from asa.graph.state import AgentState

logger = logging.getLogger(__name__)

HAZARD_WARNING = (
    "Hazards detected ({hazards}). Follow lockout/tagout and use the required PPE. A human must review before acting."
)

# The lambda looks the model call up at call time, so tests can replace it.
_GRAPH = build_graph(lambda prompt: generate_with_usage(prompt))


def _sources(state: AgentState) -> list[str]:
    return [f"{c.doc_id} {c.section_id} (rev {c.revision})" for c in state.get("citations", [])]


def _safety(state: AgentState) -> SafetyInfo:
    hazards = state.get("hazards", [])
    review = bool(hazards) or state["status"] in ("blocked", "escalated")
    return SafetyInfo(
        hazards=hazards,
        ppe_required=state.get("ppe_required", []),
        requires_human_review=review,
        warning=HAZARD_WARNING.format(hazards=", ".join(hazards)) if hazards else None,
    )


def handle_chat(message: str, conversation_id: str) -> ChatResponse:
    state = run(_GRAPH, message, conversation_id)
    logger.info("chat handled: status=%s hazards=%s", state["status"], state.get("hazards", []))
    return ChatResponse(
        response=state["final_answer"],
        conversation_id=conversation_id,
        status=state["status"],
        safety=_safety(state),
        sources=_sources(state),
    )
