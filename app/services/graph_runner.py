"""Runs the LangGraph agent workflow for one chat request.

The graph is imported lazily because building it loads the embedding model, opens
the ChromaDB index and creates the OpenAI client. Doing that at import time would
make the API (and its tests) depend on the index and an API key just to start.
"""

import logging
import time
from datetime import datetime, timezone
from functools import lru_cache

from asa.components.tracing import graph_config, instrument_openai
from asa.graph.state import AgentState, TraceEvent, new_state

logger = logging.getLogger(__name__)

# Only these state keys are copied into the trace. They are short categorical values (never text),
# so the trace can be returned to the user interface and written to the audit trail.
_TRACE_DETAIL_KEYS = (
    "request_status",
    "current_step",
    "sufficiency",
    "retry_count",
    "safety_verdict",
    "escalated",
)


@lru_cache(maxsize=1)
def _graph():
    # Must run before the agents are imported: they do `from openai import OpenAI`.
    instrument_openai()
    from asa.graph.orchestrator import service_graph

    return service_graph


def is_ready() -> bool:
    """True once the agents are loaded (the graph was built successfully)."""
    return _graph.cache_info().currsize > 0


def warm_up() -> bool:
    """Build the graph now so the first user request is not slow. Never raises."""
    try:
        _graph()
        return True
    except Exception as exc:
        # Keep the API up: /health stays green and chat reports "unavailable".
        logger.error("graph warm-up failed: %s: %s", type(exc).__name__, exc)
        return False


def _trace_detail(update, duration_ms: int) -> dict:
    detail: dict = {"duration_ms": duration_ms}
    if isinstance(update, dict):
        for key in _TRACE_DETAIL_KEYS:
            value = update.get(key)
            if isinstance(value, (bool, int, float, str)) and (not isinstance(value, str) or len(value) <= 40):
                detail[key] = value
    return detail


def run_graph(message: str, equipment_model: str | None) -> AgentState:
    """Run request -> planning -> specialist and return the final shared state.

    The graph is streamed step by step so that the order and the duration of every agent are
    recorded in `state["trace"]` (the audit trail and the interface show which agents ran).
    The agents themselves are not changed.
    """
    initial = new_state(raw_query=message, equipment_model=equipment_model)
    trace: list[TraceEvent] = []
    final = None
    previous = time.monotonic()
    for mode, chunk in _graph().stream(initial, config=graph_config(), stream_mode=["updates", "values"]):
        if mode == "updates":
            now = time.monotonic()
            duration_ms = int((now - previous) * 1000)
            previous = now
            for node, update in chunk.items():
                trace.append(
                    TraceEvent(
                        agent=node,
                        action="completed",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        detail=_trace_detail(update, duration_ms),
                    )
                )
        elif mode == "values":
            final = chunk
    if final is None:
        raise RuntimeError("the agent graph produced no state")
    state = dict(final)
    state["trace"] = trace
    return state
