"""Runs the LangGraph agent workflow for one chat request.

The graph is imported lazily because building it loads the embedding model, opens
the ChromaDB index and creates the OpenAI client. Doing that at import time would
make the API (and its tests) depend on the index and an API key just to start.
"""

import logging
from functools import lru_cache

from asa.graph.state import AgentState, new_state

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _graph():
    from asa.graph.orchestrator import service_graph

    return service_graph


def warm_up() -> bool:
    """Build the graph now so the first user request is not slow. Never raises."""
    try:
        _graph()
        return True
    except Exception as exc:
        # Keep the API up: /health stays green and chat reports "unavailable".
        logger.error("graph warm-up failed: %s: %s", type(exc).__name__, exc)
        return False


def run_graph(message: str, equipment_model: str | None) -> AgentState:
    """Run request -> planning -> specialist and return the final shared state."""
    return _graph().invoke(new_state(raw_query=message, equipment_model=equipment_model))
