"""Conditional routing functions for the LangGraph service workflow."""

from __future__ import annotations

from asa.graph.state import AgentState


def route_after_request(
    state: AgentState,
) -> str:
    """
    Route after the Request Agent.

    READY   -> Planning Agent
    CLARIFY -> End the current graph invocation so the
               application can ask the engineer for more input.
    """

    status = state.get(
        "request_status"
    )

    if status == "READY":
        return "planning"

    if status == "CLARIFY":
        return "clarify"

    # Fail conservatively if Request Agent produced
    # an unexpected state.
    return "clarify"


def route_after_planning(
    state: AgentState,
) -> str:
    """
    Route to the specialist selected by the Planning Agent.
    """

    current_step = state.get(
        "current_step"
    )

    if current_step == "agentic_rag":
        return "agentic_rag"

    if current_step == "diagnostic":
        return "diagnostic"

    if current_step == "safety":
        return "safety"

    # Unknown planning output should not silently
    # continue to an arbitrary specialist.
    return "safety"