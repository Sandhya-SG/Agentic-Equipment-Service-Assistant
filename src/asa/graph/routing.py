"""Conditional edges for the graph. Routing is plain code, not a model decision."""

from __future__ import annotations

from typing import Literal

from asa.graph.state import AgentState


def route_after_guard(state: AgentState) -> Literal["blocked", "continue"]:
    return "blocked" if state.get("status") == "blocked" else "continue"


def route_after_generate(state: AgentState) -> Literal["unavailable", "continue"]:
    return "unavailable" if state.get("status") == "unavailable" else "continue"
