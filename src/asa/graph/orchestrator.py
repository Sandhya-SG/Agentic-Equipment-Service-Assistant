"""Builds and runs the LangGraph workflow.

    guard_input -> retrieve -> generate -> safety_check -> respond
         |                         |
      blocked (end)          unavailable (end)

The model call is passed in, so this package does not depend on the web layer and
tests can supply a fake.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from asa.components.tracing import annotate, traced
from asa.graph import nodes
from asa.graph.nodes import GenerateFn
from asa.graph.routing import route_after_generate, route_after_guard
from asa.graph.state import AgentState, new_state


def build_graph(generate: GenerateFn):
    graph = StateGraph(AgentState)
    graph.add_node("guard_input", nodes.guard_input)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.make_generate_node(generate))
    graph.add_node("safety_check", nodes.safety_check)
    graph.add_node("respond", nodes.respond)

    graph.add_edge(START, "guard_input")
    graph.add_conditional_edges("guard_input", route_after_guard, {"blocked": END, "continue": "retrieve"})
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", route_after_generate, {"unavailable": END, "continue": "safety_check"})
    graph.add_edge("safety_check", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


@traced("chat")
def run(graph, query: str, conversation_id: str = "") -> AgentState:
    """Run one request through the graph and return the final state."""
    annotate(conversation_id=conversation_id)
    return graph.invoke(new_state(query))
