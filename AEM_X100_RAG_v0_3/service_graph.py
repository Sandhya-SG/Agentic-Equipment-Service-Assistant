from langgraph.graph import (
    StateGraph,
    START,
    END
)

from graph_state import AgentState

from graph_nodes import (
    request_node,
    planning_node,
    agentic_rag_node,
    response_node
)


def build_service_graph():

    builder = StateGraph(
        AgentState
    )

    # ----------------------------------------
    # Nodes
    # ----------------------------------------

    builder.add_node(
        "request",
        request_node
    )

    builder.add_node(
        "planning",
        planning_node
    )

    builder.add_node(
        "agentic_rag",
        agentic_rag_node
    )

    builder.add_node(
        "response",
        response_node
    )

    # ----------------------------------------
    # Edges
    # ----------------------------------------

    builder.add_edge(
        START,
        "request"
    )

    builder.add_edge(
        "request",
        "planning"
    )

    builder.add_edge(
        "planning",
        "agentic_rag"
    )

    builder.add_edge(
        "agentic_rag",
        "response"
    )

    builder.add_edge(
        "response",
        END
    )

    return builder.compile()