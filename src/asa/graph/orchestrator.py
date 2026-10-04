"""LangGraph orchestration for the AEM service assistant."""

from __future__ import annotations

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from asa.agents.diagnostic import diagnostic_node
from asa.agents.planning import planning_node
from asa.agents.rag import AgenticRAGAgent
from asa.agents.request import request_node
from asa.agents.safety import safety_node

from asa.graph.routing import (
    route_after_planning,
    route_after_request,
)

from asa.graph.state import (
    AgentState,
    Chunk,
)


# ---------------------------------------------------------------------------
# Specialist instances
# ---------------------------------------------------------------------------

rag_agent = AgenticRAGAgent()


# ---------------------------------------------------------------------------
# Real Agentic RAG node
# ---------------------------------------------------------------------------

def agentic_rag_node(
    state: AgentState,
) -> dict:
    """
    Execute the real equipment-aware Agentic RAG specialist
    and translate its result into shared LangGraph state.
    """

    print("\n[Agentic RAG Node]")

    question = (
        state.get("resolved_query")
        or state.get("raw_query")
        or ""
    ).strip()

    equipment_model = (
        state.get("equipment_model")
        or ""
    ).strip()

    if not question:
        raise ValueError(
            "Agentic RAG node requires a query."
        )

    if not equipment_model:
        raise ValueError(
            "Agentic RAG node requires equipment_model."
        )

    result = rag_agent.run(
        question=question,
        equipment_model=equipment_model,
        k=5,
    )

    retrieved_chunks = [
        Chunk(
            chunk_id=item["chunk_id"],
            doc_id=item["doc_id"],
            section_id=item["section_id"],
            revision=item["revision"],
            equipment_model=item["equipment_model"],
            text=item["text"],
            score=item["rrf_score"],
            source_file=item["source_file"],
            page=item["page"],
            section_title=item["section_title"],
        )
        for item in result["results"]
    ]

    sufficient = (
        result["retrieval_status"]
        == "GOOD"
    )

    return {
        "retrieval_strategy":
            "hybrid",

        "retrieved_chunks":
            retrieved_chunks,

        "sufficiency":
            sufficient,

        "retry_count":
            max(
                result["retrieval_attempts"] - 1,
                0,
            ),

        "escalated":
            result.get(
                "escalation_required",
                False,
            ),

        "escalation_reason":
            result.get(
                "senior_reason"
            )
            or "",

        "final_answer":
            result["answer"],
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_service_graph():
    """
    Build the v1.0.2D LangGraph.

    Request
        -> CLARIFY -> END
        -> READY -> Planning
                       -> real Agentic RAG
                       -> Diagnostic stub
                       -> Safety stub
    """

    graph = StateGraph(
        AgentState
    )

    graph.add_node(
        "request",
        request_node,
    )

    graph.add_node(
        "planning",
        planning_node,
    )

    graph.add_node(
        "agentic_rag",
        agentic_rag_node,
    )

    graph.add_node(
        "diagnostic",
        diagnostic_node,
    )

    graph.add_node(
        "safety",
        safety_node,
    )

    graph.add_edge(
        START,
        "request",
    )

    graph.add_conditional_edges(
        "request",
        route_after_request,
        {
            "planning":
                "planning",

            "clarify":
                END,
        },
    )

    graph.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "agentic_rag":
                "agentic_rag",

            "diagnostic":
                "diagnostic",

            "safety":
                "safety",
        },
    )

    graph.add_edge(
        "agentic_rag",
        END,
    )

    graph.add_edge(
        "diagnostic",
        END,
    )

    graph.add_edge(
        "safety",
        END,
    )

    return graph.compile()


service_graph = build_service_graph()