"""Integration test for the real Agentic RAG LangGraph branch."""

from asa.graph.orchestrator import service_graph
from asa.graph.state import new_state

from asa.ingestion.metadata import (
    THERMAL_STATION,
)


def main():

    print("\n" + "#" * 80)
    print("REAL RAG GRAPH TEST")
    print("#" * 80)

    initial_state = new_state(
        raw_query=(
            "What preventive maintenance "
            "should be performed?"
        ),
        equipment_model=THERMAL_STATION,
    )

    result = service_graph.invoke(
        initial_state
    )

    print("\n" + "=" * 80)
    print("FINAL GRAPH STATE")
    print("=" * 80)

    print(
        "\nRequest status:",
        result.get(
            "request_status"
        ),
    )

    print(
        "Selected specialist:",
        result.get(
            "current_step"
        ),
    )

    print(
        "Planning reason:",
        result.get(
            "planning_reason"
        ),
    )

    print(
        "Retrieval strategy:",
        result.get(
            "retrieval_strategy"
        ),
    )

    print(
        "Sufficiency:",
        result.get(
            "sufficiency"
        ),
    )

    print(
        "Retry count:",
        result.get(
            "retry_count"
        ),
    )

    print(
        "Escalated:",
        result.get(
            "escalated"
        ),
    )

    print(
        "\nFinal answer:\n",
        result.get(
            "final_answer"
        ),
    )

    retrieved_chunks = result.get(
        "retrieved_chunks",
        [],
    )

    print(
        "\nRetrieved chunks:",
        len(retrieved_chunks),
    )

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{chunk.source_file} "
            f"| page {chunk.page} "
            f"| section="
            f"{chunk.section_title} "
            f"| equipment="
            f"{chunk.equipment_model}"
        )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert (
        result.get(
            "request_status"
        )
        == "READY"
    )

    assert (
        result.get(
            "current_step"
        )
        == "agentic_rag"
    )

    assert (
        result.get(
            "retrieval_strategy"
        )
        == "hybrid"
    )

    assert (
        result.get(
            "sufficiency"
        )
        is True
    )

    assert (
        result.get(
            "escalated"
        )
        is False
    )

    assert retrieved_chunks

    assert result.get(
        "final_answer"
    )

    for chunk in retrieved_chunks:

        assert (
            chunk.equipment_model
            == THERMAL_STATION
        )

    print("\n" + "=" * 80)
    print("REAL RAG GRAPH TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()