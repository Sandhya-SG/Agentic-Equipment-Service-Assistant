"""Integration test for the real Diagnostic LangGraph branch."""

from asa.graph.orchestrator import service_graph
from asa.graph.state import new_state

from asa.ingestion.metadata import (
    THERMAL_STATION,
)


def main():

    print("\n" + "#" * 80)
    print("REAL DIAGNOSTIC GRAPH TEST")
    print("#" * 80)

    initial_state = new_state(
        raw_query=(
            "The equipment is not reaching "
            "the temperature setpoint. "
            "What should I check?"
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

    root_causes = result.get(
        "root_causes",
        [],
    )

    troubleshooting_steps = result.get(
        "troubleshooting_steps",
        [],
    )

    retrieved_chunks = result.get(
        "retrieved_chunks",
        [],
    )

    print(
        "\nCandidate causes:",
        len(root_causes),
    )

    for rank, cause in enumerate(
        root_causes,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{cause.cause} "
            f"| likelihood="
            f"{cause.likelihood:.2f} "
            f"| evidence="
            f"{cause.supporting_chunk_ids}"
        )

    print(
        "\nTroubleshooting steps:",
        len(troubleshooting_steps),
    )

    for step in troubleshooting_steps:

        print(
            f"{step.order}. "
            f"{step.action} "
            f"| hazard="
            f"{step.hazard_flag} "
            f"| evidence="
            f"{step.supporting_chunk_ids}"
        )

    print(
        "\nRetrieved chunks:",
        len(retrieved_chunks),
    )

    print("\n" + "=" * 80)
    print("RETRIEVED EVIDENCE CONTENT")
    print("=" * 80)

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        print(
            f"\n[EVIDENCE {rank}]"
        )

        print(
            f"Chunk ID: {chunk.chunk_id}"
        )

        print(
            f"Source: {chunk.source_file}"
        )

        print(
            f"Page: {chunk.page}"
        )

        print(
            f"Section: {chunk.section_title}"
        )

        print(
            f"Text:\n{chunk.text}"
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
        == "diagnostic"
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

    # Every diagnostic claim must point back
    # to retrieved evidence.
    retrieved_ids = {
        chunk.chunk_id
        for chunk in retrieved_chunks
    }

    for cause in root_causes:

        assert (
            cause.supporting_chunk_ids
        )

        assert set(
            cause.supporting_chunk_ids
        ).issubset(
            retrieved_ids
        )

    for step in troubleshooting_steps:

        assert (
            step.supporting_chunk_ids
        )

        assert set(
            step.supporting_chunk_ids
        ).issubset(
            retrieved_ids
        )

    print("\n" + "=" * 80)
    print(
        "REAL DIAGNOSTIC GRAPH TEST: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()