"""Smoke tests for the real AEM Agentic RAG retrieval tool."""

from asa.agents.rag import AgenticRAGAgent

from asa.ingestion.metadata import (
    THERMAL_RETROFIT_1KW,
    THERMAL_STATION,
)


def print_results(
    title,
    results,
):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{result['source_file']} "
            f"| page {result['page']} "
            f"| section="
            f"{result['section_title']} "
            f"| equipment="
            f"{result['equipment_model']} "
            f"| RRF="
            f"{result['rrf_score']:.4f}"
        )


def assert_equipment(
    results,
    expected_equipment,
):

    assert results

    for result in results:

        assert (
            result["equipment_model"]
            == expected_equipment
        )


def main():

    agent = AgenticRAGAgent()

    # --------------------------------------------------
    # Test 1 — Thermal Station
    # --------------------------------------------------

    station_results = agent.execute_tool(
        "search_service_documents",
        {
            "query":
                "preventive maintenance servicing",

            "equipment_model":
                THERMAL_STATION,

            "k": 5,
        },
    )

    assert_equipment(
        station_results,
        THERMAL_STATION,
    )

    print_results(
        "TOOL TEST 1 - THERMAL STATION",
        station_results,
    )

    # --------------------------------------------------
    # Test 2 — 1kW Thermal Retrofit
    # --------------------------------------------------

    retrofit_results = agent.execute_tool(
        "search_service_documents",
        {
            "query":
                "preventive maintenance servicing",

            "equipment_model":
                THERMAL_RETROFIT_1KW,

            "k": 5,
        },
    )

    assert_equipment(
        retrofit_results,
        THERMAL_RETROFIT_1KW,
    )

    print_results(
        "TOOL TEST 2 - 1KW THERMAL RETROFIT",
        retrofit_results,
    )

    # --------------------------------------------------
    # Test 3 — equipment model must be mandatory
    # --------------------------------------------------

    try:

        agent.execute_tool(
            "search_service_documents",
            {
                "query":
                    "preventive maintenance"
            },
        )

    except ValueError as exc:

        print(
            "\nTOOL TEST 3 - MISSING EQUIPMENT MODEL"
        )

        print(
            f"Expected rejection: {exc}"
        )

    else:

        raise AssertionError(
            "Tool accepted a search without "
            "equipment_model."
        )

    print("\n" + "=" * 80)

    print(
        "REAL AEM RAG TOOL TESTS: PASS"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()