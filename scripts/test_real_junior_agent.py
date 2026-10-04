"""Test Junior Agent explicit retrieval against real AEM manuals."""

from asa.agents.rag import AgenticRAGAgent

from asa.ingestion.metadata import (
    THERMAL_RETROFIT_1KW,
    THERMAL_STATION,
)


def run_test(
    agent,
    title,
    question,
    equipment_model,
):

    print("\n" + "#" * 80)
    print(title)
    print("#" * 80)

    print(
        f"\nQuestion:\n{question}"
    )

    print(
        f"\nEquipment:\n{equipment_model}"
    )

    result = agent.junior_search(
        question=question,
        equipment_model=equipment_model,
        k=5,
    )

    print(
        f"\nTool used:\n"
        f"{result['tool_name']}"
    )

    print(
        f"\nJunior search query:\n"
        f"{result['search_query']}"
    )

    print("\nRetrieved evidence:")

    for rank, item in enumerate(
        result["results"],
        start=1,
    ):

        print(
            f"{rank}. "
            f"{item['source_file']} "
            f"| page {item['page']} "
            f"| section="
            f"{item['section_title']} "
            f"| equipment="
            f"{item['equipment_model']} "
            f"| RRF="
            f"{item['rrf_score']:.4f}"
        )

        assert (
            item["equipment_model"]
            == equipment_model
        )


def main():

    agent = AgenticRAGAgent()

    run_test(
        agent=agent,

        title=(
            "JUNIOR TEST 1 - "
            "THERMAL STATION"
        ),

        question=(
            "What preventive maintenance "
            "should be performed?"
        ),

        equipment_model=
            THERMAL_STATION,
    )

    run_test(
        agent=agent,

        title=(
            "JUNIOR TEST 2 - "
            "1KW THERMAL RETROFIT"
        ),

        question=(
            "What safety precautions apply "
            "before servicing the equipment?"
        ),

        equipment_model=
            THERMAL_RETROFIT_1KW,
    )

    print("\n" + "=" * 80)

    print(
        "REAL JUNIOR AGENT "
        "TOOL TESTS: PASS"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()