"""Smoke tests for real AEM hybrid retrieval."""

from asa.ingestion.metadata import (
    THERMAL_RETROFIT_1KW,
    THERMAL_STATION,
)

from asa.ingestion.retrieval import (
    AEMHybridRetriever,
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
            f"{result.source_file} "
            f"| page {result.page} "
            f"| section={result.section_title} "
            f"| equipment={result.equipment_model} "
            f"| RRF={result.score:.4f}"
        )

        print(
            result.text[:300]
        )

        print()


def assert_equipment(
    results,
    expected_equipment,
):

    assert results, (
        "No retrieval results returned."
    )

    for result in results:

        assert (
            result.equipment_model
            == expected_equipment
        ), (
            "Cross-equipment retrieval detected: "
            f"expected={expected_equipment}, "
            f"actual={result.equipment_model}, "
            f"chunk={result.chunk_id}"
        )


def main():

    retriever = (
        AEMHybridRetriever()
    )

    # ----------------------------------------
    # Test 1: Thermal Station
    # ----------------------------------------

    station_results = (
        retriever.hybrid_search(
            query=(
                "What should be checked "
                "before servicing the equipment?"
            ),
            equipment_model=
                THERMAL_STATION,
            k=5,
        )
    )

    assert_equipment(
        station_results,
        THERMAL_STATION,
    )

    print_results(
        "TEST 1 - THERMAL STATION",
        station_results,
    )

    # ----------------------------------------
    # Test 2: 1kW Thermal Retrofit
    # ----------------------------------------

    retrofit_results = (
        retriever.hybrid_search(
            query=(
                "What should be checked "
                "before servicing the equipment?"
            ),
            equipment_model=
                THERMAL_RETROFIT_1KW,
            k=5,
        )
    )

    assert_equipment(
        retrofit_results,
        THERMAL_RETROFIT_1KW,
    )

    print_results(
        "TEST 2 - 1KW THERMAL RETROFIT",
        retrofit_results,
    )

    print("\n" + "=" * 80)

    print(
        "EQUIPMENT ISOLATION: PASS"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()