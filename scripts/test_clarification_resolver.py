"""Smoke tests for clarification continuation."""

from asa.agents.clarification import (
    resolve_clarification,
)

from asa.agents.request import (
    request_node,
)

from asa.ingestion.metadata import (
    THERMAL_STATION,
)


def main():

    print("\n" + "#" * 80)
    print("TEST 1 - COMBINE CLARIFICATION")
    print("#" * 80)

    resolved = resolve_clarification(
        original_query=
            "It's not working.",

        clarification_response=
            (
                "The temperature is not "
                "reaching the setpoint."
            ),
    )

    print(
        f"\nResolved query:\n{resolved}"
    )

    assert (
        "It's not working."
        in resolved
    )

    assert (
        "temperature"
        in resolved.lower()
    )

    assert (
        "setpoint"
        in resolved.lower()
    )

    print("\n" + "#" * 80)
    print("TEST 2 - REQUEST RE-ENTRY")
    print("#" * 80)

    result = request_node(
        {
            "raw_query":
                "It's not working.",

            "resolved_query":
                resolved,

            "equipment_model":
                THERMAL_STATION,

            "clarification_count":
                1,
        }
    )

    print(
        "\nRequest status:",
        result.get(
            "request_status"
        ),
    )

    print(
        "Clarification:",
        result.get(
            "clarification_question"
        ),
    )

    assert (
        result.get(
            "request_status"
        )
        == "READY"
    )

    assert (
        result.get(
            "clarification_needed"
        )
        is False
    )

    print("\n" + "=" * 80)
    print(
        "CLARIFICATION RESOLVER "
        "TESTS: 2/2 PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()