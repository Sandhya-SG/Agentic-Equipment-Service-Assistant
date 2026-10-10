"""Integration test for clarification -> Request -> Planning."""

from asa.agents.clarification import (
    resolve_clarification,
)
from asa.agents.planning import (
    planning_node,
)
from asa.agents.request import (
    request_node,
)
from asa.ingestion.metadata import (
    THERMAL_STATION,
)


def main():

    print("\n" + "#" * 80)
    print("V1.0.2G4 - CLARIFICATION -> REQUEST -> PLANNING")
    print("#" * 80)

    original_query = (
        "It's not working."
    )

    clarification_response = (
        "The temperature is not reaching "
        "the setpoint."
    )

    # --------------------------------------------------
    # Step 1 - prove original request needs clarification
    # --------------------------------------------------

    print(
        "\n[Step 1] Original ambiguous request"
    )

    initial_state = {
        "raw_query":
            original_query,

        "equipment_model":
            THERMAL_STATION,

        "clarification_count":
            0,
    }

    first_request = request_node(
        initial_state
    )

    print(
        "Initial request status:",
        first_request[
            "request_status"
        ],
    )

    assert (
        first_request[
            "request_status"
        ]
        == "CLARIFY"
    )

    assert (
        first_request[
            "clarification_needed"
        ]
        is True
    )

    assert first_request[
        "clarification_question"
    ]

    # --------------------------------------------------
    # Step 2 - resolve engineer clarification
    # --------------------------------------------------

    print(
        "\n[Step 2] Engineer clarification"
    )

    resolved_query = resolve_clarification(
        original_query=
            original_query,

        clarification_response=
            clarification_response,
    )

    print(
        f"Resolved query:\n"
        f"{resolved_query}"
    )

    # --------------------------------------------------
    # Step 3 - re-enter real Request Agent
    # --------------------------------------------------

    print(
        "\n[Step 3] Request Agent re-entry"
    )

    clarified_state = {
        "raw_query":
            clarification_response,

        "resolved_query":
            resolved_query,

        "clarification_response":
            clarification_response,

        "clarification_count":
            1,

        "equipment_model":
            THERMAL_STATION,
    }

    second_request = request_node(
        clarified_state
    )

    clarified_state.update(
        second_request
    )

    print(
        "Resolved request status:",
        clarified_state[
            "request_status"
        ],
    )

    assert (
        clarified_state[
            "request_status"
        ]
        == "READY"
    )

    assert (
        clarified_state[
            "clarification_needed"
        ]
        is False
    )

    assert (
        clarified_state[
            "clarification_question"
        ]
        is None
    )

    # --------------------------------------------------
    # Step 4 - send resolved state to real Planning Agent
    # --------------------------------------------------

    print(
        "\n[Step 4] Planning Agent"
    )

    planning_result = planning_node(
        clarified_state
    )

    clarified_state.update(
        planning_result
    )

    print(
        "Selected specialist:",
        clarified_state[
            "current_step"
        ],
    )

    print(
        "Planning reason:",
        clarified_state[
            "planning_reason"
        ],
    )

    # --------------------------------------------------
    # Final assertions
    # --------------------------------------------------

    assert (
        clarified_state[
            "current_step"
        ]
        == "diagnostic"
    )

    assert (
        clarified_state[
            "plan"
        ]
        == ["diagnostic"]
    )

    assert (
        clarified_state[
            "iteration_count"
        ]
        == 1
    )

    # Provenance must remain intact.
    assert (
        clarified_state[
            "raw_query"
        ]
        == clarification_response
    )

    assert (
        clarified_state[
            "clarification_response"
        ]
        == clarification_response
    )

    assert (
        original_query
        in clarified_state[
            "resolved_query"
        ]
    )

    assert (
        clarification_response
        in clarified_state[
            "resolved_query"
        ]
    )

    assert (
        clarified_state[
            "clarification_count"
        ]
        == 1
    )

    print("\n" + "=" * 80)
    print(
        "V1.0.2G4 CLARIFICATION PLANNING "
        "INTEGRATION: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()