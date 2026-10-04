"""Test minimal Request -> Planning -> Specialist LangGraph routing."""

from asa.graph.orchestrator import service_graph
from asa.graph.state import new_state

from asa.ingestion.metadata import (
    THERMAL_STATION,
)


TESTS = [
    {
        "name":
            "TEST 1 - RAG ROUTE",

        "question":
            "What preventive maintenance should be performed?",

        "expected_status":
            "READY",

        "expected_step":
            "agentic_rag",

        "expected_answer":
            "ROUTED_TO_AGENTIC_RAG",
    },

    {
        "name":
            "TEST 2 - DIAGNOSTIC ROUTE",

        "question":
            (
                "The equipment is not reaching "
                "the temperature setpoint. "
                "What should I check?"
            ),

        "expected_status":
            "READY",

        "expected_step":
            "diagnostic",

        "expected_answer":
            "ROUTED_TO_DIAGNOSTIC",
    },

    {
        "name":
            "TEST 3 - SAFETY ROUTE",

        "question":
            "Can I bypass the safety interlock?",

        "expected_status":
            "READY",

        "expected_step":
            "safety",

        "expected_answer":
            "ROUTED_TO_SAFETY",
    },

    {
        "name":
            "TEST 4 - SAFETY OVERRIDES DIAGNOSTIC",

        "question":
            (
                "The temperature is not increasing. "
                "Can I bypass the interlock to test it?"
            ),

        "expected_status":
            "READY",

        "expected_step":
            "safety",

        "expected_answer":
            "ROUTED_TO_SAFETY",
    },
]


def main():

    passed = 0

    # ------------------------------------------------------------------
    # READY cases
    # ------------------------------------------------------------------

    for test in TESTS:

        print("\n" + "#" * 80)
        print(test["name"])
        print("#" * 80)

        initial_state = new_state(
            raw_query=test["question"],
            equipment_model=THERMAL_STATION,
        )

        result = service_graph.invoke(
            initial_state
        )

        print(
            "\nRequest status:",
            result.get(
                "request_status"
            ),
        )

        print(
            "Planning reason:",
            result.get(
                "planning_reason"
            ),
        )

        print(
            "Current step:",
            result.get(
                "current_step"
            ),
        )

        print(
            "Final answer:",
            result.get(
                "final_answer"
            ),
        )

        assert (
            result.get(
                "request_status"
            )
            == test[
                "expected_status"
            ]
        )

        assert (
            result.get(
                "current_step"
            )
            == test[
                "expected_step"
            ]
        )

        assert (
            result.get(
                "final_answer"
            )
            == test[
                "expected_answer"
            ]
        )

        passed += 1

    # ------------------------------------------------------------------
    # CLARIFY case
    # ------------------------------------------------------------------

    print("\n" + "#" * 80)
    print(
        "TEST 5 - CLARIFY ENDS BEFORE PLANNING"
    )
    print("#" * 80)

    initial_state = new_state(
        raw_query="It's not working.",
        equipment_model=THERMAL_STATION,
    )

    result = service_graph.invoke(
        initial_state
    )

    print(
        "\nRequest status:",
        result.get(
            "request_status"
        ),
    )

    print(
        "Clarification needed:",
        result.get(
            "clarification_needed"
        ),
    )

    print(
        "Clarification question:",
        result.get(
            "clarification_question"
        ),
    )

    print(
        "Current step:",
        result.get(
            "current_step"
        ),
    )

    print(
        "Final answer:",
        result.get(
            "final_answer"
        ),
    )

    assert (
        result.get(
            "request_status"
        )
        == "CLARIFY"
    )

    assert (
        result.get(
            "clarification_needed"
        )
        is True
    )

    assert result.get(
        "clarification_question"
    )

    # Planning must never have run.
    assert (
        result.get(
            "current_step"
        )
        is None
    )

    # No specialist must have run.
    assert (
        result.get(
            "final_answer"
        )
        is None
    )

    passed += 1

    print("\n" + "=" * 80)

    print(
        f"MINIMAL GRAPH TESTS: "
        f"{passed}/5 PASS"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()