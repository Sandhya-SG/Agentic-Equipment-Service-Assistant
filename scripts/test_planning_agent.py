"""Smoke tests for the AEM Planning Agent."""

from asa.agents.planning import planning_node
from asa.ingestion.metadata import THERMAL_STATION


# ---------------------------------------------------------------------------
# Tests 1-7
# ---------------------------------------------------------------------------

TESTS = [
    {
        "name": "TEST 1 - DOCUMENTATION",
        "question": (
            "What preventive maintenance should be performed?"
        ),
        "expected": "agentic_rag",
    },
    {
        "name": "TEST 2 - ERROR MEANING",
        "question": (
            "What does error code E101 mean?"
        ),
        "expected": "agentic_rag",
    },
    {
        "name": "TEST 3 - DIAGNOSTIC SYMPTOM",
        "question": (
            "The equipment is not reaching "
            "the temperature setpoint. "
            "What should I check?"
        ),
        "expected": "diagnostic",
    },
    {
        "name": "TEST 4 - SAFETY INTERLOCK",
        "question": (
            "Can I bypass the safety interlock?"
        ),
        "expected": "safety",
    },
    {
        "name": "TEST 5 - EMERGENCY STOP",
        "question": (
            "The emergency stop is active. "
            "What should I do?"
        ),
        "expected": "safety",
    },
    {
        "name": "TEST 6 - SAFETY OVERRIDES DIAGNOSTIC",
        "question": (
            "The temperature is not increasing. "
            "Can I bypass the interlock to test it?"
        ),
        "expected": "safety",
    },
    {
        "name": "TEST 7 - SAFETY SERVICING QUESTION",
        "question": (
            "What safety precautions apply "
            "before servicing the equipment?"
        ),
        "expected": "safety",
    },
]


def main():

    passed = 0

    # ------------------------------------------------------------------
    # Tests 1-7
    # ------------------------------------------------------------------

    for test in TESTS:

        print("\n" + "#" * 80)
        print(test["name"])
        print("#" * 80)

        state = {
            "raw_query":
                test["question"],

            "equipment_model":
                THERMAL_STATION,

            "request_status":
                "READY",

            "iteration_count":
                0,
        }

        result = planning_node(
            state
        )

        selected = result[
            "current_step"
        ]

        print(
            f"\nExpected: "
            f"{test['expected']}"
        )

        print(
            f"Actual:   {selected}"
        )

        print(
            "Reason:   "
            f"{result['planning_reason']}"
        )

        assert (
            selected
            == test["expected"]
        ), (
            f"{test['name']} failed: "
            f"expected {test['expected']}, "
            f"got {selected}"
        )

        assert (
            result["plan"]
            == [selected]
        )

        assert (
            result["iteration_count"]
            == 1
        )

        passed += 1

    # ------------------------------------------------------------------
    # Test 8
    # CLARIFY must never reach Planning Agent
    # ------------------------------------------------------------------

    print("\n" + "#" * 80)
    print(
        "TEST 8 - CLARIFY MUST NOT REACH PLANNER"
    )
    print("#" * 80)

    try:

        planning_node(
            {
                "raw_query":
                    "It's not working.",

                "equipment_model":
                    THERMAL_STATION,

                "request_status":
                    "CLARIFY",

                "iteration_count":
                    0,
            }
        )

    except ValueError as exc:

        print(
            f"Expected rejection: {exc}"
        )

        passed += 1

    else:

        raise AssertionError(
            "Planning Agent accepted a "
            "CLARIFY request."
        )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    total_tests = (
        len(TESTS) + 1
    )

    print("\n" + "=" * 80)

    print(
        f"PLANNING AGENT TESTS: "
        f"{passed}/{total_tests} PASS"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()