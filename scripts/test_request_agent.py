"""Smoke tests for the real AEM Request Agent."""

from asa.agents.request import request_node

from asa.ingestion.metadata import (
    THERMAL_STATION,
)


TESTS = [
    {
        "name":
            "TEST 1 - CLEAR DOCUMENTATION REQUEST",

        "state": {
            "raw_query":
                (
                    "What preventive maintenance "
                    "should be performed?"
                ),

            "equipment_model":
                THERMAL_STATION,
        },

        "expected":
            "READY",
    },

    {
        "name":
            "TEST 2 - SPECIFIC MAINTENANCE DOCUMENTATION REQUEST",

        "state": {
            "raw_query":
                (
                    "What are the semi annual "
                    "preventive maintenance checks?"
                ),

            "equipment_model":
                THERMAL_STATION,
        },

        "expected":
            "READY",
    },

    {
        "name":
            "TEST 2 - AMBIGUOUS SYMPTOM",

        "state": {
            "raw_query":
                "It's not working.",

            "equipment_model":
                THERMAL_STATION,
        },

        "expected":
            "CLARIFY",
    },

    {
        "name":
            "TEST 3 - MISSING OBJECT",

        "state": {
            "raw_query":
                "How do I replace it?",

            "equipment_model":
                THERMAL_STATION,
        },

        "expected":
            "CLARIFY",
    },

    {
        "name":
            "TEST 4 - CLEAR SAFETY REQUEST",

        "state": {
            "raw_query":
                (
                    "Can I bypass the "
                    "safety interlock?"
                ),

            "equipment_model":
                THERMAL_STATION,
        },

        "expected":
            "READY",
    },

    {
        "name":
            "TEST 5 - CLEAR DIAGNOSTIC REQUEST",

        "state": {
            "raw_query":
                (
                    "The equipment is not reaching "
                    "the temperature setpoint. "
                    "What should I check?"
                ),

            "equipment_model":
                THERMAL_STATION,
        },

        "expected":
            "READY",
    },

    {
        "name":
            "TEST 6 - NO EQUIPMENT SELECTED",

        "state": {
            "raw_query":
                (
                    "What preventive maintenance "
                    "should be performed?"
                ),

            "equipment_model":
                None,
        },

        "expected":
            "CLARIFY",
    },

    {
        "name":
            "TEST 7 - EMPTY REQUEST",

        "state": {
            "raw_query":
                "",

            "equipment_model":
                THERMAL_STATION,
        },

        "expected":
            "CLARIFY",
    },

    {
        "name":
            "TEST 8 - SAFETY INTENT OVERRIDES AMBIGUITY",

        "state": {
            "raw_query":
                (
                    "The temperature is not increasing. "
                    "Can I bypass the interlock to test it?"
                ),

            "equipment_model":
                THERMAL_STATION,
        },

        "expected":
            "READY",
    },

    {
        "name":
            "TEST 9 - POWERED PANEL ACCESS",

        "state": {
            "raw_query":
                (
                    "Can I open the panel while "
                    "the equipment is powered?"
                ),

            "equipment_model":
                THERMAL_STATION,
        },

        "expected":
            "READY",
    },
]


def main():

    passed = 0

    for test in TESTS:

        print("\n" + "#" * 80)
        print(test["name"])
        print("#" * 80)

        result = request_node(
            test["state"]
        )

        actual = result[
            "request_status"
        ]

        print(
            f"\nExpected: "
            f"{test['expected']}"
        )

        print(
            f"Actual:   {actual}"
        )

        print(
            "Clarification: "
            f"{result.get('clarification_question')}"
        )

        assert (
            actual
            == test["expected"]
        ), (
            f"{test['name']} failed: "
            f"expected {test['expected']}, "
            f"got {actual}"
        )

        if actual == "CLARIFY":

            assert result.get(
                "clarification_question"
            ), (
                "CLARIFY must provide "
                "one clarification question."
            )

        if actual == "READY":

            assert (
                result.get(
                    "clarification_question"
                )
                is None
            )

        passed += 1

    print("\n" + "=" * 80)

    print(
        f"REQUEST AGENT TESTS: "
        f"{passed}/{len(TESTS)} PASS"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()