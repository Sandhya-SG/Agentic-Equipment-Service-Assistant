from asa.agents.rag import AgenticRAGAgent

from asa.ingestion.metadata import (
    THERMAL_RETROFIT_1KW,
    THERMAL_STATION,
)


TESTS = [
    {
        "name":
            "TEST 1 - DOCUMENTED MAINTENANCE",

        "question":
            "What preventive maintenance should be performed?",

        "equipment":
            THERMAL_STATION,
    },

    {
        "name":
            "TEST 2 - DOCUMENTED SAFETY",

        "question":
            (
                "What safety precautions apply "
                "before servicing the equipment?"
            ),

        "equipment":
            THERMAL_RETROFIT_1KW,
    },

    {
        "name":
            "TEST 3 - UNSUPPORTED PROCEDURE",

        "question":
            (
                "What is the procedure for replacing "
                "the main thermal controller CPU?"
            ),

        "equipment":
            THERMAL_STATION,
    },
]


def main():

    agent = AgenticRAGAgent()

    for test in TESTS:

        print("\n\n" + "#" * 80)
        print(test["name"])
        print("#" * 80)

        result = agent.run(
            question=test["question"],
            equipment_model=test["equipment"],
            k=5,
        )

        print("\n" + "=" * 80)
        print("FINAL AGENTIC RAG RESULT")
        print("=" * 80)

        print(
            "\nEquipment:",
            result["equipment_model"],
        )

        print(
            "\nRetrieval status:",
            result["retrieval_status"],
        )

        print(
            "\nRetrieval attempts:",
            result["retrieval_attempts"],
        )

        print(
            "\nSenior used:",
            result.get("senior_used"),
        )

        print(
            "\nSenior decision:",
            result.get("senior_decision"),
        )

        print(
            "\nEscalation required:",
            result.get(
                "escalation_required"
            ),
        )

        print(
            "\nGenerated answer:\n"
            f"{result['answer']}"
        )

        # Equipment isolation must always hold.
        for item in result["results"]:

            assert (
                item["equipment_model"]
                == test["equipment"]
            )

        # Unsupported requests must not receive
        # a fabricated grounded procedure.
        if (
            result["retrieval_status"]
            == "INSUFFICIENT"
        ):

            assert (
                result["escalation_required"]
                is True
            )

            assert (
                result["answer"].startswith(
                    "No grounded answer generated."
                )
            )


if __name__ == "__main__":
    main()