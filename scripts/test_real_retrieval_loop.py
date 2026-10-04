from asa.agents.rag import AgenticRAGAgent

from asa.ingestion.metadata import (
    THERMAL_RETROFIT_1KW,
    THERMAL_STATION,
)


TESTS = [
    {
        "name":
            "TEST 1 - DIRECT DOCUMENTED QUESTION",

        "question":
            "What preventive maintenance should be performed?",

        "equipment":
            THERMAL_STATION,
    },

    {
        "name":
            "TEST 2 - SAFETY SERVICING QUESTION",

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
            "TEST 3 - LIKELY UNSUPPORTED PROCEDURE",

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

        result = agent.run_retrieval_loop(
            question=test["question"],
            equipment_model=test["equipment"],
            k=5,
        )

        print("\nFINAL RETRIEVAL RESULT")
        print("-" * 80)

        print(
            "Status:",
            result["retrieval_status"],
        )

        print(
            "Attempts:",
            result["retrieval_attempts"],
        )

        print(
            "Senior used:",
            result.get(
                "senior_used",
                False,
            ),
        )

        print(
            "Senior decision:",
            result.get(
                "senior_decision"
            ),
        )

        print(
            "Senior reason:",
            result.get(
                "senior_reason"
            ),
        )

        print(
            "Escalation required:",
            result.get(
                "escalation_required",
                False,
            ),
        )

        print(
            "Final query:",
            result["final_query"],
        )

        print("\nFinal evidence:")

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
                f"{item['equipment_model']}"
            )

            assert (
                item["equipment_model"]
                == test["equipment"]
            )


if __name__ == "__main__":
    main()