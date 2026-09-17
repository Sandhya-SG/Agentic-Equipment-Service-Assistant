from agentic_rag import AgenticRAGAgent


def main():

    agent = AgenticRAGAgent()

    test_questions = [
        # Test 1: Out-of-knowledge-base / difficult case
        "How do I replace the AEM-X100 vacuum pump?",

        # Test 2: Ambiguous troubleshooting case
        "The wafer is not moving properly. What should I do?",

        # Test 3: Strong known-answer case
        "What does error code E101 mean?"
    ]

    for test_number, question in enumerate(
        test_questions,
        start=1
    ):

        print("\n\n" + "#" * 80)
        print(f"TEST {test_number}")
        print("#" * 80)

        result = agent.run(
            question,
            k=5
        )

        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)

        print(
            f"\nOriginal query:\n"
            f"{result['original_query']}"
        )

        print(
            f"\nFinal query:\n"
            f"{result['final_query']}"
        )

        print(
            f"\nRetrieval attempts: "
            f"{result['retrieval_attempts']}"
        )

        print(
            f"Retrieval tool: "
            f"{result.get('tool_used')}"
        )

        print(
            f"Initial search query: "
            f"{result.get('initial_search_query')}"
        )

        print(
            f"\nRetrieval status: "
            f"{result['retrieval_status']}"
        )

        print(
            f"Escalation required: "
            f"{result['escalation_required']}"
        )

        print(
            f"\nSenior Agent used: "
            f"{result['senior_used']}"
        )

        if result["senior_used"]:

            print(
                f"Senior decision: "
                f"{result['senior_decision']}"
            )

            print(
                f"Senior reason: "
                f"{result['senior_reason']}"
            )

        print("\nGenerated answer:")

        if result["answer"]:
            print(result["answer"])
        else:
            print(
                "No grounded answer generated. "
                "Escalation required."
            )

        print("\nFinal retrieved documents:")

        for rank, item in enumerate(
            result["results"],
            start=1
        ):
            print(
                f"{rank}. "
                f"{item['source']} "
                f"| page {item['page']} "
                f"| RRF={item['rrf_score']:.4f}"
            )


if __name__ == "__main__":
    main()