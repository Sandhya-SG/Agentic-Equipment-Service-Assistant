from agentic_rag import AgenticRAGAgent


def main():

    agent = AgenticRAGAgent()

    test_questions = [

        # Tool 1 only
        "What does error code E101 mean?",

        # Tool 2 only
        "What is the current date and time?",

        # Tool 1 only
        (
            "What safety procedure should I follow "
            "if the AEM-X100 emergency stop is active?"
        ),

        # Potential multiple-tool case
        (
            "At the current date and time, create a brief "
            "evidence-based assessment of what error code "
            "E101 means."
        )
    ]

    for test_number, question in enumerate(
        test_questions,
        start=1
    ):

        print("\n\n" + "#" * 80)
        print(f"TOOL TEST {test_number}")
        print("#" * 80)

        result = agent.run_with_tools(
            question
        )

        print("\n" + "-" * 80)

        print(
            f"Tools selected: "
            f"{result['tools_used']}"
        )

        print(
            f"Total tool calls: "
            f"{result['tool_call_count']}"
        )


if __name__ == "__main__":
    main()