import json

from agentic_rag import AgenticRAGAgent


def main():

    with open(
        "evaluation_agentic.json",
        "r",
        encoding="utf-8"
    ) as file:
        evaluation_cases = json.load(file)

    agent = AgenticRAGAgent()

    evaluation_results = []

    for case in evaluation_cases:

        print("\n\n" + "#" * 80)
        print(
            f"EVALUATION {case['id']} "
            f"- {case['category']}"
        )
        print("#" * 80)

        print("\nQuestion:")
        print(case["question"])

        result = agent.run(
            case["question"],
            k=5
        )

        retrieved_sources = [
            item["source"]
            for item in result["results"]
        ]

        # ----------------------------------------
        # Source evaluation
        # ----------------------------------------

        expected_source = case["expected_source"]

        if expected_source is None:
            source_pass = True
        else:
            source_pass = (
                expected_source
                in retrieved_sources
            )

        # ----------------------------------------
        # Escalation evaluation
        # ----------------------------------------

        escalation_pass = (
            result["escalation_required"]
            == case["expected_escalation"]
        )

        # ----------------------------------------
        # Tool evaluation
        # ----------------------------------------

        actual_tool = result.get(
            "tool_used"
        )

        tool_pass = (
            actual_tool
            in case["expected_tools"]
        )

        case_result = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],

            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "source_pass": source_pass,

            "expected_escalation":
                case["expected_escalation"],

            "actual_escalation":
                result["escalation_required"],

            "escalation_pass":
                escalation_pass,

            "expected_tools":
                case["expected_tools"],

            "actual_tool":
                actual_tool,

            "tool_pass":
                tool_pass,

            "retrieval_attempts":
                result["retrieval_attempts"],

            "retrieval_status":
                result["retrieval_status"],

            "senior_used":
                result["senior_used"],

            "senior_decision":
                result["senior_decision"],

            "answer":
                result["answer"]
        }

        evaluation_results.append(
            case_result
        )

    # --------------------------------------------
    # Save detailed results
    # --------------------------------------------

    with open(
        "evaluation_agentic_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_results,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------
    # Summary metrics
    # --------------------------------------------

    total = len(evaluation_results)

    source_passes = sum(
        result["source_pass"]
        for result in evaluation_results
    )

    escalation_passes = sum(
        result["escalation_pass"]
        for result in evaluation_results
    )

    tool_passes = sum(
        result["tool_pass"]
        for result in evaluation_results
    )

    senior_uses = sum(
        result["senior_used"]
        for result in evaluation_results
    )

    print("\n\n" + "=" * 80)
    print("AGENTIC RAG EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"\nSource Coverage: "
        f"{source_passes}/{total} "
        f"({source_passes / total:.2%})"
    )

    print(
        f"Tool Selection Accuracy: "
        f"{tool_passes}/{total} "
        f"({tool_passes / total:.2%})"
    )

    print(
        f"Escalation Accuracy: "
        f"{escalation_passes}/{total} "
        f"({escalation_passes / total:.2%})"
    )

    print(
        f"Senior Agent Invocations: "
        f"{senior_uses}/{total} "
        f"({senior_uses / total:.2%})"
    )

    print(
        "\nDetailed results saved to:"
        "\nevaluation_agentic_results.json"
    )


if __name__ == "__main__":
    main()