import json


def main():

    with open(
        "evaluation_agentic_results.json",
        "r",
        encoding="utf-8"
    ) as file:
        results = json.load(file)

    with open(
        "evaluation_ground_truth.json",
        "r",
        encoding="utf-8"
    ) as file:
        ground_truth = json.load(file)

    result_map = {
        item["id"]: item
        for item in results
    }

    review_cases = []

    for truth in ground_truth:

        case_id = truth["id"]

        result = result_map[case_id]

        review_cases.append({
            "id": case_id,
            "question": result["question"],

            "answer": result["answer"],

            "retrieved_sources":
                result["retrieved_sources"],

            "required_facts":
                truth["required_facts"],

            "forbidden_facts":
                truth["forbidden_facts"],

            "required_sources":
                truth["required_sources"],

            "safety_expectation":
                truth["safety_expectation"],

            "expected_answer":
                truth["expected_answer"],

            "actual_escalation":
                result["actual_escalation"],

            # Human review fields
            "correctness_score": None,
            "faithfulness_score": None,
            "citation_score": None,
            "safety_pass": None,
            "review_notes": ""
        })

    with open(
        "phase2_review.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            review_cases,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "Phase 2 review file created:"
        "\nphase2_review.json"
    )


if __name__ == "__main__":
    main()