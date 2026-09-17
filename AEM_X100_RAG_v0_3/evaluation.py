from rag import LocalRAG


# ============================================================
# Evaluation Dataset
# ============================================================

EVALUATION_DATASET = [

    {
        "id": "Q001",
        "question": "The wafer is not detected after loading. What should I check?",
        "expected_sources": [
            "wafer_loading_procedure.pdf",
            "sensor_troubleshooting.pdf",
            "error_code_reference.pdf",
        ],
    },

    {
        "id": "Q002",
        "question": "What does error E101 mean?",
        "expected_sources": [
            "error_code_reference.pdf",
        ],
    },

    {
        "id": "Q003",
        "question": "Can I open the panel while the equipment is powered?",
        "expected_sources": [
            "safety_procedure.pdf",
        ],
    },

    {
        "id": "Q004",
        "question": "What are the monthly preventive maintenance activities?",
        "expected_sources": [
            "preventive_maintenance.pdf",
        ],
    },

    {
        "id": "Q005",
        "question": "What should happen if the required procedure is not found?",
        "expected_sources": [
            "escalation_procedure.pdf",
        ],
    },
]


# ============================================================
# Source Coverage
# ============================================================

def calculate_source_coverage(expected_sources, retrieved_sources):

    expected = set(expected_sources)
    retrieved = set(retrieved_sources)

    matched = expected.intersection(retrieved)

    if len(expected) == 0:
        return 0.0

    return len(matched) / len(expected)


# ============================================================
# Recall@K Evaluation
# ============================================================

def evaluate_recall_at_k(rag, k):

    total_questions = len(EVALUATION_DATASET)

    passed_questions = 0

    print()
    print("=" * 80)
    print(f"Retrieval Evaluation - Recall@{k}")
    print("=" * 80)

    for item in EVALUATION_DATASET:

        question_id = item["id"]

        question = item["question"]

        expected_sources = item["expected_sources"]


        # ----------------------------------------------------
        # Run Hybrid RAG
        # ----------------------------------------------------

        results = rag.hybrid_search(
            question,
            k=k
        )


        # ----------------------------------------------------
        # Extract retrieved sources
        # ----------------------------------------------------

        retrieved_sources = [
            result["source"]
            for result in results
        ]


        # ----------------------------------------------------
        # Calculate Recall@K
        #
        # PASS if at least one expected source
        # appears in the retrieved results.
        # ----------------------------------------------------

        hit = any(
            source in expected_sources
            for source in retrieved_sources
        )


        if hit:
            passed_questions += 1


        # ----------------------------------------------------
        # Calculate Source Coverage
        # ----------------------------------------------------

        coverage = calculate_source_coverage(
            expected_sources,
            retrieved_sources
        )


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        status = "PASS" if hit else "FAIL"

        print()
        print(f"{question_id}: {status}")

        print(f"Question:")
        print(f"  {question}")

        print()
        print("Expected sources:")
        for source in expected_sources:
            print(f"  - {source}")

        print()
        print("Retrieved sources:")

        for rank, result in enumerate(results, start=1):

            print(
                f"  {rank}. "
                f"{result['source']} "
                f"| page {result['page']} "
                f"| RRF={result['rrf_score']:.4f}"
            )

        print()
        print(f"Source Coverage@{k}: {coverage:.2%}")

        print("-" * 80)


    # ========================================================
    # Overall Recall
    # ========================================================

    recall = passed_questions / total_questions


    print()
    print("=" * 80)
    print(f"Recall@{k}: {recall:.2%}")
    print(
        f"Passed questions: "
        f"{passed_questions}/{total_questions}"
    )
    print("=" * 80)


    return recall


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Create RAG system
    # --------------------------------------------------------

    print("Loading Local RAG...")

    rag = LocalRAG()

    print("RAG loaded successfully.")


    # --------------------------------------------------------
    # Evaluate Recall@1
    # --------------------------------------------------------

    recall_1 = evaluate_recall_at_k(
        rag,
        k=1
    )


    # --------------------------------------------------------
    # Evaluate Recall@3
    # --------------------------------------------------------

    recall_3 = evaluate_recall_at_k(
        rag,
        k=3
    )


    # --------------------------------------------------------
    # Evaluate Recall@5
    # --------------------------------------------------------

    recall_5 = evaluate_recall_at_k(
        rag,
        k=5
    )


    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------

    print()
    print()
    print("=" * 80)
    print("FINAL RETRIEVAL EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Recall@1 : {recall_1:.2%}")
    print(f"Recall@3 : {recall_3:.2%}")
    print(f"Recall@5 : {recall_5:.2%}")

    print("=" * 80)