from service_graph import build_service_graph


def main():

    graph = build_service_graph()

    question = (
        "What does error code E101 mean?"
    )

    print("\n" + "=" * 80)
    print("LANGGRAPH SERVICE ASSISTANT")
    print("=" * 80)

    result = graph.invoke({
        "question": question
    })

    print("\n" + "=" * 80)
    print("FINAL GRAPH STATE")
    print("=" * 80)

    print(
        f"\nQuestion:\n"
        f"{result['question']}"
    )

    print(
        f"\nRequest type:\n"
        f"{result['request_type']}"
    )

    print(
        f"\nPlan:\n"
        f"{result['plan']}"
    )

    print(
        f"\nRetrieval status:\n"
        f"{result['retrieval_status']}"
    )

    print(
        f"\nSenior used:\n"
        f"{result['senior_used']}"
    )

    print(
        f"\nFinal response:\n"
        f"{result['final_response']}"
    )


if __name__ == "__main__":
    main()