import os

from dotenv import load_dotenv
from openai import OpenAI

from rag import LocalRAG

load_dotenv()

class AgenticRAGAgent:

    def __init__(self):
        self.rag = LocalRAG()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.model = "gpt-4o-mini"

    def retrieve(self, query, k=5):
        return self.rag.hybrid_search(query, k=k)

    def assess_retrieval(self, query, results):
        """
        Use gpt-4o-mini to determine whether the retrieved
        evidence is sufficient to answer the engineer's question.
        """

        if not results:
            return False

        context_parts = []

        for rank, result in enumerate(results, start=1):
            context_parts.append(
                f"Result {rank}\n"
                f"Source: {result['source']}\n"
                f"Page: {result['page']}\n"
                f"Content:\n{result['text']}\n"
            )

        context = "\n".join(context_parts)

        prompt = f"""
    You are a retrieval quality evaluator for an
    equipment servicing assistant.

    Engineer question:
    {query}

    Retrieved evidence:
    {context}

    Determine whether the retrieved evidence contains
    enough relevant information to answer the engineer's
    question accurately.

    Do not judge retrieval quality from ranking scores.
    Judge whether the CONTENT actually answers the question.

    Respond with exactly one word:

    GOOD

    or

    POOR
    """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate whether retrieved equipment "
                        "service documentation adequately answers "
                        "an engineer's question."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        assessment = (
            response.choices[0]
            .message.content
            .strip()
            .upper()
        )

        print(f"LLM assessment: {assessment}")

        return assessment == "GOOD"

    def reformulate_query(self, query, results):
        """
        Use gpt-4o-mini to reformulate a query when
        the initial retrieval is insufficient.
        """

        context_parts = []

        for rank, result in enumerate(results, start=1):
            context_parts.append(
                f"{rank}. {result['source']}: "
                f"{result['text']}"
            )

        context = "\n".join(context_parts)

        prompt = f"""
    You are an equipment servicing retrieval agent.

    The original engineer question is:

    {query}

    The initial retrieval results were insufficient:

    {context}

    Rewrite the engineer's question into a better search
    query for the equipment service knowledge base.

    The new query should preserve the engineer's intent,
    but use terminology likely to appear in service manuals,
    procedures, troubleshooting guides, safety procedures,
    or escalation procedures.

    Return only the reformulated query.
    """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You reformulate equipment servicing "
                        "questions to improve document retrieval."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    def run(self, query, k=5):

        print("\n" + "=" * 80)
        print("AGENTIC RAG")
        print("=" * 80)

        print("\nOriginal question:")
        print(query)

        # ------------------------------------------------
        # Step 1: Initial retrieval
        # ------------------------------------------------

        print("\n[Step 1] Initial retrieval...")

        initial_results = self.retrieve(query, k=k)

        for rank, result in enumerate(initial_results, start=1):
            print(
                f"{rank}. "
                f"{result['source']} "
                f"| page {result['page']} "
                f"| RRF={result['rrf_score']:.4f}"
            )

        # ------------------------------------------------
        # Step 2: Assess retrieval
        # ------------------------------------------------

        print("\n[Step 2] Assess retrieval quality...")

        good = self.assess_retrieval(
            query,
            initial_results
        )

        if good:

            print("Retrieval assessment: GOOD")
            print("Returning initial results.")

            return {
                "original_query": query,
                "final_query": query,
                "retrieval_attempts": 1,
                "retrieval_status": "GOOD",
                "escalation_required": False,
                "results": initial_results
            }

        # ------------------------------------------------
        # Step 3: Reformulate query
        # ------------------------------------------------

        print("Retrieval assessment: POOR")

        print("\n[Step 3] Reformulating query...")

        new_query = self.reformulate_query(
            query,
            initial_results
        )

        print("New query:")
        print(new_query)

        # ------------------------------------------------
        # Step 4: Second retrieval
        # ------------------------------------------------

        print("\n[Step 4] Second retrieval...")

        second_results = self.retrieve(
            new_query,
            k=k
        )

        for rank, result in enumerate(second_results, start=1):
            print(
                f"{rank}. "
                f"{result['source']} "
                f"| page {result['page']} "
                f"| RRF={result['rrf_score']:.4f}"
            )

        # ------------------------------------------------
        # Step 5: Reassess second retrieval
        # ------------------------------------------------

        print("\n[Step 5] Reassess retrieval quality...")

        second_good = self.assess_retrieval(
            new_query,
            second_results
        )

        if second_good:

            print("Second retrieval assessment: GOOD")
            print("Returning second retrieval results.")

            return {
                "original_query": query,
                "final_query": new_query,
                "retrieval_attempts": 2,
                "retrieval_status": "GOOD",
                "escalation_required": False,
                "results": second_results
            }

        # ------------------------------------------------
        # Step 6: Insufficient evidence / escalation
        # ------------------------------------------------

        print("Second retrieval assessment: POOR")

        print(
            "\nInsufficient documented evidence was found "
            "after two retrieval attempts."
        )

        print(
            "The system should not invent a servicing procedure. "
            "Escalation is required."
        )

        return {
            "original_query": query,
            "final_query": new_query,
            "retrieval_attempts": 2,
            "retrieval_status": "INSUFFICIENT",
            "escalation_required": True,
            "results": second_results
        }


if __name__ == "__main__":

    agent = AgenticRAGAgent()

    while True:

        question = input(
            "\nEngineer question (or quit): "
        ).strip()

        if question.lower() in {"quit", "exit"}:
            break

        agent.run(question)