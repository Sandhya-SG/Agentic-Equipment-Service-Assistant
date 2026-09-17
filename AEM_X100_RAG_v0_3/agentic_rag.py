import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

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

        # Junior Agent
        self.model = "gpt-4o-mini"

        # Senior Agent
        self.senior_model = "gpt-5.4-mini"

    def retrieve(self, query, k=5):
        return self.rag.hybrid_search(query, k=k)

    def search_service_documents(self, query, k=5):
        """
        Tool: Search the local AEM-X100 service knowledge base.

        Uses the existing LocalRAG hybrid retrieval system.
        """

        results = self.rag.hybrid_search(
            query,
            k=k
        )

        tool_results = []

        for result in results:
            tool_results.append({
                "source": result["source"],
                "page": result["page"],
                "section": result["section"],
                "text": result["text"],
                "rrf_score": result["rrf_score"]
            })

        return tool_results

    def get_current_datetime(self):
        """
        Tool: Return the current Singapore date and time.

        Used when the agent genuinely needs a timestamp,
        for example for an escalation or audit record.
        """

        now = datetime.now(
            ZoneInfo("Asia/Singapore")
        )

        return {
            "datetime": now.isoformat(),
            "timezone": "Asia/Singapore"
        }

    def get_tool_definitions(self):
        """
        Define tools available to the Junior Agent.
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": "search_service_documents",
                    "description": (
                        "Search the local AEM-X100 equipment "
                        "service knowledge base for documented "
                        "procedures, troubleshooting information, "
                        "error codes, safety guidance, and "
                        "escalation information."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Search query describing the "
                                    "equipment servicing information "
                                    "required."
                                )
                            },
                            "k": {
                                "type": "integer",
                                "description": (
                                    "Number of retrieval results "
                                    "to return."
                                ),
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_datetime",
                    "description": (
                        "Get the current date and time in "
                        "Singapore. Use this only when the task "
                        "requires the current time or a timestamp."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False
                    }
                }
            }
        ]

    def execute_tool(self, tool_name, arguments):
        """
        Execute a tool requested by the Junior Agent.
        """

        if tool_name == "search_service_documents":

            query = arguments["query"]
            k = arguments.get("k", 5)

            return self.search_service_documents(
                query=query,
                k=k
            )

        if tool_name == "get_current_datetime":

            return self.get_current_datetime()

        raise ValueError(
            f"Unknown tool requested: {tool_name}"
        )

    def junior_select_retrieval_tool(self, question, k=5):
        """
        Junior Agent explicitly selects and calls the
        service-document retrieval tool.

        Returns the search query chosen by the LLM,
        retrieved results, and tool-use metadata.
        """

        print("\n[Tool Selection] Junior Agent deciding retrieval action...")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Junior Agent for an AEM-X100 "
                    "equipment servicing assistant. "

                    "Any question about equipment servicing, procedures, "
                    "troubleshooting, error codes, safety, maintenance, "
                    "escalation, missing procedures, unavailable "
                    "instructions, or what an engineer should do must "
                    "be grounded in the equipment knowledge base. "

                    "For such questions, you MUST use the "
                    "search_service_documents tool before answering "
                    "or deciding that documented evidence is unavailable. "

                    "Do not assume that a requested procedure or piece "
                    "of information is absent without first searching "
                    "the knowledge base. "

                    "Do not answer equipment-policy, safety, servicing, "
                    "maintenance, or escalation questions from general "
                    "knowledge. "

                    "When calling search_service_documents, formulate "
                    "a concise search query that preserves the engineer's "
                    "intent and uses terminology likely to appear in "
                    "service procedures, troubleshooting guides, safety "
                    "documents, maintenance documents, or escalation "
                    "procedures."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]

        # Only expose the RAG tool here.
        # Time is not relevant to retrieval.
        tools = [
            tool
            for tool in self.get_tool_definitions()
            if tool["function"]["name"]
            == "search_service_documents"
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {
                    "name": "search_service_documents"
                }
            },
            temperature=0
        )

        message = response.choices[0].message

        if not message.tool_calls:
            print(
                "Junior Agent did not request "
                "service-document retrieval."
            )

            return {
                "tool_used": False,
                "tool_name": None,
                "search_query": question,
                "results": []
            }

        tool_call = message.tool_calls[0]

        tool_name = tool_call.function.name

        arguments = json.loads(
            tool_call.function.arguments
        )

        search_query = arguments.get(
            "query",
            question
        )

        # Keep retrieval size controlled by our application.
        arguments["k"] = k

        print(
            f"Tool requested: {tool_name}"
        )

        print(
            f"Search query selected by Junior Agent: "
            f"{search_query}"
        )

        results = self.execute_tool(
            tool_name,
            arguments
        )

        print(
            f"Tool executed: {tool_name}"
        )

        return {
            "tool_used": True,
            "tool_name": tool_name,
            "search_query": search_query,
            "results": results
        }

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

    def generate_answer(self, query, results):
        """
        Generate an engineer-facing answer using only
        the retrieved equipment documentation.
        """

        context_parts = []

        for rank, result in enumerate(results, start=1):
            context_parts.append(
                f"[Source {rank}]\n"
                f"Document: {result['source']}\n"
                f"Page: {result['page']}\n"
                f"Content:\n{result['text']}\n"
            )

        context = "\n".join(context_parts)

        prompt = f"""
    You are a junior equipment servicing assistant.

    Engineer question:
    {query}

    Retrieved equipment documentation:
    {context}

    Answer the engineer's question using ONLY the retrieved
    documentation above.

    Rules:
    1. Do not use outside knowledge.
    2. Do not invent servicing procedures, specifications,
    safety instructions, or equipment behavior.
    3. If the documents contain conflicting information,
    explicitly state that there is a conflict.
    4. Keep the response concise and practical.
    5. Cite supporting evidence using the document filename
    and page number.
    6. Do not claim that an action is safe unless the retrieved
    documentation supports it.

    Return a response in this format:

    Answer:
    <grounded answer>

    Sources:
    - <document>, page <page>
    """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a junior field-service assistant. "
                        "You must answer only from retrieved "
                        "equipment documentation."
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

    def senior_review(
        self,
        original_query,
        first_results,
        junior_query,
        second_results
    ):
        """
        Senior Agent reviews the failed Junior Agent retrieval
        attempts and decides whether another retrieval strategy
        is justified or human escalation is required.

        The Senior Agent must NOT answer the servicing question
        from its own knowledge.
        """

        first_context = []

        for rank, result in enumerate(first_results, start=1):
            first_context.append(
                f"{rank}. "
                f"{result['source']} "
                f"(page {result['page']}):\n"
                f"{result['text']}\n"
            )

        second_context = []

        for rank, result in enumerate(second_results, start=1):
            second_context.append(
                f"{rank}. "
                f"{result['source']} "
                f"(page {result['page']}):\n"
                f"{result['text']}\n"
            )

        first_context = "\n".join(first_context)
        second_context = "\n".join(second_context)

        prompt = f"""
    You are the Senior Retrieval Agent for an equipment
    servicing assistant.

    A Junior Agent attempted to retrieve documented evidence
    for an engineer's question twice, but judged both retrieval
    attempts insufficient.

    Original engineer question:
    {original_query}

    FIRST RETRIEVAL:
    {first_context}

    The Junior Agent then reformulated the question as:
    {junior_query}

    SECOND RETRIEVAL:
    {second_context}

    Your task is NOT to answer the engineer's servicing
    question from your own knowledge.

    Instead, review the Junior Agent's retrieval attempts and
    make one decision:

    RETRY
    Use RETRY only if a materially different search query is
    likely to locate relevant information that may exist in the
    equipment knowledge base.

    ESCALATE
    Use ESCALATE if the available evidence suggests that the
    required procedure or information is not documented, or
    if another search would merely repeat the same retrieval.

    Important rules:
    - Never invent a servicing procedure.
    - Never use outside equipment knowledge as evidence.
    - Do not assume a document exists merely because the
    engineer asks about it.
    - Prefer ESCALATE when the required documented evidence
    appears unavailable.

    Return exactly this format:

    DECISION: RETRY or ESCALATE
    REASON: <brief reason>
    QUERY: <new search query if RETRY, otherwise NONE>
    """

        response = self.client.chat.completions.create(
            model=self.senior_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the senior retrieval and "
                        "cross-reflection agent. Review failed "
                        "retrieval attempts without inventing "
                        "equipment knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        review = response.choices[0].message.content.strip()

        return review

    def parse_senior_review(self, review):
        """
        Parse the structured Senior Agent response.
        """

        decision = "ESCALATE"
        reason = ""
        query = None

        for line in review.splitlines():

            line = line.strip()

            if line.startswith("DECISION:"):
                value = line.split(":", 1)[1].strip().upper()

                if value in {"RETRY", "ESCALATE"}:
                    decision = value

            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

            elif line.startswith("QUERY:"):
                value = line.split(":", 1)[1].strip()

                if value.upper() != "NONE":
                    query = value

        # Safety fallback
        if decision == "RETRY" and not query:
            decision = "ESCALATE"

        return {
            "decision": decision,
            "reason": reason,
            "query": query
        }

    def run(self, query, k=5):

        print("\n" + "=" * 80)
        print("AGENTIC RAG")
        print("=" * 80)

        print("\nOriginal question:")
        print(query)

        # ------------------------------------------------
        # Step 1: Initial retrieval
        # ------------------------------------------------

        print("\n[Step 1] Junior Agent tool-based retrieval...")

        initial_tool_call = self.junior_select_retrieval_tool(
            query,
            k=k
        )

        initial_results = initial_tool_call["results"]

        if not initial_results:

            print(
                "\nNo service-document retrieval "
                "was performed."
            )

            return {
                "original_query": query,
                "final_query": query,
                "retrieval_attempts": 0,
                "retrieval_status": "NO_RETRIEVAL",
                "escalation_required": True,
                "senior_used": False,
                "senior_decision": None,
                "senior_reason": None,
                "tool_used": initial_tool_call["tool_name"],
                "answer": None,
                "results": []
            }

        initial_search_query = (
            initial_tool_call["search_query"]
        )

        print(
            f"Tool used: "
            f"{initial_tool_call['tool_name']}"
        )

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

            print("\n[Step 3] Generating grounded answer...")

            answer = self.generate_answer(
                query,
                initial_results
            )

            print("\nGrounded answer:")
            print(answer)

            return {
                "original_query": query,
                "final_query": query,
                "retrieval_attempts": 1,
                "retrieval_status": "GOOD",
                "escalation_required": False,
                "senior_used": False,
                "senior_decision": None,
                "senior_reason": None,
                "tool_used": initial_tool_call["tool_name"],
                "initial_search_query": initial_search_query,
                "answer": answer,
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

        print("\n[Step 4] Second tool-based retrieval...")

        second_results = self.execute_tool(
            "search_service_documents",
            {
                "query": new_query,
                "k": k
            }
        )

        print(
            "Tool executed: "
            "search_service_documents"
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

            print("\n[Step 6] Generating grounded answer...")

            answer = self.generate_answer(
                query,
                second_results
            )

            print("\nGrounded answer:")
            print(answer)

            return {
                "original_query": query,
                "final_query": new_query,
                "retrieval_attempts": 2,
                "retrieval_status": "GOOD",
                "escalation_required": False,
                "senior_used": False,
                "senior_decision": None,
                "senior_reason": None,
                "tool_used": "search_service_documents",
                "initial_search_query": initial_search_query,
                "answer": answer,
                "results": second_results
            }

        print("Second retrieval assessment: POOR")

        # ------------------------------------------------
        # Step 6: Escalate to Senior Agent
        # ------------------------------------------------

        print("\n[Step 6] Escalating to Senior Agent...")
        print(f"Senior model: {self.senior_model}")

        senior_raw = self.senior_review(
            original_query=query,
            first_results=initial_results,
            junior_query=new_query,
            second_results=second_results
        )

        print("\nSenior Agent review:")
        print(senior_raw)

        senior = self.parse_senior_review(senior_raw)

        print(
            f"\nSenior decision: "
            f"{senior['decision']}"
        )

        print(
            f"Senior reason: "
            f"{senior['reason']}"
        )

        if senior["decision"] == "ESCALATE":

            print(
                "\nSenior Agent determined that another "
                "retrieval attempt is not justified."
            )

            print(
                "Human escalation is required."
            )

            return {
                "original_query": query,
                "final_query": new_query,
                "retrieval_attempts": 2,
                "retrieval_status": "INSUFFICIENT",
                "escalation_required": True,
                "senior_used": True,
                "senior_decision": "ESCALATE",
                "senior_reason": senior["reason"],
                "tool_used": initial_tool_call["tool_name"],
                "initial_search_query": initial_search_query,
                "answer": None,
                "results": second_results
            }

        # ------------------------------------------------
        # Step 7: Senior-directed retrieval
        # ------------------------------------------------

        senior_query = senior["query"]

        print("\n[Step 7] Senior-directed retrieval...")

        print("Senior query:")
        print(senior_query)

        senior_results = self.execute_tool(
            "search_service_documents",
            {
                "query": senior_query,
                "k": k
            }
        )

        print(
            "Tool executed: "
            "search_service_documents"
        )

        for rank, result in enumerate(senior_results, start=1):
            print(
                f"{rank}. "
                f"{result['source']} "
                f"| page {result['page']} "
                f"| RRF={result['rrf_score']:.4f}"
            )

        # ------------------------------------------------
        # Step 8: Evaluate Senior retrieval
        # ------------------------------------------------

        print(
            "\n[Step 8] Evaluating "
            "Senior-directed retrieval..."
        )

        senior_good = self.assess_retrieval(
            senior_query,
            senior_results
        )

        if senior_good:

            print(
                "Senior-directed retrieval "
                "assessment: GOOD"
            )

            print(
                "\n[Step 9] Generating grounded answer..."
            )

            answer = self.generate_answer(
                query,
                senior_results
            )

            print("\nGrounded answer:")
            print(answer)

            return {
                "original_query": query,
                "final_query": senior_query,
                "retrieval_attempts": 3,
                "retrieval_status": "GOOD",
                "escalation_required": False,
                "senior_used": True,
                "senior_decision": "RETRY",
                "senior_reason": senior["reason"],
                "tool_used": initial_tool_call["tool_name"],
                "initial_search_query": initial_search_query,
                "answer": answer,
                "results": senior_results
            }

        print(
            "Senior-directed retrieval "
            "assessment: POOR"
        )

        print(
            "\nNo sufficient documented evidence was "
            "found after Senior Agent review."
        )

        print(
            "Human escalation is required."
        )

        return {
            "original_query": query,
            "final_query": senior_query,
            "retrieval_attempts": 3,
            "retrieval_status": "INSUFFICIENT",
            "escalation_required": True,
            "senior_used": True,
            "senior_decision": "RETRY_FAILED",
            "senior_reason": senior["reason"],
            "answer": None,
            "results": senior_results
        }

    def run_with_tools(self, question):
        """
        Full Junior Agent tool-calling loop.

        The Junior Agent may dynamically call one or more
        available tools. Tool results are returned to the
        model until the model decides that no further tool
        calls are required.
        """

        print("\n" + "=" * 80)
        print("JUNIOR AGENT - FULL TOOL LOOP")
        print("=" * 80)

        print("\nQuestion:")
        print(question)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Junior Agent for an AEM-X100 "
                    "equipment servicing assistant. "

                    "Use the available tools when required. "

                    "For equipment-specific facts, procedures, "
                    "error codes, troubleshooting, safety, or "
                    "escalation information, use "
                    "search_service_documents rather than "
                    "answering from your own knowledge. "

                    "Use get_current_datetime only when the "
                    "task genuinely requires the current date, "
                    "time, or a timestamp. "

                    "Never invent equipment information. "

                    "After receiving a tool result, determine "
                    "whether another tool call is required. "
                    "If no additional tool is required, provide "
                    "the final response."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]

        tools = self.get_tool_definitions()

        tools_used = []

        tool_call_count = 0

        # ------------------------------------------------
        # Full agent loop
        # ------------------------------------------------

        while True:

            print(
                f"\n[Agent Loop] Model call "
                f"{tool_call_count + 1}..."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0
            )

            message = response.choices[0].message

            # ------------------------------------------------
            # No tool call -> Agent has finished
            # ------------------------------------------------

            if not message.tool_calls:

                answer = message.content

                print(
                    "\n[Agent Loop] "
                    "No further tool requested."
                )

                print("\nJunior Agent final response:")
                print(answer)

                return {
                    "question": question,
                    "tools_used": tools_used,
                    "tool_call_count": tool_call_count,
                    "answer": answer
                }

            # ------------------------------------------------
            # Tool call(s) requested
            # ------------------------------------------------

            messages.append(
                message.model_dump(
                    exclude_none=True
                )
            )

            for tool_call in message.tool_calls:

                tool_call_count += 1

                tool_name = tool_call.function.name

                try:
                    arguments = json.loads(
                        tool_call.function.arguments
                    )
                except json.JSONDecodeError:

                    arguments = {}

                print(
                    f"\n[Tool Call {tool_call_count}]"
                )

                print(
                    f"Tool requested: {tool_name}"
                )

                print(
                    f"Tool arguments: {arguments}"
                )

                # --------------------------------------------
                # Execute requested tool
                # --------------------------------------------

                try:

                    tool_result = self.execute_tool(
                        tool_name,
                        arguments
                    )

                except Exception as error:

                    tool_result = {
                        "error": str(error)
                    }

                tools_used.append(tool_name)

                print(
                    f"Tool executed: {tool_name}"
                )

                # --------------------------------------------
                # Return tool result to the model
                # --------------------------------------------

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        tool_result
                    )
                })

                print(
                    "Tool result returned to Junior Agent."
                )

if __name__ == "__main__":

    agent = AgenticRAGAgent()

    while True:

        question = input(
            "\nEngineer question (or quit): "
        ).strip()

        if question.lower() in {"quit", "exit"}:
            break

        agent.run(question)