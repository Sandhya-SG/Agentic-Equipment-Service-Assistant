"""Agentic RAG for real AEM equipment service documentation."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from asa.graph.state import Chunk
from asa.ingestion.retrieval import AEMHybridRetriever


class AgenticRAGAgent:
    """
    Equipment-aware Agentic RAG specialist.

    v1.0.1A establishes explicit document-search tool use.
    Later checkpoints add LLM retrieval assessment,
    reformulation, Senior review, and grounded generation.
    """

    def __init__(self) -> None:

        load_dotenv()

        self.retriever = AEMHybridRetriever()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.junior_model = "gpt-4o-mini"
        self.senior_model = "gpt-5.4-mini"

    # ------------------------------------------------------------------
    # Retrieval capability
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        equipment_model: str,
        k: int = 5,
    ) -> list[Chunk]:
        """
        Retrieve evidence strictly for the selected equipment.
        """

        return self.retriever.hybrid_search(
            query=query,
            equipment_model=equipment_model,
            k=k,
        )

    # ------------------------------------------------------------------
    # Explicit tool implementation
    # ------------------------------------------------------------------

    def search_service_documents(
        self,
        query: str,
        equipment_model: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Explicit retrieval tool exposed to the Junior Agent.

        The equipment model is mandatory so retrieval cannot
        silently search across different AEM systems.
        """

        results = self.retrieve(
            query=query,
            equipment_model=equipment_model,
            k=k,
        )

        return [
            {
                "chunk_id":
                    item.chunk_id,

                "doc_id":
                    item.doc_id,

                "source_file":
                    item.source_file,

                "page":
                    item.page,

                "section_id":
                    item.section_id,

                "section_title":
                    item.section_title,

                "revision":
                    item.revision,

                "equipment_model":
                    item.equipment_model,

                "text":
                    item.text,

                "rrf_score":
                    item.score,
            }

            for item in results
        ]

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Execute one supported Agentic RAG tool.
        """

        if tool_name == "search_service_documents":

            query = arguments.get(
                "query"
            )

            equipment_model = arguments.get(
                "equipment_model"
            )

            k = int(
                arguments.get(
                    "k",
                    5,
                )
            )

            if not query:
                raise ValueError(
                    "search_service_documents "
                    "requires query."
                )

            if not equipment_model:
                raise ValueError(
                    "search_service_documents "
                    "requires equipment_model."
                )

            return self.search_service_documents(
                query=query,
                equipment_model=equipment_model,
                k=k,
            )

        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    def get_tool_definitions(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return OpenAI-compatible tool definitions.

        The Junior Agent will use these definitions
        in v1.0.1B.
        """

        return [
            {
                "type": "function",
                "function": {
                    "name":
                        "search_service_documents",

                    "description": (
                        "Search the authoritative AEM "
                        "service documentation for the "
                        "selected equipment model."
                    ),

                    "parameters": {
                        "type": "object",

                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Concise service-document "
                                    "search query."
                                ),
                            },

                            "equipment_model": {
                                "type": "string",
                                "enum": [
                                    "thermal_station",
                                    "thermal_retrofit_1kw",
                                ],
                                "description": (
                                    "Equipment model whose "
                                    "documentation must be searched."
                                ),
                            },

                            "k": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "default": 5,
                            },
                        },

                        "required": [
                            "query",
                            "equipment_model",
                        ],

                        "additionalProperties":
                            False,
                    },
                },
            }
        ]

    def junior_search(
        self,
        question: str,
        equipment_model: str,
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Junior Agent explicitly calls the service-document
        search tool for the selected equipment.
        """

        if not equipment_model:
            raise ValueError(
                "Junior Agent requires equipment_model."
            )

        print(
            "\n[Junior Agent] "
            "Selecting retrieval tool..."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Junior Retrieval Agent for an "
                    "AEM equipment-service assistant. "
                    "For equipment service questions, you must "
                    "search the authoritative service documents "
                    "before answering. "
                    "Use the supplied equipment_model exactly. "
                    "Do not substitute or infer a different "
                    "equipment model."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Equipment model:\n"
                    f"{equipment_model}\n\n"
                    f"Engineer question:\n"
                    f"{question}\n\n"
                    "Search the service documentation for "
                    "evidence relevant to this question."
                ),
            },
        ]

        tools = self.get_tool_definitions()

        response = self.client.chat.completions.create(
            model=self.junior_model,
            messages=messages,
            tools=tools,

            # Retrieval is mandatory at this node.
            tool_choice={
                "type": "function",
                "function": {
                    "name":
                        "search_service_documents"
                },
            },

            temperature=0,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            raise RuntimeError(
                "Junior Agent did not request the "
                "required retrieval tool."
            )

        tool_call = message.tool_calls[0]

        tool_name = (
            tool_call.function.name
        )

        arguments = json.loads(
            tool_call.function.arguments
        )

        # --------------------------------------------------
        # Security / isolation boundary
        # --------------------------------------------------
        # The caller owns equipment identity.
        # Never allow the LLM to switch equipment.
        # --------------------------------------------------

        arguments["equipment_model"] = (
            equipment_model
        )

        arguments["k"] = k

        print(
            f"Tool requested: {tool_name}"
        )

        print(
            "Search query selected by Junior Agent: "
            f"{arguments.get('query')}"
        )

        print(
            "Equipment enforced by application: "
            f"{equipment_model}"
        )

        results = self.execute_tool(
            tool_name,
            arguments,
        )

        print(
            f"Tool executed: {tool_name}"
        )

        return {
            "tool_name":
                tool_name,

            "search_query":
                arguments.get("query"),

            "equipment_model":
                equipment_model,

            "results":
                results,
        }

    def format_retrieval_context(
        self,
        results: list[dict[str, Any]],
    ) -> str:
        """
        Format retrieved chunks for LLM evidence assessment.
        """

        context_parts = []

        for rank, result in enumerate(
            results,
            start=1,
        ):

            context_parts.append(
                f"[EVIDENCE {rank}]\n"
                f"Source: {result['source_file']}\n"
                f"Page: {result['page']}\n"
                f"Section: {result['section_title']}\n"
                f"Equipment: {result['equipment_model']}\n"
                f"Content:\n{result['text']}\n"
            )

        return "\n".join(
            context_parts
        )

    def assess_retrieval(
        self,
        question: str,
        equipment_model: str,
        results: list[dict[str, Any]],
    ) -> bool:
        """
        Ask the Junior model whether the retrieved CONTENT
        is sufficient to answer the engineer accurately.
        """

        if not results:

            print(
                "Retrieval assessment: POOR "
                "(no results)"
            )

            return False

        context = self.format_retrieval_context(
            results
        )

        prompt = f"""
You are a retrieval-quality evaluator for an
AEM equipment-service assistant.

Selected equipment:
{equipment_model}

Engineer question:
{question}

Retrieved evidence:
{context}

Determine whether the retrieved evidence contains
enough relevant documented information to answer the
engineer's question accurately and safely.

Important rules:

1. Judge the CONTENT, not the RRF ranking score.

2. Evidence must apply to the selected equipment.

3. Do not assume missing service instructions from
   general engineering knowledge.

4. For a procedure question, GOOD requires enough
   documented procedural evidence to answer the
   requested action.

5. For a safety question, GOOD requires documented
   safety evidence relevant to the requested condition.

6. If the evidence is only generally related but does
   not actually support the requested answer, return POOR.

Respond with exactly one word:

GOOD

or

POOR
"""

        response = self.client.chat.completions.create(
            model=self.junior_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate whether retrieved "
                        "AEM service documentation provides "
                        "sufficient evidence for an engineer's "
                        "question."
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

        print(
            f"LLM retrieval assessment: "
            f"{assessment}"
        )

        return assessment == "GOOD"

    def reformulate_query(
        self,
        question: str,
        equipment_model: str,
        results: list[dict[str, Any]],
    ) -> str:
        """
        Reformulate an insufficient search while preserving
        the original engineer intent and equipment identity.
        """

        context = self.format_retrieval_context(
            results
        )

        prompt = f"""
You are the Junior Retrieval Agent for an
AEM equipment-service assistant.

Selected equipment:
{equipment_model}

Original engineer question:
{question}

The first retrieval did not provide sufficient evidence.

Initial retrieved evidence:
{context}

Rewrite the engineer's question into a better search
query for the authoritative service manual.

Rules:

1. Preserve the original engineer intent.

2. Do not change or infer a different equipment model.

3. Use terminology likely to appear in an equipment
   operation manual, safety section, maintenance section,
   fault description, troubleshooting section, or
   servicing procedure.

4. Do not answer the question.

5. Return only the new search query.
"""

        response = self.client.chat.completions.create(
            model=self.junior_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You reformulate AEM equipment "
                        "service-document searches when "
                        "retrieval evidence is insufficient."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        return (
            response.choices[0]
            .message.content
            .strip()
        )

    def senior_review(
        self,
        question: str,
        equipment_model: str,
        first_query: str,
        first_results: list[dict[str, Any]],
        second_query: str,
        second_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Senior Agent reviews two failed retrieval attempts.

        It decides whether one materially different search
        is justified or whether human escalation is required.
        """

        first_context = (
            self.format_retrieval_context(
                first_results
            )
        )

        second_context = (
            self.format_retrieval_context(
                second_results
            )
        )

        prompt = f"""
            You are the Senior Retrieval Agent for an
            AEM equipment-service assistant.

            Selected equipment:
            {equipment_model}

            Original engineer question:
            {question}

            The Junior Agent made two retrieval attempts and
            both were assessed as insufficient.

            FIRST SEARCH QUERY:
            {first_query}

            FIRST RETRIEVED EVIDENCE:
            {first_context}

            SECOND SEARCH QUERY:
            {second_query}

            SECOND RETRIEVED EVIDENCE:
            {second_context}

            Decide whether ONE final retrieval attempt is justified.

            Choose RETRY only when:
            - a materially different search strategy or terminology
            is apparent from the evidence;
            - the new query has a realistic chance of locating
            documented information that the Junior missed.

            Choose ESCALATE when:
            - the requested procedure or information appears absent;
            - the retrieved material is only generally related;
            - another search would likely repeat the same evidence;
            - answering would require inventing undocumented steps.

            Important:
            - Do not answer the engineer's question.
            - Do not invent a procedure.
            - Do not change the selected equipment model.
            - RETRY must provide a meaningfully different query.

            Return exactly:

            DECISION: RETRY or ESCALATE
            REASON: <brief reason>
            QUERY: <new query if RETRY, otherwise NONE>
            """

        response = self.client.chat.completions.create(
            model=self.senior_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the Senior Retrieval Agent. "
                        "You review failed AEM document "
                        "retrieval and decide whether a final "
                        "search is justified or escalation "
                        "is required."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        review = (
            response.choices[0]
            .message.content
            .strip()
        )

        decision = "ESCALATE"
        reason = (
            "Senior Agent did not return "
            "a valid decision."
        )
        query = None

        for line in review.splitlines():

            line = line.strip()

            if line.startswith("DECISION:"):

                value = (
                    line.split(":", 1)[1]
                    .strip()
                    .upper()
                )

                if value in {
                    "RETRY",
                    "ESCALATE",
                }:
                    decision = value

            elif line.startswith("REASON:"):

                reason = (
                    line.split(":", 1)[1]
                    .strip()
                )

            elif line.startswith("QUERY:"):

                value = (
                    line.split(":", 1)[1]
                    .strip()
                )

                if (
                    value
                    and value.upper()
                    != "NONE"
                ):
                    query = value

        # RETRY without a usable query is unsafe /
        # operationally meaningless.
        if (
            decision == "RETRY"
            and not query
        ):
            decision = "ESCALATE"

            reason = (
                "Senior Agent requested RETRY "
                "without providing a valid query."
            )

        print(
            f"Senior decision: {decision}"
        )

        print(
            f"Senior reason: {reason}"
        )

        if query:

            print(
                f"Senior query: {query}"
            )

        return {
            "decision": decision,
            "reason": reason,
            "query": query,
        }

    def run_retrieval_loop(
        self,
        question: str,
        equipment_model: str,
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Junior Agent retrieval loop.

        Maximum retrieval attempts in v1.0.1C: 2.
        """

        print("\n" + "=" * 80)
        print("REAL AEM AGENTIC RETRIEVAL")
        print("=" * 80)

        print(
            f"\nEngineer question:\n"
            f"{question}"
        )

        print(
            f"\nEquipment model:\n"
            f"{equipment_model}"
        )

        # --------------------------------------------------
        # Attempt 1
        # --------------------------------------------------

        print(
            "\n[Step 1] Initial Junior retrieval..."
        )

        first = self.junior_search(
            question=question,
            equipment_model=equipment_model,
            k=k,
        )

        first_results = first["results"]

        # --------------------------------------------------
        # Assessment 1
        # --------------------------------------------------

        print(
            "\n[Step 2] Assess initial evidence..."
        )

        first_good = self.assess_retrieval(
            question=question,
            equipment_model=equipment_model,
            results=first_results,
        )

        if first_good:

            print(
                "Retrieval assessment: GOOD"
            )

            return {
                "original_query":
                    question,

                "final_query":
                    first["search_query"],

                "equipment_model":
                    equipment_model,

                "retrieval_attempts":
                    1,

                "retrieval_status":
                    "GOOD",

                "tool_name":
                    first["tool_name"],

                "senior_used":
                    False,

                "senior_decision":
                    None,

                "senior_reason":
                    None,

                "escalation_required":
                    False,

                "results":
                    first_results,
            }

        print(
            "Retrieval assessment: POOR"
        )

        # --------------------------------------------------
        # Reformulation
        # --------------------------------------------------

        print(
            "\n[Step 3] Reformulating search..."
        )

        new_query = self.reformulate_query(
            question=question,
            equipment_model=equipment_model,
            results=first_results,
        )

        print(
            f"Reformulated query:\n"
            f"{new_query}"
        )

        # --------------------------------------------------
        # Attempt 2
        # --------------------------------------------------

        print(
            "\n[Step 4] Second tool-based retrieval..."
        )

        second_results = self.execute_tool(
            "search_service_documents",
            {
                "query":
                    new_query,

                "equipment_model":
                    equipment_model,

                "k":
                    k,
            },
        )

        print(
            "Tool executed: "
            "search_service_documents"
        )

        # --------------------------------------------------
        # Assessment 2
        # --------------------------------------------------

        print(
            "\n[Step 5] Reassess evidence..."
        )

        second_good = self.assess_retrieval(
            question=question,
            equipment_model=equipment_model,
            results=second_results,
        )

        if second_good:

            print(
                "Second retrieval assessment: GOOD"
            )

            return {
                "original_query":
                    question,

                "final_query":
                    new_query,

                "equipment_model":
                    equipment_model,

                "retrieval_attempts":
                    2,

                "retrieval_status":
                    "GOOD",

                "tool_name":
                    "search_service_documents",

                "senior_used":
                    False,

                "senior_decision":
                    None,

                "senior_reason":
                    None,

                "escalation_required":
                    False,

                "results":
                    second_results,
            }

        print(
            "Second retrieval assessment: POOR"
        )

        # --------------------------------------------------
        # Senior review
        # --------------------------------------------------

        print(
            "\n[Step 6] Escalating retrieval review "
            "to Senior Agent..."
        )

        senior = self.senior_review(
            question=question,
            equipment_model=equipment_model,

            first_query=
                first["search_query"],

            first_results=
                first_results,

            second_query=
                new_query,

            second_results=
                second_results,
        )

        if (
            senior["decision"]
            == "ESCALATE"
        ):

            print(
                "Senior Agent determined that "
                "another retrieval attempt is "
                "not justified."
            )

            return {
                "original_query":
                    question,

                "final_query":
                    new_query,

                "equipment_model":
                    equipment_model,

                "retrieval_attempts":
                    2,

                "retrieval_status":
                    "INSUFFICIENT",

                "tool_name":
                    "search_service_documents",

                "results":
                    second_results,

                "senior_used":
                    True,

                "senior_decision":
                    "ESCALATE",

                "senior_reason":
                    senior["reason"],

                "escalation_required":
                    True,
            }

        senior_query = senior["query"]

        print(
            "\n[Step 7] Senior-directed "
            "final retrieval..."
        )

        third_results = self.execute_tool(
            "search_service_documents",
            {
                "query":
                    senior_query,

                "equipment_model":
                    equipment_model,

                "k":
                    k,
            },
        )

        print(
            "Tool executed: "
            "search_service_documents"
        )

        print(
            "\n[Step 8] Assess Senior-directed "
            "retrieval..."
        )

        third_good = self.assess_retrieval(
            question=question,
            equipment_model=equipment_model,
            results=third_results,
        )

        if third_good:

            print(
                "Senior-directed retrieval "
                "assessment: GOOD"
            )

            return {
                "original_query":
                    question,

                "final_query":
                    senior_query,

                "equipment_model":
                    equipment_model,

                "retrieval_attempts":
                    3,

                "retrieval_status":
                    "GOOD",

                "tool_name":
                    "search_service_documents",

                "results":
                    third_results,

                "senior_used":
                    True,

                "senior_decision":
                    "RETRY",

                "senior_reason":
                    senior["reason"],

                "escalation_required":
                    False,
            }

        print(
            "Senior-directed retrieval "
            "assessment: POOR"
        )

        return {
            "original_query":
                question,

            "final_query":
                senior_query,

            "equipment_model":
                equipment_model,

            "retrieval_attempts":
                3,

            "retrieval_status":
                "INSUFFICIENT",

            "tool_name":
                "search_service_documents",

            "results":
                third_results,

            "senior_used":
                True,

            "senior_decision":
                "RETRY",

            "senior_reason":
                senior["reason"],

            "escalation_required":
                True,
        }

    def generate_grounded_answer(
        self,
        question: str,
        equipment_model: str,
        results: list[dict[str, Any]],
    ) -> str:
        """
        Generate an engineer-facing answer using only
        retrieved AEM manual evidence.
        """

        context = self.format_retrieval_context(
            results
        )

        prompt = f"""
            You are an AEM equipment-service assistant.

            Selected equipment:
            {equipment_model}

            Engineer question:
            {question}

            Authoritative retrieved manual evidence:
            {context}

            Answer the engineer using ONLY the retrieved evidence.

            Rules:

            1. Do not use undocumented equipment knowledge.

            2. Do not invent procedures, parameter values,
            tools, replacement steps, safety steps, causes,
            warnings, or prerequisites.

            3. Preserve the terminology used in the manual.

            4. The selected equipment identity is authoritative.
            Do not rename the selected equipment or substitute
            another equipment model.

            5. If the retrieved evidence mentions another equipment,
            subsystem, station, module, or related machine,
            preserve that term only when necessary to accurately
            describe the documented relationship.

            6. Never imply that evidence for another equipment model
            applies to the selected equipment unless the retrieved
            manual explicitly establishes that relationship.

            7. If the evidence contains qualifications, warnings,
            prerequisites, or safety conditions relevant to the
            answer, include them.

            8. Do not claim a root cause unless the evidence
            explicitly establishes one.

            9. Be concise and practical for a service engineer.

            10. Every procedural, safety, maintenance, numerical,
                or equipment-specific factual claim must have an
                immediately following citation.

            11. Place each citation immediately after the sentence
                or bullet point it supports.

            12. Do not use one citation at the end of a long list
                to imply that it supports every item unless that
                source actually supports every item.

            13. If a claim cannot be directly supported by the
                retrieved evidence, omit it.

            14. Cite only evidence actually used in the answer.
                Do not cite a retrieved chunk merely because it was
                returned by the search.

            15. Use this exact citation format:

                [Source: <source_file>, page <page>,
                section <section_title>]

            Return only the grounded engineer-facing answer.
            """
        
        response = self.client.chat.completions.create(
            model=self.junior_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate grounded AEM "
                        "equipment-service answers only "
                        "from provided manual evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        return (
            response.choices[0]
            .message.content
            .strip()
        )

    def run(
        self,
        question: str,
        equipment_model: str,
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Execute the complete real AEM Agentic RAG workflow.
        """

        retrieval = self.run_retrieval_loop(
            question=question,
            equipment_model=equipment_model,
            k=k,
        )

        # --------------------------------------------------
        # Evidence insufficient
        # --------------------------------------------------

        if (
            retrieval["retrieval_status"]
            != "GOOD"
        ):

            print(
                "\nNo grounded answer generated."
            )

            print(
                "Documented evidence is insufficient. "
                "Human escalation is required."
            )

            return {
                **retrieval,

                "answer":
                    (
                        "No grounded answer generated. "
                        "The available AEM documentation "
                        "does not provide sufficient evidence "
                        "to answer this request. "
                        "Human escalation is required."
                    ),
            }

        # --------------------------------------------------
        # Generate grounded answer
        # --------------------------------------------------

        print(
            "\n[Final Step] Generating "
            "grounded answer..."
        )

        answer = self.generate_grounded_answer(
            question=question,
            equipment_model=equipment_model,
            results=retrieval["results"],
        )

        print("\nGrounded answer:")
        print(answer)

        return {
            **retrieval,
            "answer": answer,
        }