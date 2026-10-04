"""Diagnostic Agent for evidence-grounded equipment troubleshooting."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from asa.agents.rag import AgenticRAGAgent
from asa.graph.state import (
    AgentState,
    Chunk,
    RankedCause,
    Step,
)


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

DIAGNOSTIC_MODEL = "gpt-4o-mini"

rag_agent = AgenticRAGAgent()


def _format_diagnostic_context(
    chunks: list[Chunk],
) -> str:
    """
    Format retrieved evidence for the Diagnostic Agent.
    """

    parts = []

    for rank, chunk in enumerate(
        chunks,
        start=1,
    ):

        parts.append(
            f"[EVIDENCE {rank}]\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Source: {chunk.source_file}\n"
            f"Page: {chunk.page}\n"
            f"Section: {chunk.section_title}\n"
            f"Equipment: {chunk.equipment_model}\n"
            f"Content:\n{chunk.text}\n"
        )

    return "\n".join(parts)

def _validate_diagnostic_claims(
    question: str,
    chunks: list[Chunk],
    candidate_causes: list[RankedCause],
    troubleshooting_steps: list[Step],
) -> tuple[list[RankedCause], list[Step]]:
    """
    Validate generated diagnostic claims against the
    exact chunks cited as supporting evidence.

    Unsupported claims are removed.
    """

    chunk_map = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    valid_causes = []

    for cause in candidate_causes:

        evidence = [
            chunk_map[chunk_id]
            for chunk_id
            in cause.supporting_chunk_ids
            if chunk_id in chunk_map
        ]

        if not evidence:
            continue

        evidence_text = "\n\n".join(
            chunk.text
            for chunk in evidence
        )

        prompt = f"""
You are validating one candidate diagnostic cause
against authoritative AEM manual evidence.

Engineer symptom:
{question}

Candidate cause:
{cause.cause}

Cited evidence:
{evidence_text}

Determine whether the cited evidence EXPLICITLY supports
the candidate cause as a possible explanation for the
engineer's reported symptom.

Important:

- Mere keyword overlap is not support.
- A retrieved chunk being related to the equipment is
  not enough.
- A symptom that merely restates the engineer's problem
  is not a cause.
- Do not infer a causal relationship that the evidence
  does not state or clearly establish.

Return exactly:

SUPPORTED

or

UNSUPPORTED
"""

        response = client.chat.completions.create(
            model=DIAGNOSTIC_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You validate diagnostic claims "
                        "strictly against cited manual evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        verdict = (
            response.choices[0]
            .message.content
            .strip()
            .upper()
        )

        if verdict == "SUPPORTED":
            valid_causes.append(
                cause
            )

    valid_steps = []

    for step in troubleshooting_steps:

        evidence = [
            chunk_map[chunk_id]
            for chunk_id
            in step.supporting_chunk_ids
            if chunk_id in chunk_map
        ]

        if not evidence:
            continue

        evidence_text = "\n\n".join(
            chunk.text
            for chunk in evidence
        )

        prompt = f"""
You are validating one recommended diagnostic check
against authoritative AEM manual evidence.

Engineer symptom:
{question}

Recommended check:
{step.action}

Cited evidence:
{evidence_text}

Determine whether the cited evidence EXPLICITLY supports
this action as a documented check, troubleshooting action,
or relevant documented response.

Important:

- Do not approve an action merely because its terminology
  appears in the evidence.
- Do not infer an adjustment or reset procedure.
- The evidence must actually support performing the
  stated check or action.

Return exactly:

SUPPORTED

or

UNSUPPORTED
"""

        response = client.chat.completions.create(
            model=DIAGNOSTIC_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You validate troubleshooting actions "
                        "strictly against cited manual evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        verdict = (
            response.choices[0]
            .message.content
            .strip()
            .upper()
        )

        if verdict == "SUPPORTED":
            valid_steps.append(
                step
            )

    return (
        valid_causes,
        valid_steps,
    )


def diagnostic_node(
    state: AgentState,
) -> dict[str, Any]:
    """
    Diagnose an observed equipment symptom using only
    documented evidence retrieved for the selected equipment.
    """

    print("\n[Diagnostic Agent]")

    question = (
        state.get("resolved_query")
        or state.get("raw_query")
        or ""
    ).strip()

    equipment_model = (
        state.get("equipment_model")
        or ""
    ).strip()

    if not question:
        raise ValueError(
            "Diagnostic Agent requires a query."
        )

    if not equipment_model:
        raise ValueError(
            "Diagnostic Agent requires equipment_model."
        )

    print(
        "Diagnostic Agent requesting "
        "documented evidence..."
    )

    # --------------------------------------------------
    # Use the proven Agentic RAG evidence gate
    # --------------------------------------------------

    rag_result = rag_agent.run_retrieval_loop(
        question=question,
        equipment_model=equipment_model,
        k=5,
    )

    retrieved_chunks = [
        Chunk(
            chunk_id=item["chunk_id"],
            doc_id=item["doc_id"],
            section_id=item["section_id"],
            revision=item["revision"],
            equipment_model=item["equipment_model"],
            text=item["text"],
            score=item["rrf_score"],
            source_file=item["source_file"],
            page=item["page"],
            section_title=item["section_title"],
        )
        for item in rag_result["results"]
    ]

    # --------------------------------------------------
    # RAG could not establish sufficient evidence
    # --------------------------------------------------

    if (
        rag_result["retrieval_status"]
        != "GOOD"
    ):

        reason = (
            rag_result.get("senior_reason")
            or (
                "The available AEM documentation "
                "does not provide sufficient evidence "
                "for a grounded diagnostic assessment."
            )
        )

        print(
            "Diagnostic evidence insufficient."
        )

        return {
            "retrieval_strategy":
                "hybrid",

            "retrieved_chunks":
                retrieved_chunks,

            "sufficiency":
                False,

            "retry_count":
                max(
                    rag_result[
                        "retrieval_attempts"
                    ] - 1,
                    0,
                ),

            "root_causes":
                [],

            "troubleshooting_steps":
                [],

            "escalated":
                True,

            "escalation_reason":
                reason,

            "final_answer":
                (
                    "A document-grounded diagnosis "
                    "cannot be made from the available "
                    "AEM manual evidence. "
                    "Human escalation is required."
                ),
        }

    # --------------------------------------------------
    # Evidence is sufficient
    # --------------------------------------------------

    context = _format_diagnostic_context(
        retrieved_chunks
    )

    prompt = f"""
You are the Diagnostic Agent for an AEM
equipment-service assistant.

Selected equipment:
{equipment_model}

Engineer problem:
{question}

Authoritative retrieved manual evidence:
{context}

Using ONLY the supplied evidence, produce a conservative
diagnostic assessment.

Important rules:

1. Do not introduce equipment facts that are not contained
   in the supplied evidence.

2. Do not invent a root cause.

3. A candidate cause may be returned ONLY when the
   supplied evidence explicitly links that condition,
   fault, component failure, or state to the engineer's
   reported symptom.

4. Do not infer causation merely because a retrieved
   chunk contains related terminology.

5. Do not convert a symptom, alarm description,
   operating condition, parameter, or unrelated
   troubleshooting item into a candidate cause unless
   the evidence explicitly establishes the causal
   relationship.

6. If the evidence identifies a possible component
   failure and explicitly states its effect, you may
   include it as a candidate cause.

7. If no explicit causal relationship is documented,
   return an empty candidate_causes list.

8. Recommended checks must be explicitly documented
   as checks or troubleshooting actions relevant to
   the engineer's symptom or to a supported candidate
   cause.

9. Do not turn an operating value or parameter mentioned
   in a retrieved chunk into a recommended adjustment
   unless the supplied evidence explicitly instructs
   that adjustment for the relevant condition.

10. Do not invent diagnostic tests, measurements,
    parameter values, tools, adjustment procedures,
    replacement procedures, or reset procedures.

11. Each candidate cause must list the chunk IDs that
    support it.

12. Each recommended check must list the chunk IDs that
    support it.

13. A symptom or failure description that merely restates
    the engineer's reported problem is NOT a candidate cause.

14. For example, if the engineer reports that temperature
    cannot reach setpoint, evidence stating that temperature
    ramp-up cannot reach setpoint describes the symptom and
    must not be returned as a cause unless the evidence also
    identifies why it occurs.

15. The assessment field must be consistent with the
    structured candidate_causes and recommended_checks.

16. Do not mention a possible cause in the assessment unless
    that same cause is included in candidate_causes and has
    explicit supporting evidence.

17. Do not mention a recommended action in the assessment
    unless that action is included in recommended_checks and
    has explicit supporting evidence.

Return VALID JSON only in this exact structure:

{{
  "assessment": "<brief diagnostic assessment>",
  "root_cause_confirmed": false,
  "candidate_causes": [
    {{
      "cause": "<candidate cause>",
      "likelihood": 0.0,
      "supporting_chunk_ids": ["<chunk_id>"]
    }}
  ],
  "recommended_checks": [
    {{
      "action": "<documented check>",
      "supporting_chunk_ids": ["<chunk_id>"],
      "hazard_flag": false
    }}
  ]
}}

Likelihood must be between 0.0 and 1.0.

Likelihood is NOT a failure probability and must not be
derived from general engineering knowledge.

Use likelihood only as a relative ordering score among
candidate causes that are explicitly supported by the
supplied evidence.

If the evidence does not provide enough basis to
differentiate supported candidate causes, assign them
the same likelihood score.

Do not use unsupported numerical precision to imply
statistical confidence.

If the evidence does not support any candidate cause,
return an empty candidate_causes list.

If the evidence does not support a documented check,
do not include that check.
"""

    response = client.chat.completions.create(
        model=DIAGNOSTIC_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a conservative equipment "
                    "Diagnostic Agent. Diagnose only from "
                    "provided AEM manual evidence and "
                    "return valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        response_format={
            "type": "json_object"
        },
    )

    raw = (
        response.choices[0]
        .message.content
        .strip()
    )

    diagnostic = json.loads(
        raw
    )

    # --------------------------------------------------
    # Convert model output into shared state types
    # --------------------------------------------------

    root_causes = []

    for item in diagnostic.get(
        "candidate_causes",
        [],
    ):

        likelihood = float(
            item.get(
                "likelihood",
                0.0,
            )
        )

        likelihood = max(
            0.0,
            min(
                1.0,
                likelihood,
            ),
        )

        root_causes.append(
            RankedCause(
                cause=str(
                    item.get(
                        "cause",
                        ""
                    )
                ).strip(),

                likelihood=
                    likelihood,

                supporting_chunk_ids=list(
                    item.get(
                        "supporting_chunk_ids",
                        [],
                    )
                ),
            )
        )

    troubleshooting_steps = []

    for order, item in enumerate(
        diagnostic.get(
            "recommended_checks",
            [],
        ),
        start=1,
    ):

        troubleshooting_steps.append(
            Step(
                order=order,

                action=str(
                    item.get(
                        "action",
                        ""
                    )
                ).strip(),

                supporting_chunk_ids=list(
                    item.get(
                        "supporting_chunk_ids",
                        [],
                    )
                ),

                hazard_flag=bool(
                    item.get(
                        "hazard_flag",
                        False,
                    )
                ),
            )
        )

    # --------------------------------------------------
    # Validate diagnostic claims against cited evidence
    # --------------------------------------------------

    print(
        "\n[Diagnostic Grounding Validation]"
    )

    root_causes, troubleshooting_steps = (
        _validate_diagnostic_claims(
            question=question,
            chunks=retrieved_chunks,
            candidate_causes=root_causes,
            troubleshooting_steps=
                troubleshooting_steps,
        )
    )

    # --------------------------------------------------
    # Build conservative validated assessment
    # --------------------------------------------------

    if root_causes:

        assessment = (
            "The root cause is not confirmed. "
            "The AEM manual identifies documented conditions "
            "that may be relevant to the reported symptom. "
            "Use the documented checks below for further assessment."
        )

    else:

        assessment = (
            "The root cause is not confirmed from the "
            "available AEM manual evidence. The retrieved "
            "documentation does not explicitly support a "
            "specific candidate cause."
        )

    # --------------------------------------------------
    # Build engineer-facing diagnostic response
    # --------------------------------------------------

    answer_parts = [
        "Diagnostic assessment:",
        assessment,
    ]

    if root_causes:

        answer_parts.append(
            "\nDocumented candidate conditions:"
        )

        for rank, cause in enumerate(
            root_causes,
            start=1,
        ):

            answer_parts.append(
                f"{rank}. {cause.cause}"
            )

    if troubleshooting_steps:

        answer_parts.append(
            "\nRecommended documented checks:"
        )

        for step in troubleshooting_steps:

            answer_parts.append(
                f"{step.order}. "
                f"{step.action}"
            )

    answer = "\n".join(
        answer_parts
    )

    print("\nDiagnostic result:")
    print(answer)

    return {
        "retrieval_strategy":
            "hybrid",

        "retrieved_chunks":
            retrieved_chunks,

        "sufficiency":
            True,

        "retry_count":
            max(
                rag_result[
                    "retrieval_attempts"
                ] - 1,
                0,
            ),

        "root_causes":
            root_causes,

        "troubleshooting_steps":
            troubleshooting_steps,

        "escalated":
            False,

        "escalation_reason":
            "",

        "final_answer":
            answer,
    }