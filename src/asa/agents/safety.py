"""Safety Agent for evidence-grounded equipment service decisions."""

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
)


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

SAFETY_MODEL = "gpt-4o-mini"

rag_agent = AgenticRAGAgent()


def _format_safety_context(
    chunks: list[Chunk],
) -> str:
    """
    Format retrieved AEM evidence for safety assessment.
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



def _build_safety_search_query(
    question: str,
    equipment_model: str,
) -> str:
    """
    Convert the engineer request into a safety-requirement
    retrieval query.

    The search should look for applicable safety controls,
    requirements, prohibitions, and prerequisites rather
    than instructions for performing a potentially unsafe
    requested action.
    """

    prompt = f"""
You are preparing a document-search query for an AEM
equipment Safety Agent.

Selected equipment:
{equipment_model}

Engineer request:
{question}

Create ONE concise search query for the authoritative
AEM manual.

The search query must look for the safety requirements,
controls, prerequisites, hazards, prohibitions, or
required equipment state that govern the engineer's
request.

Do NOT search for instructions to perform an unsafe
action.

Examples:

Engineer:
"Can I bypass the safety interlock?"

Search:
safety interlock requirements door interlock enabled
panel doors closed

Engineer:
"Can I open the panel while the equipment is powered?"

Search:
panel access electrical safety power off servicing
hazardous voltage

Engineer:
"What safety precautions apply before servicing?"

Search:
servicing safety precautions power off PPE qualified
personnel maintenance

Return only the search query.
"""

    response = client.chat.completions.create(
        model=SAFETY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You create conservative safety-document "
                    "search queries. Search for governing "
                    "safety requirements, not unsafe "
                    "procedures."
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
        .strip('"')
    )


def _validate_safety_items(
    question: str,
    chunks: list[Chunk],
    hazards: list[str],
    ppe_required: list[str],
    required_controls: list[str],
) -> tuple[
    list[str],
    list[str],
    list[str],
]:
    """
    Validate Safety Agent structured claims against the
    retrieved AEM manual evidence.

    A claim must be explicitly supported by the evidence
    and applicable to the engineer's request.
    """

    context = _format_safety_context(
        chunks
    )

    def validate_item(
        item: str,
        item_type: str,
    ) -> bool:

        prompt = f"""
You are validating one structured safety claim against
authoritative AEM manual evidence.

Engineer request:
{question}

Claim type:
{item_type}

Candidate claim:
{item}

Retrieved AEM manual evidence:
{context}

Determine whether the candidate claim is BOTH:

1. EXPLICITLY SUPPORTED by the supplied evidence; and

2. APPLICABLE to the engineer's requested activity,
   condition, or safety question.

Important rules:

- Mere keyword overlap is not sufficient.
- Do not use general engineering knowledge.
- Do not infer requirements that the manual does not state.
- Do not approve an item merely because it appears somewhere
  in the retrieved documentation.
- The item must be relevant to the engineer's actual request.

For HAZARD:
The evidence must explicitly identify the hazard and it
must be relevant to the requested activity or condition.

For PPE:
The evidence must explicitly identify the PPE and establish
that it applies to the relevant activity or condition.
PPE listed only for a different activity such as installation
or decommissioning is not automatically applicable to
servicing, troubleshooting, or another task.

For REQUIRED_CONTROL:
The evidence must explicitly support the control or
prerequisite as applicable to the request.
Do not convert recommendations into mandatory requirements.

Return exactly:

SUPPORTED

or

UNSUPPORTED
"""

        response = client.chat.completions.create(
            model=SAFETY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You validate safety claims strictly "
                        "against supplied AEM manual evidence."
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

        return verdict == "SUPPORTED"

    valid_hazards = [
        item
        for item in hazards
        if validate_item(
            item,
            "HAZARD",
        )
    ]

    valid_ppe = [
        item
        for item in ppe_required
        if validate_item(
            item,
            "PPE",
        )
    ]

    valid_controls = [
        item
        for item in required_controls
        if validate_item(
            item,
            "REQUIRED_CONTROL",
        )
    ]

    return (
        valid_hazards,
        valid_ppe,
        valid_controls,
    )


def _validate_safety_verdict(
    question: str,
    chunks: list[Chunk],
    candidate_verdict: str,
    validated_controls: list[str],
) -> str:
    """
    Validate the proposed Safety Agent verdict against
    retrieved AEM manual evidence.

    A verdict that cannot be grounded conservatively is
    converted to ESCALATE.
    """

    context = _format_safety_context(
        chunks
    )

    controls_text = (
        "\n".join(
            f"- {item}"
            for item in validated_controls
        )
        or "NONE"
    )

    prompt = f"""
You are validating a proposed equipment safety verdict
against authoritative AEM manual evidence.

Engineer request:
{question}

Proposed verdict:
{candidate_verdict.upper()}

Validated documented controls:
{controls_text}

Retrieved AEM manual evidence:
{context}

Determine whether the proposed verdict is explicitly
supported by the supplied evidence.

Verdict meanings:

ALLOW:
The request is informational about documented safety
requirements, OR the requested activity is supported
when the validated documented controls are followed.

HALT:
The engineer explicitly proposes an action that conflicts
with a documented safety requirement, OR describes a
condition for which the documentation requires operation
or work to stop.

ESCALATE:
The documentation is insufficient to make a grounded
safety determination.

Important rules:

- Use ONLY the supplied AEM evidence.
- Do not use general engineering knowledge.
- Do not infer permission from silence.
- Absence of a documented prohibition does NOT establish
  ALLOW.
- A safety control must not be weakened.
- If the evidence is insufficient or ambiguous, the safe
  result is ESCALATE.
- Do not change HALT to ALLOW merely because a bypass
  procedure is absent.
- Consider the engineer's actual requested action together
  with the documented requirements.

Return exactly:

SUPPORTED

or

UNSUPPORTED
"""

    response = client.chat.completions.create(
        model=SAFETY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You validate equipment safety verdicts "
                    "strictly against supplied AEM manual "
                    "evidence."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    validation = (
        response.choices[0]
        .message.content
        .strip()
        .upper()
    )

    if validation == "SUPPORTED":
        return candidate_verdict

    return "escalate"


def safety_node(
    state: AgentState,
) -> dict[str, Any]:
    """
    Evaluate a safety-related engineer request using only
    authoritative documentation for the selected equipment.

    Verdicts:

    allow:
        Documentation supports proceeding only under the
        documented safety controls.

    halt:
        The requested action or present condition conflicts
        with documented safety requirements.

    escalate:
        Available documentation is insufficient to make a
        safe determination.
    """

    print("\n[Safety Agent]")

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
            "Safety Agent requires a query."
        )

    if not equipment_model:
        raise ValueError(
            "Safety Agent requires equipment_model."
        )

    print(
        "Safety Agent requesting "
        "documented safety evidence..."
    )

    # --------------------------------------------------
    # Safety-oriented retrieval
    # --------------------------------------------------

    safety_query = _build_safety_search_query(
        question=question,
        equipment_model=equipment_model,
    )

    print(
        "Safety search query: "
        f"{safety_query}"
    )

    results = rag_agent.search_service_documents(
        query=safety_query,
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
        for item in results
    ]

    if not retrieved_chunks:

        reason = (
            "No relevant AEM safety documentation "
            "was retrieved for this request."
        )

        return {
            "retrieval_strategy":
                "hybrid",

            "retrieved_chunks":
                [],

            "sufficiency":
                False,

            "retry_count":
                0,

            "safety_verdict":
                "escalate",

            "hazards":
                [],

            "ppe_required":
                [],

            "safety_reason":
                reason,

            "escalated":
                True,

            "escalation_reason":
                reason,

            "final_answer":
                (
                    "Safety decision: ESCALATE\n\n"
                    f"{reason} Human review is required."
                ),
        }

    # --------------------------------------------------
    # Evidence sufficient for Safety Agent assessment
    # --------------------------------------------------

    context = _format_safety_context(
        retrieved_chunks
    )

    prompt = f"""
You are the Safety Agent for an AEM
equipment-service assistant.

Selected equipment:
{equipment_model}

Engineer request:
{question}

Authoritative retrieved AEM manual evidence:
{context}

Using ONLY the supplied evidence, make a conservative
safety decision.

Choose exactly one verdict:

ALLOW

Use ALLOW when:

- the engineer is asking for documented safety
  precautions, requirements, prerequisites, PPE,
  hazards, or controls; OR

- the documentation supports the requested activity
  when its documented safety controls are followed.

A request asking what safety precautions apply does NOT
mean that the engineer is currently violating those
precautions.

Do not choose HALT merely because the engineer has not
explicitly stated that every required control is already
satisfied.

ALLOW does NOT mean unrestricted permission to perform
the work. The response must preserve all documented
prerequisites, shutdown requirements, PPE, isolation
requirements, personnel requirements, and other controls.

HALT

Use HALT only when:

- the engineer explicitly proposes or requests an action
  that conflicts with documented safety requirements; OR

- the engineer describes a present equipment condition
  for which the documentation requires operation or work
  to stop.

Examples include:
- requesting to bypass or defeat a documented safety
  control;
- requesting to perform work energized when the
  documentation explicitly requires power-off;
- continuing operation when the documentation explicitly
  requires stopping.

Do NOT use HALT merely because a request asks what safety
precautions or controls are required.

ESCALATE

Use ESCALATE when the retrieved evidence is relevant
but does not establish whether the requested action can
be performed safely, or when making a safe determination
would require assumptions beyond the documentation.

Important rules:

1. Use ONLY the supplied AEM manual evidence.

2. Do not introduce safety requirements from general
   engineering knowledge.

3. Do not invent PPE, lockout steps, isolation steps,
   voltage limits, pressure limits, tools, procedures,
   or qualifications.

4. Preserve the terminology and qualifications used in
   the manual.

5. Do not weaken a documented prohibition or warning.

6. Do not convert a recommendation into a mandatory
   requirement unless the manual states it as mandatory.

7. Do not convert a mandatory requirement into an
   optional recommendation.

8. Never recommend bypassing, defeating, disabling,
   overriding, or circumventing a documented safety
   control.

9. If the documentation is ambiguous about whether an
   action is safe, choose ESCALATE rather than ALLOW.

10. List only hazards explicitly supported by the
    supplied evidence.

11. List PPE only when the supplied evidence explicitly
    identifies the PPE or personal safety equipment.

12. The selected equipment identity is authoritative.
    Do not substitute another equipment model.

13. Distinguish an informational safety question from
    an unsafe proposed action.

    Example:

    "What safety precautions apply before servicing?"
    -> ALLOW with the documented controls.

    "Can I service the equipment without powering it off?"
    -> HALT if the supplied evidence requires power-off.

    Do not assume that asking about precautions means
    those precautions will be ignored.

14. In the reason field, distinguish explicit manual
    requirements from interpretations of the engineer's
    requested action.

    Do not state that the manual "explicitly" covers a
    specific action unless the supplied evidence actually
    names that action.

    Prefer wording that states the documented requirement
    directly.

Return VALID JSON only in this exact structure:

{{
  "verdict": "allow" | "halt" | "escalate",
  "reason": "<brief evidence-grounded reason>",
  "hazards": [
    "<documented hazard>"
  ],
  "ppe_required": [
    "<documented PPE>"
  ],
  "required_controls": [
    "<documented safety control or prerequisite>"
  ]
}}
"""

    response = client.chat.completions.create(
        model=SAFETY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a conservative equipment "
                    "Safety Agent. Make safety decisions "
                    "only from supplied AEM manual evidence "
                    "and return valid JSON."
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

    safety = json.loads(
        raw
    )

    # --------------------------------------------------
    # Normalize verdict conservatively
    # --------------------------------------------------

    verdict = str(
        safety.get(
            "verdict",
            "escalate",
        )
    ).strip().lower()

    if verdict not in {
        "allow",
        "halt",
        "escalate",
    }:
        verdict = "escalate"

    reason = str(
        safety.get(
            "reason",
            (
                "A safe determination could not "
                "be established from the supplied "
                "documentation."
            ),
        )
    ).strip()

    hazards = [
        str(item).strip()
        for item in safety.get(
            "hazards",
            [],
        )
        if str(item).strip()
    ]

    ppe_required = [
        str(item).strip()
        for item in safety.get(
            "ppe_required",
            [],
        )
        if str(item).strip()
    ]

    required_controls = [
        str(item).strip()
        for item in safety.get(
            "required_controls",
            [],
        )
        if str(item).strip()
    ]

    # --------------------------------------------------
    # Safety grounding validation
    # --------------------------------------------------

    print(
        "\n[Safety Grounding Validation]"
    )

    (
        hazards,
        ppe_required,
        required_controls,
    ) = _validate_safety_items(
        question=question,
        chunks=retrieved_chunks,
        hazards=hazards,
        ppe_required=ppe_required,
        required_controls=required_controls,
    )

    verdict = _validate_safety_verdict(
        question=question,
        chunks=retrieved_chunks,
        candidate_verdict=verdict,
        validated_controls=required_controls,
    )

    # --------------------------------------------------
    # Construct conservative validated reason
    #
    # Do not expose the Safety LLM's original free-text
    # reason after grounding validation.
    # --------------------------------------------------

    if verdict == "allow":

        reason = (
            "The retrieved AEM manual evidence provides "
            "documented safety requirements applicable "
            "to this request. Follow the validated "
            "controls below."
        )

    elif verdict == "halt":

        reason = (
            "The requested action conflicts with a "
            "documented safety requirement in the "
            "retrieved AEM manual evidence."
        )

    else:

        reason = (
            "The available AEM manual evidence is "
            "insufficient to make a grounded safety "
            "determination. Human review is required."
        )

    # --------------------------------------------------
    # Build engineer-facing response
    # --------------------------------------------------

    answer_parts = [
        f"Safety decision: {verdict.upper()}",
        "",
        reason,
    ]

    if hazards:

        answer_parts.append(
            "\nDocumented hazards:"
        )

        for item in hazards:

            answer_parts.append(
                f"- {item}"
            )

    if ppe_required:

        answer_parts.append(
            "\nDocumented PPE:"
        )

        for item in ppe_required:

            answer_parts.append(
                f"- {item}"
            )

    if required_controls:

        answer_parts.append(
            "\nRequired documented controls:"
        )

        for item in required_controls:

            answer_parts.append(
                f"- {item}"
            )

    answer = "\n".join(
        answer_parts
    )

    print(
        f"Safety decision: "
        f"{verdict.upper()}"
    )

    print(
        f"Safety reason: {reason}"
    )

    return {
        "retrieval_strategy":
            "hybrid",

        "retrieved_chunks":
            retrieved_chunks,

        "sufficiency":
            verdict != "escalate",

        "retry_count":
            0,

        "safety_verdict":
            verdict,

        "hazards":
            hazards,

        "ppe_required":
            ppe_required,

        "safety_reason":
            reason,

        # Only ESCALATE requires human escalation.
        # HALT means the Safety Agent has enough
        # documented evidence to prohibit the action.
        "escalated":
            verdict == "escalate",

        "escalation_reason":
            (
                reason
                if verdict == "escalate"
                else ""
            ),

        "final_answer":
            answer,
    }