"""Request Agent for engineer-request completeness assessment."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

REQUEST_MODEL = "gpt-4o-mini"


def request_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Determine whether the engineer request contains enough
    information to proceed to specialist planning.

    This node does not answer, diagnose, retrieve documents,
    or select a specialist.
    """

    print("\n[Request Agent]")

    question = (
        state.get("resolved_query")
        or state.get("raw_query")
        or ""
    ).strip()

    is_clarification_continuation = bool(
        state.get("resolved_query")
    )

    equipment_model = (
        state.get("equipment_model")
        or ""
    ).strip()

    print(
        f"Engineer request: {question}"
    )

    print(
        f"Equipment model: "
        f"{equipment_model or 'NOT SELECTED'}"
    )

    # --------------------------------------------------
    # Deterministic equipment gate
    # --------------------------------------------------

    if not equipment_model:

        clarification = (
            "Please select the equipment model "
            "before continuing."
        )

        print(
            "Request status: CLARIFY"
        )

        print(
            f"Clarification question: "
            f"{clarification}"
        )

        return {
            "request_status":
                "CLARIFY",

            "clarification_needed": True,

            "clarification_question":
                clarification,
        }

    # --------------------------------------------------
    # Empty request gate
    # --------------------------------------------------

    if not question:

        clarification = (
            "Please describe the equipment-service "
            "question or issue you need help with."
        )

        print(
            "Request status: CLARIFY"
        )

        print(
            f"Clarification question: "
            f"{clarification}"
        )

        return {
            "request_status":
                "CLARIFY",

            "clarification_needed":
                True,

            "clarification_question":
                clarification,
        }

    # --------------------------------------------------
    # LLM completeness assessment
    # --------------------------------------------------

    prompt = f"""
You are the Request Agent for an AEM
equipment-service assistant.

Selected equipment:
{equipment_model}

Engineer request:
{question}

Clarification continuation:
{"YES" if is_clarification_continuation else "NO"}

Your ONLY task is to determine whether the request
contains enough information to proceed to specialist
planning.

Return READY when:
- the engineer asks a clear documentation question;
- the engineer asks about a specific procedure,
  maintenance task, error, alarm, symptom, component,
  operating condition, or safety issue;
- the request provides enough information for the next
  specialist to begin meaningful retrieval or analysis.

Return CLARIFY only when an important missing detail
prevents meaningful routing or retrieval.

Examples that should be READY:

"What preventive maintenance should be performed?"

"What safety precautions apply before servicing?"

"The equipment is not reaching the temperature setpoint.
What should I check?"

"What does error code E101 mean?"

"Can I bypass the safety interlock?"

"The emergency stop is active. What should I do?"

"Can I open the panel while the equipment is powered?"

Examples that should be CLARIFY:

"It's not working."

"There is an error."

"How do I replace it?"

"What should I do about this?"

Important rules:

1. Do not answer the engineer's question.

2. Do not retrieve documents.

3. Do not diagnose the equipment.

4. Do not select RAG, Diagnostic, or Safety.

5. Do not ask for optional information merely because
   more detail could be useful. READY means there is
   enough information for meaningful routing or retrieval;
   it does NOT mean that every detail required for the
   eventual diagnosis or answer is already known.

6. Ask for clarification only when the missing detail
   prevents meaningful routing or retrieval.

7. SAFETY-CRITICAL INTENT HAS PRIORITY OVER
   CLARIFICATION.

   If the engineer clearly asks to bypass, defeat,
   disable, override, circumvent, or operate around a
   safety control, interlock, emergency stop, guard,
   protective device, or other safety mechanism,
   return READY even if the exact safety device is not
   identified.

   Also return READY when the engineer asks whether a
    specific physical action may be performed under a
    specified equipment state or potentially hazardous
    condition.

    Examples include:
    - opening a panel while the equipment is powered;
    - accessing a protected area while equipment is
    energized;
    - servicing equipment without powering it off;
    - continuing operation while a safety-related condition
    is active.

    The Request Agent does not need to determine whether the
    action is safe. It only determines whether the request
    is sufficiently clear for the Safety Agent to evaluate.

    Do not ask which safety procedure or precaution the
    engineer means when the requested action and equipment
    condition are already clear.

8. Also return READY when the request clearly describes
   an active safety-critical condition, such as an
   emergency stop, active interlock, hazardous-energy
   condition, or powered access request.

9. Do not ask for details such as which interlock,
   which emergency stop, or which safety device merely
   to determine whether the request should proceed to
   specialist planning. The Safety Agent can determine
   whether additional documented information is needed.

10. When evaluating a resolved request after
    clarification, treat the engineer's clarification
    as additional context to the original request.

    Do not ask another clarification merely to obtain a
    numerical value, exact setting, measurement, error
    value, operating parameter, or other optional detail
    when the combined request already identifies a
    meaningful symptom, component, procedure, safety
    issue, or documentation topic.

    Example:

    Original request:
    "It's not working."

    Engineer clarification:
    "The temperature is not reaching the setpoint."

    -> READY

    The exact temperature setpoint is not required for
    the Planning Agent to route this request to
    Diagnostic.   

11. If clarification is required for a non-safety
    request, ask exactly ONE concise question requesting
    the most important missing detail.

Return exactly one of these formats:

STATUS: READY
QUESTION: NONE

or

STATUS: CLARIFY
QUESTION: <one concise clarification question>
"""

    response = client.chat.completions.create(
        model=REQUEST_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You assess whether an equipment-service "
                    "request is sufficiently clear to proceed. "
                    "You do not answer or diagnose it."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    decision = (
        response.choices[0]
        .message.content
        .strip()
    )

    status = "CLARIFY"

    clarification_question = (
        "Please provide more detail about "
        "the equipment-service request."
    )

    for line in decision.splitlines():

        line = line.strip()

        if line.startswith("STATUS:"):

            value = (
                line.split(":", 1)[1]
                .strip()
                .upper()
            )

            if value in {
                "READY",
                "CLARIFY",
            }:
                status = value

        elif line.startswith("QUESTION:"):

            value = (
                line.split(":", 1)[1]
                .strip()
            )

            if (
                value
                and value.upper()
                != "NONE"
            ):
                clarification_question = value

    # --------------------------------------------------
    # Normalize output
    # --------------------------------------------------

    if status == "READY":

        clarification_question = None

    print(
        f"Request status: {status}"
    )

    if clarification_question:

        print(
            "Clarification question: "
            f"{clarification_question}"
        )

    return {
        "request_status":
            status,

        "clarification_needed":
            status == "CLARIFY",

        "clarification_question":
            clarification_question,
    }