"""Request Agent for engineer-request completeness assessment."""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv

from asa.components.llm import client
from asa.components.tracing import llm_name

load_dotenv()


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

    question = (state.get("resolved_query") or state.get("raw_query") or "").strip()

    is_clarification_continuation = bool(state.get("resolved_query"))

    equipment_model = (state.get("equipment_model") or "").strip()

    print(f"Engineer request: {question}")

    print(f"Equipment model: {equipment_model or 'NOT SELECTED'}")

    # --------------------------------------------------
    # Deterministic equipment gate
    # --------------------------------------------------

    if not equipment_model:
        clarification = "Please select the equipment model before continuing."

        print("Request status: CLARIFY")

        print(f"Clarification question: {clarification}")

        return {
            "request_status": "CLARIFY",
            "clarification_needed": True,
            "clarification_question": clarification,
        }

    # --------------------------------------------------
    # Empty request gate
    # --------------------------------------------------

    if not question:
        clarification = "Please describe the equipment-service question or issue you need help with."

        print("Request status: CLARIFY")

        print(f"Clarification question: {clarification}")

        return {
            "request_status": "CLARIFY",
            "clarification_needed": True,
            "clarification_question": clarification,
        }

    # --------------------------------------------------
    # LLM completeness assessment
    # --------------------------------------------------

    prompt = f"""
You are the Request Agent of an AEM equipment-service assistant. You only decide whether the request is clear
enough to look up in the equipment manual. Do not answer, diagnose, retrieve documents or choose a specialist.

Selected equipment:
{equipment_model}

Engineer request:
{question}

Clarification continuation:
{"YES" if is_clarification_continuation else "NO"}

The default answer is READY.

READY: the request names an identifiable subject: a topic, requirement, specification, procedure, task, component,
error, alarm, symptom, operating condition, hazard or safety topic. A broad topic is still a subject: the manual
holds the details. Never ask which requirements, steps, values or items the engineer means, and never ask for
optional detail such as a value, measurement or setting.

CLARIFY: only when no subject can be identified at all. A pronoun ("it", "this", "that") is not a subject, and
neither are the bare words "checks", "steps", "procedure" or "information" ("What checks should I do?", "How do I
replace it?", "There is an error").

SAFETY-CRITICAL INTENT HAS PRIORITY OVER CLARIFICATION. Return READY, even if the exact device is not named, when
the engineer asks to bypass, defeat, disable, override or work around a safety control (interlock, emergency
stop, guard); asks whether a physical action is allowed in a hazardous state (opening a panel while powered,
servicing without powering off, working on live equipment, continuing operation while a safety condition is
active); or describes an active safety condition (emergency stop, active interlock, hazardous energy). The
Safety Agent judges it. Never ask which device or precaution.

CONTINUATION (only when "Clarification continuation" is YES): the request holds the original text and the
engineer's clarification together, and the clarification supplies what "it" or "this" was. If together they name a
subject, return READY and do not ask for a procedure, value or detail. If the reply adds no subject (for example
"yes", "the machine"), return CLARIFY again.
Continuation examples: original "It's not working." with clarification "The temperature is not reaching the
setpoint." is READY. Original "How do I replace it?" with clarification "the heating element" is READY. Original
"It's not working." with clarification "yes" is CLARIFY.

READY examples (not continuations):
- "What are the power requirements?"
- "What PPE is required?"
- "How do I start the system?"
- "What does error code E101 mean?"
- "The equipment is not reaching the temperature setpoint. What should I check?"
- "What are the semi annual preventive maintenance checks?"
- "Can I bypass the safety interlock?"
- "The emergency stop is active. What should I do?"
- "Can I open the panel while the equipment is powered?"

CLARIFY examples (not continuations, no subject):
- "It's not working."
- "There is an error."
- "How do I replace it?"
- "What should I do about this?"
- "What checks should I do?"

Return exactly one of these formats:

STATUS: READY
QUESTION: NONE

or

STATUS: CLARIFY
QUESTION: <one concise question asking for the most important missing detail>
"""

    response = client.chat.completions.create(
        **llm_name("assess-request-readiness"),
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

    decision = response.choices[0].message.content.strip()

    status = "CLARIFY"

    clarification_question = "Please provide more detail about the equipment-service request."

    for line in decision.splitlines():
        line = line.strip()

        if line.startswith("STATUS:"):
            value = line.split(":", 1)[1].strip().upper()

            if value in {
                "READY",
                "CLARIFY",
            }:
                status = value

        elif line.startswith("QUESTION:"):
            value = line.split(":", 1)[1].strip()

            if value and value.upper() != "NONE":
                clarification_question = value

    # --------------------------------------------------
    # Normalize output
    # --------------------------------------------------

    if status == "READY":
        clarification_question = None

    print(f"Request status: {status}")

    if clarification_question:
        print(f"Clarification question: {clarification_question}")

    return {
        "request_status": status,
        "clarification_needed": status == "CLARIFY",
        "clarification_question": clarification_question,
    }
