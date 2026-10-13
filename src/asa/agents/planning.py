"""Planning Agent for specialist routing."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from asa.components.tracing import llm_name


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

PLANNER_MODEL = "gpt-4o-mini"


def planning_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Select the specialist agent that should handle
    a READY engineer request.

    Routing priority:
        SAFETY > DIAGNOSTIC > AGENTIC_RAG
    """

    print("\n[Planning Agent]")

    # --------------------------------------------------
    # Guard: Planner should only receive READY requests
    # --------------------------------------------------

    request_status = state.get(
        "request_status"
    )

    if request_status != "READY":

        raise ValueError(
            "Planning Agent received a request "
            "that is not READY."
        )

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
            "Planning Agent requires a query."
        )

    if not equipment_model:
        raise ValueError(
            "Planning Agent requires equipment_model."
        )

    prompt = f"""
You are the Planning Agent for an AEM
equipment-service assistant.

Selected equipment:
{equipment_model}

Engineer request:
{question}

Choose exactly ONE specialist agent.

SAFETY
Use SAFETY when the engineer is going to DO something
or is FACING something where safety controls must take
priority:
- asks whether an action is allowed, or how to perform
  it safely (servicing, opening, accessing, bypassing,
  defeating, overriding, operating, lockout);
- describes a safety-critical condition that is active
  now (emergency stop, tripped interlock, hazardous
  energy present, unsafe operating state);
- asks what precautions apply before or during a task.

Examples:
- "Can I bypass the safety interlock?"
- "The emergency stop is active. What should I do?"
- "Can I open the panel while the equipment is powered?"
- "What safety precautions apply before servicing?"
- "I need to replace the heater. What lockout steps must I follow?"
- "Can I service the unit without wearing gloves?"

A safety WORD alone (hazard, hazardous energy, PPE,
interlock, safety device) does not make a request
SAFETY. When the engineer only asks what the manual
lists, specifies or documents about a safety topic, with
no task being performed and no hazardous condition
active, use AGENTIC_RAG.

DIAGNOSTIC
Use DIAGNOSTIC when the engineer describes an observed
equipment problem, symptom, abnormal behaviour,
performance issue, recurring fault, or troubleshooting
situation and needs help determining what evidence or
checks should be considered.

Examples:
- "The equipment is not reaching the temperature setpoint."
- "The temperature keeps dropping during operation."
- "The machine stops during operation. What should I check?"
- "The same alarm keeps returning after restart."

AGENTIC_RAG
Use AGENTIC_RAG when the engineer primarily asks for
documented information, such as:
- preventive maintenance requirements;
- operating procedures;
- documented component information;
- error or alarm meaning;
- manual/reference information;
- documented servicing information;
- documented safety information that is only being
  looked up: hazard listings, hazardous energies, PPE
  lists, safety devices, warning labels and symbols.

Examples:
- "What preventive maintenance should be performed?"
- "What does error code E101 mean?"
- "What is the documented startup procedure?"
- "What does the manual say about the thermal controller?"
- "List all the potentially hazardous energy sources."
- "What PPE is required?"
- "What safety devices does the system have?"

ROUTING PRIORITY:

1. SAFETY has the highest priority.

If a request contains both a troubleshooting symptom
and a safety-critical condition or unsafe requested
action, choose SAFETY. When it is unclear whether the
engineer is about to perform a task, choose SAFETY.

2. DIAGNOSTIC has priority over AGENTIC_RAG when the
engineer is describing an actual observed equipment
problem and asking what may be wrong or what should
be checked.

3. Otherwise choose AGENTIC_RAG for documented
information requests.

Important rules:

1. Do not answer the engineer's question.

2. Do not retrieve documents.

3. Do not diagnose the equipment.

4. Do not provide safety instructions.

5. Choose exactly one specialist.

Return exactly:

PLAN: SAFETY or DIAGNOSTIC or AGENTIC_RAG
REASON: <one brief reason>
"""

    response = client.chat.completions.create(
        **llm_name("plan-specialist-route"),
        model=PLANNER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You route READY AEM equipment-service "
                    "requests to exactly one specialist agent. "
                    "Safety-critical requests have priority."
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

    # --------------------------------------------------
    # Conservative default
    # --------------------------------------------------

    selected = "safety"

    reason = (
        "Planner output could not be parsed; "
        "routing conservatively to Safety."
    )

    for line in decision.splitlines():

        line = line.strip()

        if line.startswith("PLAN:"):

            value = (
                line.split(":", 1)[1]
                .strip()
                .upper()
            )

            if value == "SAFETY":
                selected = "safety"

            elif value == "DIAGNOSTIC":
                selected = "diagnostic"

            elif value == "AGENTIC_RAG":
                selected = "agentic_rag"

        elif line.startswith("REASON:"):

            reason = (
                line.split(":", 1)[1]
                .strip()
            )

    print(
        f"Selected specialist: {selected}"
    )

    print(
        f"Planning reason: {reason}"
    )

    return {
        "plan": [
            selected
        ],

        "current_step":
            selected,

        "planning_reason":
            reason,

        "iteration_count":
            state.get(
                "iteration_count",
                0,
            ) + 1,
    }