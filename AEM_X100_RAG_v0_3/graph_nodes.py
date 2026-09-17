import os

from dotenv import load_dotenv
from openai import OpenAI

from agentic_rag import AgenticRAGAgent

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

planner_model = "gpt-4o-mini"

rag_agent = AgenticRAGAgent()

def request_node(state):
    """
    Initial Request Agent node.

    v0.3.0 starts deterministically.
    Later this becomes an LLM-based Request Agent.
    """

    question = state["question"]

    print("\n[Request Node]")
    print(f"Engineer question: {question}")

    return {
        "request_type": "equipment_service"
    }


def planning_node(state):
    """
    LLM Planning Agent.

    Determines which specialist agent should handle
    the engineer's request.
    """

    print("\n[Planning Agent]")

    question = state["question"]

    prompt = f"""
You are the Planning Agent for an AEM-X100
field-service assistant.

Engineer request:
{question}

Choose exactly one specialist:

AGENTIC_RAG
Use this when the engineer primarily needs documented
information, such as:
- procedure information
- error-code meaning
- maintenance information
- safety documentation
- escalation procedure
- equipment reference information

DIAGNOSTIC
Use this when the engineer describes an equipment
problem, symptom, abnormal behaviour, recurring fault,
or troubleshooting situation and needs help determining
what may be wrong or what evidence should be checked.

Examples:

"What does error code E101 mean?"
→ AGENTIC_RAG

"What preventive maintenance should be performed?"
→ AGENTIC_RAG

"What should happen if a procedure is unavailable?"
→ AGENTIC_RAG

"The wafer is not moving properly. What should I check?"
→ DIAGNOSTIC

"E201 keeps appearing after I retry the transfer."
→ DIAGNOSTIC

Return exactly:

PLAN: AGENTIC_RAG or DIAGNOSTIC
REASON: <brief reason>
"""

    response = client.chat.completions.create(
        model=planner_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You route field-service requests "
                    "to the appropriate specialist agent."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    decision = (
        response.choices[0]
        .message.content
        .strip()
    )

    plan = "agentic_rag"
    reason = ""

    for line in decision.splitlines():

        line = line.strip()

        if line.startswith("PLAN:"):

            value = (
                line.split(":", 1)[1]
                .strip()
                .upper()
            )

            if value == "DIAGNOSTIC":
                plan = "diagnostic"

            elif value == "AGENTIC_RAG":
                plan = "agentic_rag"

        elif line.startswith("REASON:"):

            reason = (
                line.split(":", 1)[1]
                .strip()
            )

    print(f"Selected plan: {plan}")
    print(f"Planning reason: {reason}")

    return {
        "plan": plan,
        "planning_reason": reason
    }

def diagnostic_node(state):
    """
    Diagnostic Agent.

    Uses documented evidence from the existing
    Agentic RAG component and presents it from
    a diagnostic/troubleshooting perspective.
    """

    print("\n[Diagnostic Agent]")

    question = state["question"]

    print(
        "Diagnostic Agent requesting "
        "documented evidence..."
    )

    rag_result = rag_agent.run(
        question,
        k=5
    )

    # ----------------------------------------
    # RAG could not support diagnosis
    # ----------------------------------------

    if rag_result["escalation_required"]:

        reason = (
            rag_result.get("senior_reason")
            or "Insufficient documented evidence."
        )

        diagnostic_result = (
            "A documented diagnosis cannot be made "
            "from the available evidence. "
            "Human escalation is required.\n\n"
            f"Reason: {reason}"
        )

        return {
            "diagnostic_result": diagnostic_result,
            "retrieval_status":
                rag_result["retrieval_status"],
            "retrieval_attempts":
                rag_result["retrieval_attempts"],
            "escalation_required": True,
            "senior_used":
                rag_result["senior_used"],
            "senior_decision":
                rag_result["senior_decision"],
            "senior_reason":
                rag_result["senior_reason"]
        }

    # ----------------------------------------
    # RAG provided documented evidence
    # ----------------------------------------

    evidence_answer = rag_result["answer"]

    prompt = f"""
You are the Diagnostic Agent for an AEM-X100
field-service assistant.

Engineer problem:
{question}

The Agentic RAG component produced this grounded
evidence-based response:

{evidence_answer}

Using ONLY this documented evidence, produce a concise
diagnostic assessment.

Rules:
1. Do not introduce equipment facts not contained in
   the evidence.
2. Do not invent a root cause.
3. Distinguish observations/checks from confirmed causes.
4. Preserve error codes, component names, and acronyms.
5. Do not recommend bypassing interlocks or safety controls.
6. If the evidence does not establish a root cause,
   explicitly say that the cause is not yet confirmed.

Return:

Diagnostic assessment:
<assessment>

Recommended documented checks:
<checks>
"""

    response = client.chat.completions.create(
        model=planner_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a conservative equipment "
                    "diagnostic agent. Diagnose only from "
                    "provided documented evidence."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    diagnostic_result = (
        response.choices[0]
        .message.content
        .strip()
    )

    print("\nDiagnostic result:")
    print(diagnostic_result)

    return {
        "diagnostic_result":
            diagnostic_result,

        "retrieval_status":
            rag_result["retrieval_status"],

        "retrieval_attempts":
            rag_result["retrieval_attempts"],

        "escalation_required": False,

        "senior_used":
            rag_result["senior_used"],

        "senior_decision":
            rag_result["senior_decision"],

        "senior_reason":
            rag_result["senior_reason"]
    }

def agentic_rag_node(state):
    """
    Execute the existing Agentic RAG component.
    """

    print("\n[Agentic RAG Node]")

    question = state["question"]

    result = rag_agent.run(
        question,
        k=5
    )

    return {
        "retrieval_status":
            result["retrieval_status"],

        "retrieval_attempts":
            result["retrieval_attempts"],

        "answer":
            result["answer"],

        "escalation_required":
            result["escalation_required"],

        "senior_used":
            result["senior_used"],

        "senior_decision":
            result["senior_decision"],

        "senior_reason":
            result["senior_reason"]
    }


def response_node(state):
    """
    Prepare the final engineer-facing response.
    """

    print("\n[Response Node]")

    if state.get("escalation_required"):

        reason = state.get(
            "senior_reason",
            "Insufficient documented evidence."
        )

        response = (
            "The available documented evidence is "
            "insufficient to answer this request safely. "
            "Human escalation is required.\n\n"
            f"Reason: {reason}"
        )

    else:

        response = state.get(
            "answer",
            "No answer was generated."
        )

    return {
        "final_response": response
    }