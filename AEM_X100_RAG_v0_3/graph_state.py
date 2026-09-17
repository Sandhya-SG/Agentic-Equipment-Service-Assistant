from typing import TypedDict


class AgentState(TypedDict, total=False):

    # Original engineer request
    question: str

    # Request Agent output
    request_type: str

    # Planning Agent output
    plan: str
    planning_reason: str

    # Agentic RAG output
    retrieval_status: str
    retrieval_attempts: int

    answer: str | None

    escalation_required: bool

    senior_used: bool
    senior_decision: str | None
    senior_reason: str | None

    # Diagnostic Agent output
    diagnostic_result: str | None

    # Final response
    final_response: str | None