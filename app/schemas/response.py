from typing import Literal

from pydantic import BaseModel, Field

# ok            answered from the manual
# clarification the assistant needs more input (see `response`)
# halted        the manual prohibits the requested action
# escalated     evidence was insufficient or the task is hazardous: a human must take over
# blocked       input guardrail rejected the request
# unavailable   the model or index could not be reached
ChatStatus = Literal["ok", "clarification", "halted", "escalated", "blocked", "unavailable"]


class SafetyInfo(BaseModel):
    hazards: list[str] = Field(default_factory=list)
    ppe_required: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    warning: str | None = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    status: ChatStatus
    safety: SafetyInfo = Field(default_factory=SafetyInfo)
    # Manual passages the answer was grounded in, e.g. "file.pdf, p.117, 11.1 Preventive Maintenance".
    sources: list[str] = Field(default_factory=list)
    specialist: str | None = Field(default=None, description="Agent that handled the request")
    escalation_reason: str | None = None
    run_id: str | None = Field(default=None, description="Audit-trail correlation id")
