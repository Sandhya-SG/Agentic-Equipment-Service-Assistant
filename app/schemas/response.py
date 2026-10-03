from typing import Literal

from pydantic import BaseModel, Field

ChatStatus = Literal["ok", "blocked", "escalated", "unavailable"]


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
    # Filled in once retrieval is connected: citations for the answer.
    sources: list[str] = Field(default_factory=list)
