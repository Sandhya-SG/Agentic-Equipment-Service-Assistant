"""Pydantic schemas for the feedback endpoint."""

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    run_id: str = Field(pattern=r"^[0-9a-f]{12}$", description="The run_id returned with the answer being rated")
    helpful: bool
    comment: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    recorded: bool
