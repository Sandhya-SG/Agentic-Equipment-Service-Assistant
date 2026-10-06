"""Pydantic schemas for API payloads."""

from typing import Literal

from pydantic import BaseModel, Field, constr

MAX_MESSAGE_CHARS = 2000

# Must match the equipment identifiers in asa.ingestion.metadata.
EquipmentModel = Literal["thermal_station", "thermal_retrofit_1kw"]


class ChatRequest(BaseModel):
    message: constr(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS) = Field(
        ...,
        description="User input to the assistant",
    )
    equipment_model: EquipmentModel | None = Field(
        default=None,
        description="Equipment the question is about. Retrieval is restricted to its manual.",
    )
    conversation_id: str | None = Field(
        default=None,
        max_length=64,
        description="Optional conversation identifier",
    )
    context: str | None = Field(
        default=None,
        max_length=MAX_MESSAGE_CHARS,
        description="Optional context string (not sent to the model yet)",
    )
