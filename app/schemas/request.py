from pydantic import BaseModel, Field, constr


class ChatRequest(BaseModel):
    message: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="User input to the assistant",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation identifier",
    )
    context: str | None = Field(
        default=None,
        description="Optional context string",
    )
