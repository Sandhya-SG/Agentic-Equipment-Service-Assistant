from pydantic import BaseModel, Field, constr

MAX_MESSAGE_CHARS = 2000


class ChatRequest(BaseModel):
    message: constr(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS) = Field(
        ...,
        description="User input to the assistant",
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
