from pydantic import BaseModel, Field
from typing import Optional


# -----------------------------
# Request Schemas
# -----------------------------

class ChatRequest(BaseModel):
    """
    What the frontend sends to /api/chat

    WHY feeling is Optional:
    Users may not know what "feeling" category to pick.
    The semantic search handles emotional matching automatically.
    Feeling filter is a precision booster, not a requirement.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's question or emotional expression",
        examples=["I feel overwhelmed with anxiety about my future"]
    )
    feeling_filter: Optional[str] = Field(
        default=None,
        description="Optional emotional category filter (e.g. Anxiety, Hope, Fear)",
        examples=["Anxiety"]
    )


# -----------------------------
# Response Schemas
# -----------------------------

class ScriptureResult(BaseModel):
    """
    A single retrieved scripture with its metadata.

    WHY include similarity_score:
    Useful for debugging retrieval quality.
    Frontend can hide this from users but developers need it.
    """
    reference: str
    feeling: Optional[str]
    categories: Optional[list[str]]
    content: str
    similarity_score: float


class ChatResponse(BaseModel):
    """
    What the backend returns to the frontend.

    WHY return both scriptures AND the generated response:
    - Transparency: users can see the source verses
    - Trust: grounding is visible, not hidden
    - Debugging: developers can inspect retrieval quality
    - Frontend flexibility: UI can choose what to display
    """
    query: str
    feeling_filter: Optional[str]
    scriptures: list[ScriptureResult]
    response: str


class HealthResponse(BaseModel):
    status: str
    message: str
