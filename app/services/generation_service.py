from openai import AsyncOpenAI

from app.core.config import Settings
from app.models.schemas import ScriptureResult


# -----------------------------
# Prompt Engineering
# -----------------------------

SYSTEM_PROMPT = """You are a compassionate Christian AI assistant called WalkWithJesus.

Your role is to offer Scripture-based encouragement and gentle spiritual guidance.

Rules you must always follow:
- Ground every response in the scripture passages provided to you.
- Never invent, paraphrase, or fabricate Bible verses.
- Keep each answer focused on the user's expressed feelings and needs.
- If the user expresses a specific feeling, address that feeling directly.
- Design every answer to be unique.
- If the provided scripture doesn't fully address the question, say so humbly.
- Speak with warmth, empathy, and calm — like a trusted pastor or spiritual friend.
- Keep responses focused and meaningful — not overly long.
- End every response with a short, sincere prayer (2-3 sentences) related to the person's need."""


def _build_context(scriptures: list[ScriptureResult]) -> str:
    """
    Formats retrieved ScriptureResult objects into a clean context block
    for injection into the LLM prompt.

    WHY a separate function:
    Prompt context construction is its own concern.
    Easy to tweak formatting without touching the API call logic.
    """
    parts = []
    for i, s in enumerate(scriptures, 1):
        parts.append(
            f"[Scripture {i}]\n"
            f"Reference: {s.reference}\n"
            f"Feeling: {s.feeling or 'General'}\n"
            f"Content: {s.content}"
        )
    return "\n\n".join(parts)


def _build_user_prompt(query: str, scriptures: list[ScriptureResult]) -> str:
    context = _build_context(scriptures)
    return (
        f'A person came to you expressing: "{query}"\n\n'
        f"Here are the relevant Scripture passages retrieved for this situation:\n\n"
        f"{context}\n\n"
        f"Based only on these scriptures, offer them compassionate Christian guidance "
        f"and encouragement. Close with a short prayer."
    )


# -----------------------------
# Generation Service
# -----------------------------

class GenerationService:
    """
    Handles OpenAI LLM calls.

    WHY async (AsyncOpenAI):
    FastAPI is fully async. Using AsyncOpenAI means the server can handle
    other requests while waiting for OpenAI to respond — not blocked.
    This matters as soon as you have more than one user.

    WHY separate from RetrievalService:
    Single responsibility. Retrieval and generation are independent concerns.
    You can swap the LLM (Claude, Gemini, local model) without touching retrieval.
    You can swap the vector DB without touching generation.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate(
        self,
        query: str,
        scriptures: list[ScriptureResult],
    ) -> str:
        """
        Takes the user query and retrieved scriptures.
        Returns a grounded, compassionate guidance response.
        """
        if not scriptures:
            return (
                "I wasn't able to find specific scripture for your situation right now. "
                "Please know that God sees you and cares deeply for you. "
                "Consider speaking with a pastor or trusted spiritual mentor for guidance."
            )

        user_prompt = _build_user_prompt(query, scriptures)

        response = await self._client.chat.completions.create(
            model=self._settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._settings.LLM_TEMPERATURE,
            max_tokens=self._settings.LLM_MAX_TOKENS,
        )

        return response.choices[0].message.content
