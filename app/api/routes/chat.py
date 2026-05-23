from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_retrieval_service, get_generation_service
from app.models.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.core.config import get_settings
import traceback

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Simple health check endpoint.

    WHY this exists:
    - Frontend can ping this to confirm the backend is alive before sending requests
    - Useful for deployment monitoring (Docker, cloud health checks)
    - Takes zero resources to implement now, saves hours of debugging later
    """
    return HealthResponse(
        status="ok",
        message="WalkWithJesus API is running ✝️"
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    retrieval: RetrievalService = Depends(get_retrieval_service),
    generation: GenerationService = Depends(get_generation_service),
):
    """
    Main RAG endpoint. Orchestrates the full pipeline:

    1. Receive user query
    2. Retrieve semantically relevant scriptures
    3. Pass scriptures + query to LLM
    4. Return structured response

    WHY Depends():
    FastAPI automatically injects the retrieval and generation services.
    The route function doesn't create anything — it just orchestrates.
    This makes the route easy to test (swap real services for mocks).

    WHY this route is thin:
    Routes should only handle HTTP concerns (request parsing, status codes, errors).
    Business logic lives in services. This is the single most important
    architectural rule for a maintainable backend.
    """
    settings = get_settings()

    # --- Step 1: Retrieve relevant scriptures ---
    try:
        scriptures = await retrieval.retrieve(
            query=request.query,
            feeling_filter=request.feeling_filter,
            k=settings.RETRIEVAL_TOP_K,
        )
    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=503,
            detail=f"Retrieval failed: {str(e)}"
        )

    # --- Step 2: Generate compassionate response ---
    try:
        response_text = await generation.generate(
            query=request.query,
            scriptures=scriptures,
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Generation failed: {str(e)}"
        )

    # --- Step 3: Return structured response ---
    return ChatResponse(
        query=request.query,
        feeling_filter=request.feeling_filter,
        scriptures=scriptures,
        response=response_text,
    )
