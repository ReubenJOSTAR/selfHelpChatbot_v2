"""
WHY a dedicated dependencies.py:

FastAPI's dependency injection system lets you declare what a route needs,
and FastAPI provides it automatically.

This file creates the services ONCE (at startup) and returns the same
instance on every request. This is called the "singleton pattern."

Without this: every request would create a new MongoDB connection and
a new OpenAI embeddings model — extremely slow and wasteful.

With this: connection is created once, reused forever.
"""

from functools import lru_cache

from app.core.config import get_settings
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService


@lru_cache()
def get_retrieval_service() -> RetrievalService:
    """
    Returns a single RetrievalService instance.
    lru_cache ensures this is only created once for the lifetime of the server.
    """
    return RetrievalService(settings=get_settings())


@lru_cache()
def get_generation_service() -> GenerationService:
    """
    Returns a single GenerationService instance.
    """
    return GenerationService(settings=get_settings())
