from functools import lru_cache
import logging

from app.core.config import get_settings
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService

logger = logging.getLogger(__name__)


@lru_cache()
def get_retrieval_service() -> RetrievalService:
    try:
        return RetrievalService(settings=get_settings())
    except Exception:
        logger.exception("Failed to create RetrievalService")
        raise


@lru_cache()
def get_generation_service() -> GenerationService:
    try:
        return GenerationService(settings=get_settings())
    except Exception:
        logger.exception("Failed to create GenerationService")
        raise
