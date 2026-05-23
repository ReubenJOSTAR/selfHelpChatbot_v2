# 
from pymongo import MongoClient

# ── THE CORRECT IMPORTS FOR LANGCHAIN 1.x ─────────────────────
#
# OLD (0.3.x era, deprecated):
#   from langchain_community.vectorstores import MongoDBAtlasVectorSearch
#
# ALSO OLD (0.2.x era, what previous advice told you):
#   from langchain_mongodb import MongoDBAtlasVectorSearch  # version 0.3.0
#
# CORRECT (LangChain 1.x era):
#   from langchain_mongodb import MongoDBAtlasVectorSearch  # version 0.10.0
#
# The import path is identical — only the package version changes.
# ──────────────────────────────────────────────────────────────
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings
from app.models.schemas import ScriptureResult


class RetrievalService:
    """
    Wraps MongoDB Atlas vector search.

    Instance is created once at startup via dependency injection
    (lru_cache in dependencies.py) and reused on every request.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._vs = self._connect()

    def _connect(self) -> MongoDBAtlasVectorSearch:
        client = MongoClient(self._settings.MONGODB_URI)

        collection = client[
            self._settings.DATABASE_NAME
        ][
            self._settings.COLLECTION_NAME
        ]

        # ── OpenAIEmbeddings in LangChain 1.x ─────────────────
        # - No openai_api_key kwarg needed: pydantic-settings reads
        #   OPENAI_API_KEY from the environment automatically.
        # - No proxies error: langchain-openai 1.x uses the
        #   openai 1.8x+ client interface cleanly.
        # - model is pinned explicitly to avoid surprises if OpenAI
        #   changes the default in a future API update.
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )

        return MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=self._settings.VECTOR_INDEX_NAME,
        )

    async def retrieve(
        self,
        query: str,
        feeling_filter: str | None = None,
        k: int = 4,
    ) -> list[ScriptureResult]:
        """
        Semantic retrieval with optional metadata filtering.
        """
        pre_filter: dict = {}
        if feeling_filter:
            pre_filter["feeling"] = feeling_filter.strip().title()

        raw_docs = self._vs.similarity_search_with_score(
            query,
            k=k,
            pre_filter=pre_filter if pre_filter else None,
        )

        return [
            ScriptureResult(
                reference=doc.metadata.get("reference", "Unknown"),
                feeling=doc.metadata.get("feeling"),
                categories=doc.metadata.get("categories"),
                content=doc.page_content[:800],
                similarity_score=round(score, 4),
            )
            for doc, score in raw_docs
        ]