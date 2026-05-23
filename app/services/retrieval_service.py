# 
from pymongo import MongoClient

# ─── THE KEY CHANGE ───────────────────────────────────────────
# OLD: from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# NEW: from langchain_mongodb import MongoDBAtlasVectorSearch
#
# WHY: langchain-community's MongoDBAtlasVectorSearch is deprecated and was
# written against older LangChain internals. langchain-mongodb is the official
# first-party replacement maintained by MongoDB. Same API surface — it's a
# drop-in for our use case.
# ──────────────────────────────────────────────────────────────
from langchain_mongodb import MongoDBAtlasVectorSearch

# langchain-openai>=0.2 initializes OpenAIEmbeddings using the new
# openai>=1.40 client interface — no `proxies` argument, no crash.
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings
from app.models.schemas import ScriptureResult


class RetrievalService:
    """
    Wraps MongoDB Atlas vector search.

    Connection is created once at startup via dependency injection
    and reused on every request.
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

        # ─── OpenAIEmbeddings — no kwargs needed ─────────────────
        # With langchain-openai>=0.2 + openai>=1.40, the client is
        # initialized internally without the `proxies` argument.
        # Do NOT pass openai_api_key here — pydantic-settings reads
        # it from the environment automatically via OPENAI_API_KEY.
        # Passing it explicitly can trigger validation conflicts in
        # some pydantic v1/v2 bridge scenarios.
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"   # explicit is better than implicit
        )

        # ─── langchain-mongodb vectorstore ───────────────────────
        # MongoDBAtlasVectorSearch from langchain_mongodb accepts the
        # same constructor arguments as the old community version.
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