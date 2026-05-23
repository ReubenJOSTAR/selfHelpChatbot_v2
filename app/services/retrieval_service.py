# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
# from pymongo import MongoClient

# from app.core.config import Settings
# from app.models.schemas import ScriptureResult


# class RetrievalService:
#     """
#     Wraps MongoDB Atlas vector search.

#     WHY a class instead of bare functions:
#     - The vector store connection is expensive to create (network call, embedding model init)
#     - As a class, we create it ONCE at startup and reuse it on every request
#     - Dependency injection (used in routes) makes this easy to swap for testing

#     Your existing retrieval logic is preserved here — just moved into a clean service layer.
#     """

#     def __init__(self, settings: Settings):
#         self._settings = settings
#         self._vs = self._connect()

#     def _connect(self) -> MongoDBAtlasVectorSearch:
#         """
#         Build the vector store connection once.
#         Mirrors your existing connect_vector_store() exactly.
#         """
#         client = MongoClient(self._settings.MONGODB_URI)

#         collection = client[
#             self._settings.DATABASE_NAME
#         ][
#             self._settings.COLLECTION_NAME
#         ]

#         # embeddings = OpenAIEmbeddings(
#         #     openai_api_key=self._settings.OPENAI_API_KEY
#         # )
#         embeddings = OpenAIEmbeddings()

#         return MongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name=self._settings.VECTOR_INDEX_NAME,
#         )

#     async def retrieve(
#         self,
#         query: str,
#         feeling_filter: str | None = None,
#         k: int = 4,
#     ) -> list[ScriptureResult]:
#         """
#         Semantic retrieval with optional metadata filtering.

#         WHY async:
#         FastAPI is async. Even though MongoDBAtlasVectorSearch is sync under the hood,
#         wrapping it here means we can swap to a true async driver later without
#         changing the route layer at all.

#         Returns typed ScriptureResult objects — not raw LangChain docs.
#         The route layer never touches raw docs directly.
#         """
#         pre_filter: dict = {}
#         if feeling_filter:
#             # Normalize: "anxiety" → "Anxiety" — matches your dataset's title-cased values
#             pre_filter["feeling"] = feeling_filter.strip().title()

#         raw_docs = self._vs.similarity_search_with_score(
#             query,
#             k=k,
#             pre_filter=pre_filter if pre_filter else None,
#         )

#         results = []
#         for doc, score in raw_docs:
#             results.append(
#                 ScriptureResult(
#                     reference=doc.metadata.get("reference", "Unknown"),
#                     feeling=doc.metadata.get("feeling"),
#                     categories=doc.metadata.get("categories"),
#                     content=doc.page_content[:800],
#                     similarity_score=round(score, 4),
#                 )
#             )

#         return results
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings
from app.models.schemas import ScriptureResult


class RetrievalService:
    """
    Wraps MongoDB Atlas vector search.
    Created once at startup via lru_cache in dependencies.py.
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

        # OpenAIEmbeddings in langchain-openai 0.3.x
        # - Reads OPENAI_API_KEY from environment automatically.
        # - No proxies error: 0.3.x uses the openai 1.x client cleanly.
        # - model pinned explicitly to avoid surprises on OpenAI default changes.
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )

        # langchain-mongodb replaces langchain-community MongoDBAtlasVectorSearch.
        # Same constructor API — drop-in replacement.
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