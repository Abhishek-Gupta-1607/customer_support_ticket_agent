from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError

logger = logging.getLogger(__name__)

# Minimum relevance score (0–1). Chunks scoring below this are considered
# insufficiently related and excluded from results.
_RELEVANCE_THRESHOLD = 0.3


class KnowledgeRetriever:
    """Persistent Chroma retrieval scaffold with stable output contracts."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load documents, build embeddings, and populate ChromaDB.

        Generates stable document IDs from source filename and chunk content
        so that restarting the application does not insert duplicate chunks.
        """
        logger.info("Initializing knowledge retriever …")
        documents = split_support_documents(
            load_support_documents(self.documents_dir)
        )
        embeddings = build_embeddings(self.settings)

        persist_dir = Path(self.settings.vector_db_path)
        persist_dir.mkdir(parents=True, exist_ok=True)

        store = Chroma(
            collection_name=self.settings.rag_collection,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

        # Build stable IDs so duplicate chunks are never inserted on restart.
        ids: list[str] = []
        new_docs = []
        new_metadatas = []
        existing_ids = set(store._collection.get()["ids"]) if store._collection.count() > 0 else set()

        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content
            stable_id = hashlib.sha256(
                f"{source}::{content}".encode("utf-8")
            ).hexdigest()

            if stable_id not in existing_ids:
                ids.append(stable_id)
                new_docs.append(content)
                # Keep only the safe filename; never expose absolute paths.
                new_metadatas.append({"source": Path(source).name})

        if new_docs:
            store.add_texts(texts=new_docs, metadatas=new_metadatas, ids=ids)
            logger.info("Added %d new document chunks to the vector store.", len(new_docs))
        else:
            logger.info("All document chunks already present; skipping insertion.")

        self._store = store
        logger.info("Knowledge retriever initialized successfully.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self, query: str, limit: int | None = None
    ) -> list[dict[str, str]]:
        """Return relevant document chunks as plain dictionaries.

        Each result contains ``content`` and ``source`` keys.  Results are
        ordered from most relevant to least relevant and filtered by the
        configured relevance threshold.
        """
        if not query or not query.strip():
            return []

        if self._store is None:
            raise ComponentNotReadyError(
                "Knowledge retriever has not been initialized"
            )

        k = limit if limit is not None else self.settings.rag_top_k

        try:
            scored_results = self._store.similarity_search_with_relevance_scores(
                query.strip(), k=k
            )
        except Exception:
            logger.exception("Similarity search failed for query: %s", query[:80])
            return []

        # Filter by relevance threshold and deduplicate by content.
        seen_content: set[str] = set()
        results: list[dict[str, str]] = []

        for doc, score in scored_results:
            if score < _RELEVANCE_THRESHOLD:
                continue
            content = doc.page_content
            if content in seen_content:
                continue
            seen_content.add(content)
            source = Path(doc.metadata.get("source", "unknown")).name
            results.append({"content": content, "source": source})

        logger.debug(
            "Search returned %d relevant chunks for: %s", len(results), query[:60]
        )
        return results
