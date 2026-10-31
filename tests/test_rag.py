import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.retriever import KnowledgeRetriever
from src.utils.errors import ComponentNotReadyError


def test_loader_preserves_source_names(knowledge_dir: Path) -> None:
    documents = load_support_documents(knowledge_dir)
    assert {item.metadata["source"] for item in documents} == {
        "accounts.md",
        "payments.md",
        "returns.md",
        "shipping.md",
    }


def test_splitter_keeps_source_metadata(knowledge_dir: Path) -> None:
    chunks = split_support_documents(load_support_documents(knowledge_dir))
    assert chunks
    assert all(chunk.metadata.get("source", "").endswith(".md") for chunk in chunks)


def test_loader_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_support_documents(tmp_path / "nonexistent")


def test_loader_rejects_empty_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_docs"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No Markdown"):
        load_support_documents(empty_dir)


class FakeTestEmbeddings:
    """Deterministic fake embedding adapter for fast offline testing."""

    def _normalize(self, text: str) -> list[float]:
        import math
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        raw = [int(c, 16) / 15.0 for c in h[:32]]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._normalize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._normalize(text)


def make_test_settings(vdb_path: Path) -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(vdb_path),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-rag",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


@pytest.mark.asyncio
async def test_retriever_initialization(knowledge_dir: Path, tmp_path: Path) -> None:
    settings = make_test_settings(tmp_path / "vdb_init")
    retriever = KnowledgeRetriever(settings, knowledge_dir)

    with patch("src.rag.retriever.build_embeddings", return_value=FakeTestEmbeddings()):
        await retriever.initialize()

    assert retriever._store is not None
    assert retriever._store._collection.count() > 0


@pytest.mark.asyncio
async def test_retriever_initialization_idempotency(
    knowledge_dir: Path, tmp_path: Path
) -> None:
    settings = make_test_settings(tmp_path / "vdb_idem")
    retriever1 = KnowledgeRetriever(settings, knowledge_dir)

    with patch("src.rag.retriever.build_embeddings", return_value=FakeTestEmbeddings()):
        await retriever1.initialize()
        count1 = retriever1._store._collection.count()

        retriever2 = KnowledgeRetriever(settings, knowledge_dir)
        await retriever2.initialize()
        count2 = retriever2._store._collection.count()

    assert count1 == count2


@pytest.mark.asyncio
async def test_search_before_initialization_raises_error(
    knowledge_dir: Path, tmp_path: Path
) -> None:
    settings = make_test_settings(tmp_path / "vdb_uninit")
    retriever = KnowledgeRetriever(settings, knowledge_dir)

    with pytest.raises(ComponentNotReadyError):
        await retriever.search("shipping")


@pytest.mark.asyncio
async def test_search_blank_query_returns_empty_list(
    knowledge_dir: Path, tmp_path: Path
) -> None:
    settings = make_test_settings(tmp_path / "vdb_blank")
    retriever = KnowledgeRetriever(settings, knowledge_dir)

    with patch("src.rag.retriever.build_embeddings", return_value=FakeTestEmbeddings()):
        await retriever.initialize()
        results = await retriever.search("   ")

    assert results == []


@pytest.mark.asyncio
async def test_search_relevant_retrieval_and_metadata(
    knowledge_dir: Path, tmp_path: Path
) -> None:
    settings = make_test_settings(tmp_path / "vdb_search")
    retriever = KnowledgeRetriever(settings, knowledge_dir)

    with patch("src.rag.retriever.build_embeddings", return_value=FakeTestEmbeddings()):
        with patch("src.rag.retriever._RELEVANCE_THRESHOLD", 0.0):
            await retriever.initialize()
            results = await retriever.search("standard shipping delivery", limit=2)

    assert len(results) <= 2
    for r in results:
        assert "content" in r
        assert "source" in r
        assert r["source"].endswith(".md")
        assert "/" not in r["source"] and "\\" not in r["source"]


@pytest.mark.asyncio
async def test_search_irrelevant_query_threshold(
    knowledge_dir: Path, tmp_path: Path
) -> None:
    settings = make_test_settings(tmp_path / "vdb_irrel")
    retriever = KnowledgeRetriever(settings, knowledge_dir)

    with patch("src.rag.retriever.build_embeddings", return_value=FakeTestEmbeddings()):
        with patch("src.rag.retriever._RELEVANCE_THRESHOLD", 0.999):
            await retriever.initialize()
            results = await retriever.search("completely unrelated query about galaxies")

    assert results == []
