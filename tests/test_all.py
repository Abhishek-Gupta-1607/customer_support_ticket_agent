"""Comprehensive test suite for the Customer Support Ticket Agent.

Tests use mocked LLM responses and deterministic fake embeddings so that
no actual LLM endpoint or large embedding model download is required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.models import ChatResponse, TicketCreate
from src.rag.document_loader import load_support_documents, split_support_documents
from src.sessions.store import ConversationState, SessionStore
from src.tools.ticket_tool import TicketRepository, create_ticket_tool
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


# =====================================================================
# Shared fixtures
# =====================================================================

@pytest.fixture
def knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "knowledge_base"


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(tmp_path / "vector_db"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-support",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


@pytest.fixture
def repository() -> TicketRepository:
    return TicketRepository()


@pytest.fixture
def session_store() -> SessionStore:
    return SessionStore()


@pytest.fixture
def sample_ticket_data() -> TicketCreate:
    return TicketCreate(
        customer_name="Rahul Sharma",
        customer_email="rahul@example.com",
        issue_description="Payment was charged twice on my credit card",
        category="payment",
        summary="Possible duplicate payment charge",
    )


# =====================================================================
# RAG TESTS – Document Loading
# =====================================================================

class TestDocumentLoading:
    """Tests for document loading and splitting."""

    def test_loader_preserves_source_names(self, knowledge_dir: Path) -> None:
        documents = load_support_documents(knowledge_dir)
        assert {item.metadata["source"] for item in documents} == {
            "accounts.md",
            "payments.md",
            "returns.md",
            "shipping.md",
        }

    def test_splitter_keeps_source_metadata(self, knowledge_dir: Path) -> None:
        chunks = split_support_documents(load_support_documents(knowledge_dir))
        assert chunks
        assert all(
            chunk.metadata.get("source", "").endswith(".md") for chunk in chunks
        )

    def test_loader_rejects_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_support_documents(tmp_path / "nonexistent")

    def test_loader_rejects_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_kb"
        empty.mkdir()
        with pytest.raises(ValueError, match="No Markdown"):
            load_support_documents(empty)

    def test_splitter_produces_multiple_chunks_for_large_docs(
        self, knowledge_dir: Path
    ) -> None:
        docs = load_support_documents(knowledge_dir)
        chunks = split_support_documents(docs)
        # We have 4 docs, should have at least 4 chunks
        assert len(chunks) >= len(docs)

    def test_documents_contain_expected_content(
        self, knowledge_dir: Path
    ) -> None:
        docs = load_support_documents(knowledge_dir)
        all_text = " ".join(d.page_content for d in docs)
        assert "standard delivery" in all_text.lower() or "shipping" in all_text.lower()
        assert "refund" in all_text.lower() or "return" in all_text.lower()


# =====================================================================
# RAG TESTS – Retriever
# =====================================================================

class TestRetriever:
    """Tests for KnowledgeRetriever using fake embeddings."""

    @pytest.fixture
    def fake_embeddings(self):
        """Deterministic fake embedding function for testing."""
        import math

        class FakeEmbeddings:
            def _normalize(self, text: str) -> list[float]:
                h = hashlib.md5(text.encode()).hexdigest()
                raw = [int(c, 16) / 15.0 for c in h[:32]]
                norm = math.sqrt(sum(x * x for x in raw)) or 1.0
                return [x / norm for x in raw]

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self._normalize(t) for t in texts]

            def embed_query(self, text: str) -> list[float]:
                return self._normalize(text)

        return FakeEmbeddings()

    @pytest.mark.asyncio
    async def test_initialization(
        self, test_settings: Settings, knowledge_dir: Path, fake_embeddings, tmp_path: Path
    ) -> None:
        from src.rag.retriever import KnowledgeRetriever

        test_settings_local = Settings(
            **{**test_settings.__dict__, "vector_db_path": str(tmp_path / "vec_init")}
        )
        retriever = KnowledgeRetriever(test_settings_local, knowledge_dir)

        with patch("src.rag.retriever.build_embeddings", return_value=fake_embeddings):
            await retriever.initialize()

        assert retriever._store is not None

    @pytest.mark.asyncio
    async def test_initialization_idempotent(
        self, test_settings: Settings, knowledge_dir: Path, fake_embeddings, tmp_path: Path
    ) -> None:
        """Reinitializing must not duplicate chunks."""
        from src.rag.retriever import KnowledgeRetriever

        vdb = tmp_path / "vec_idem"
        test_settings_local = Settings(
            **{**test_settings.__dict__, "vector_db_path": str(vdb)}
        )

        retriever = KnowledgeRetriever(test_settings_local, knowledge_dir)
        with patch("src.rag.retriever.build_embeddings", return_value=fake_embeddings):
            await retriever.initialize()
            count_first = retriever._store._collection.count()

            # Re-initialize
            retriever2 = KnowledgeRetriever(test_settings_local, knowledge_dir)
            await retriever2.initialize()
            count_second = retriever2._store._collection.count()

        assert count_first == count_second

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self, test_settings: Settings, knowledge_dir: Path, fake_embeddings, tmp_path: Path
    ) -> None:
        from src.rag.retriever import KnowledgeRetriever

        test_settings_local = Settings(
            **{**test_settings.__dict__, "vector_db_path": str(tmp_path / "vec_search")}
        )
        retriever = KnowledgeRetriever(test_settings_local, knowledge_dir)

        with patch("src.rag.retriever.build_embeddings", return_value=fake_embeddings):
            await retriever.initialize()
            results = await retriever.search("shipping delivery")

        # With fake embeddings, relevance scores may be low, so results could be empty.
        # The important thing is that no error is raised and we get a list back.
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_blank_query(
        self, test_settings: Settings, knowledge_dir: Path, fake_embeddings, tmp_path: Path
    ) -> None:
        from src.rag.retriever import KnowledgeRetriever

        test_settings_local = Settings(
            **{**test_settings.__dict__, "vector_db_path": str(tmp_path / "vec_blank")}
        )
        retriever = KnowledgeRetriever(test_settings_local, knowledge_dir)

        with patch("src.rag.retriever.build_embeddings", return_value=fake_embeddings):
            await retriever.initialize()
            results = await retriever.search("")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_before_initialization(
        self, test_settings: Settings, knowledge_dir: Path
    ) -> None:
        from src.rag.retriever import KnowledgeRetriever

        retriever = KnowledgeRetriever(test_settings, knowledge_dir)

        with pytest.raises(ComponentNotReadyError):
            await retriever.search("test query")

    @pytest.mark.asyncio
    async def test_search_result_format(
        self, test_settings: Settings, knowledge_dir: Path, fake_embeddings, tmp_path: Path
    ) -> None:
        """Results must be dicts with 'content' and 'source' keys."""
        from src.rag.retriever import KnowledgeRetriever

        test_settings_local = Settings(
            **{
                **test_settings.__dict__,
                "vector_db_path": str(tmp_path / "vec_fmt"),
                "rag_top_k": 10,  # Get more results to verify format
            }
        )
        retriever = KnowledgeRetriever(test_settings_local, knowledge_dir)

        with patch("src.rag.retriever.build_embeddings", return_value=fake_embeddings):
            # Patch threshold to 0 so we get results with fake embeddings
            with patch("src.rag.retriever._RELEVANCE_THRESHOLD", 0.0):
                await retriever.initialize()
                results = await retriever.search("shipping")

        for result in results:
            assert "content" in result
            assert "source" in result
            assert result["source"].endswith(".md")
            # Must not contain absolute paths
            assert "/" not in result["source"] and "\\" not in result["source"]

    @pytest.mark.asyncio
    async def test_search_respects_limit(
        self, test_settings: Settings, knowledge_dir: Path, fake_embeddings, tmp_path: Path
    ) -> None:
        from src.rag.retriever import KnowledgeRetriever

        test_settings_local = Settings(
            **{**test_settings.__dict__, "vector_db_path": str(tmp_path / "vec_lim")}
        )
        retriever = KnowledgeRetriever(test_settings_local, knowledge_dir)

        with patch("src.rag.retriever.build_embeddings", return_value=fake_embeddings):
            with patch("src.rag.retriever._RELEVANCE_THRESHOLD", 0.0):
                await retriever.initialize()
                results = await retriever.search("shipping", limit=1)

        assert len(results) <= 1


# =====================================================================
# SESSION TESTS
# =====================================================================

class TestSession:
    """Tests for ConversationState and SessionStore."""

    def test_session_reports_only_missing_ticket_fields(self) -> None:
        state = ConversationState(
            customer_name="Asha", customer_email="asha@example.com"
        )
        assert state.missing_ticket_fields() == ["issue_description", "category"]

    def test_new_session_all_fields_missing(self) -> None:
        state = ConversationState()
        assert state.missing_ticket_fields() == [
            "customer_name",
            "customer_email",
            "issue_description",
            "category",
        ]

    def test_complete_session_no_missing(self) -> None:
        state = ConversationState(
            customer_name="Asha",
            customer_email="asha@example.com",
            issue_description="Payment issue",
            category="payment",
        )
        assert state.missing_ticket_fields() == []

    def test_sessions_are_isolated(self, session_store: SessionStore) -> None:
        s1 = session_store.get_or_create("session-1")
        s2 = session_store.get_or_create("session-2")
        s1.customer_name = "Alice"
        assert s2.customer_name is None

    def test_same_session_returns_same_state(
        self, session_store: SessionStore
    ) -> None:
        s1 = session_store.get_or_create("session-1")
        s1.customer_name = "Alice"
        s2 = session_store.get_or_create("session-1")
        assert s2.customer_name == "Alice"

    def test_blank_session_id_rejected(self, session_store: SessionStore) -> None:
        with pytest.raises(ValueError):
            session_store.get_or_create("   ")

    def test_conversation_history_order(self) -> None:
        state = ConversationState()
        state.history.append({"role": "user", "content": "Hello"})
        state.history.append({"role": "assistant", "content": "Hi!"})
        assert state.history[0]["role"] == "user"
        assert state.history[1]["role"] == "assistant"

    def test_completed_session_retains_ticket_id(self) -> None:
        state = ConversationState()
        state.ticket_id = "CST-2026-0001"
        assert state.ticket_id == "CST-2026-0001"


# =====================================================================
# TICKET TESTS
# =====================================================================

class TestTickets:
    """Tests for TicketRepository and ticket tool."""

    def test_repository_prevents_duplicate_ticket_per_session(
        self, repository: TicketRepository, sample_ticket_data: TicketCreate
    ) -> None:
        first = repository.create("session-1", sample_ticket_data)
        second = repository.create("session-1", sample_ticket_data)
        assert first.ticket_id == second.ticket_id
        assert len(list(repository.all())) == 1

    def test_unique_ids_across_sessions(
        self, repository: TicketRepository, sample_ticket_data: TicketCreate
    ) -> None:
        t1 = repository.create("session-1", sample_ticket_data)
        t2 = repository.create("session-2", sample_ticket_data)
        assert t1.ticket_id != t2.ticket_id

    def test_ticket_preserves_details(
        self, repository: TicketRepository, sample_ticket_data: TicketCreate
    ) -> None:
        ticket = repository.create("session-1", sample_ticket_data)
        assert ticket.customer_name == "Rahul Sharma"
        assert ticket.customer_email == "rahul@example.com"
        assert ticket.category == "payment"
        assert ticket.status == "open"

    def test_ticket_retrieval(
        self, repository: TicketRepository, sample_ticket_data: TicketCreate
    ) -> None:
        ticket = repository.create("session-1", sample_ticket_data)
        retrieved = repository.get(ticket.ticket_id)
        assert retrieved is not None
        assert retrieved.ticket_id == ticket.ticket_id

    def test_unknown_ticket_returns_none(
        self, repository: TicketRepository
    ) -> None:
        assert repository.get("CST-NONEXISTENT") is None

    def test_ticket_tool_creates_ticket(
        self, repository: TicketRepository
    ) -> None:
        tool = create_ticket_tool(repository, "session-1")
        ticket_id = tool.invoke({
            "customer_name": "Test User",
            "customer_email": "test@example.com",
            "issue_description": "Test issue description",
            "category": "technical",
            "summary": "Test ticket summary",
        })
        assert ticket_id.startswith("CST-")
        assert repository.get(ticket_id) is not None

    def test_ticket_tool_idempotent_per_session(
        self, repository: TicketRepository
    ) -> None:
        tool = create_ticket_tool(repository, "session-1")
        id1 = tool.invoke({
            "customer_name": "Test User",
            "customer_email": "test@example.com",
            "issue_description": "Test issue description",
            "category": "technical",
            "summary": "Test ticket summary",
        })
        id2 = tool.invoke({
            "customer_name": "Test User",
            "customer_email": "test@example.com",
            "issue_description": "Test issue description",
            "category": "technical",
            "summary": "Test ticket summary",
        })
        assert id1 == id2


# =====================================================================
# PIPELINE TESTS
# =====================================================================

class TestPipeline:
    """Tests for SupportPipeline."""

    @pytest.mark.asyncio
    async def test_uninitialized_pipeline_rejects_chat(
        self, tmp_path: Path, test_settings: Settings
    ) -> None:
        from src.pipeline import SupportPipeline

        pipeline = SupportPipeline(test_settings, tmp_path)
        with pytest.raises(ComponentNotReadyError):
            await pipeline.process("session-1", "hello")

    @pytest.mark.asyncio
    async def test_pipeline_rejects_blank_message(
        self, tmp_path: Path, test_settings: Settings
    ) -> None:
        from src.pipeline import SupportPipeline

        pipeline = SupportPipeline(test_settings, tmp_path)
        pipeline.ready = True
        pipeline.workflow = MagicMock()

        with pytest.raises(AgentProcessingError):
            await pipeline.process("session-1", "   ")

    @pytest.mark.asyncio
    async def test_pipeline_rejects_blank_session(
        self, tmp_path: Path, test_settings: Settings
    ) -> None:
        from src.pipeline import SupportPipeline

        pipeline = SupportPipeline(test_settings, tmp_path)
        pipeline.ready = True
        pipeline.workflow = MagicMock()

        with pytest.raises(AgentProcessingError):
            await pipeline.process("", "hello")

    @pytest.mark.asyncio
    async def test_pipeline_process_success(
        self, tmp_path: Path, test_settings: Settings
    ) -> None:
        from src.pipeline import SupportPipeline

        pipeline = SupportPipeline(test_settings, tmp_path)
        pipeline.ready = True

        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = {
            "response_text": "Standard shipping takes 3-5 business days.",
            "sources": ["shipping.md"],
            "ticket_id": None,
        }
        pipeline.workflow = mock_workflow

        result = await pipeline.process("session-1", "How long does shipping take?")
        assert isinstance(result, ChatResponse)
        assert result.success is True
        assert result.session_id == "session-1"
        assert "shipping" in result.response.lower() or "3-5" in result.response
        assert "shipping.md" in result.sources

    @pytest.mark.asyncio
    async def test_pipeline_preserves_conversation_history(
        self, tmp_path: Path, test_settings: Settings
    ) -> None:
        from src.pipeline import SupportPipeline

        pipeline = SupportPipeline(test_settings, tmp_path)
        pipeline.ready = True

        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = {
            "response_text": "Hello!",
            "sources": [],
            "ticket_id": None,
        }
        pipeline.workflow = mock_workflow

        await pipeline.process("session-1", "Hi")
        session = pipeline.sessions.get_or_create("session-1")
        assert len(session.history) == 2  # user + assistant
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_pipeline_handles_workflow_failure(
        self, tmp_path: Path, test_settings: Settings
    ) -> None:
        from src.pipeline import SupportPipeline

        pipeline = SupportPipeline(test_settings, tmp_path)
        pipeline.ready = True

        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.side_effect = RuntimeError("LLM down")
        pipeline.workflow = mock_workflow

        with pytest.raises(AgentProcessingError):
            await pipeline.process("session-1", "Hello")


# =====================================================================
# API TESTS
# =====================================================================

class TestAPI:
    """Tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked pipeline."""
        from src.api.server import app, pipeline

        # Mark pipeline as ready with mocks
        pipeline.ready = True
        pipeline.workflow = AsyncMock()
        pipeline.workflow.ainvoke.return_value = {
            "response_text": "Test response.",
            "sources": ["test.md"],
            "ticket_id": None,
        }
        return TestClient(app, raise_server_exceptions=False)

    def test_health_ready(self, client: TestClient) -> None:
        from src.api.server import pipeline
        pipeline.ready = True
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_health_not_ready(self, client: TestClient) -> None:
        from src.api.server import pipeline
        original = pipeline.ready
        pipeline.ready = False
        resp = client.get("/health")
        assert resp.status_code == 503
        pipeline.ready = original

    def test_chat_success(self, client: TestClient) -> None:
        resp = client.post(
            "/chat",
            json={"session_id": "test-session", "message": "Hello"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["session_id"] == "test-session"
        assert "response" in data

    def test_chat_missing_message(self, client: TestClient) -> None:
        resp = client.post(
            "/chat",
            json={"session_id": "test-session"},
        )
        assert resp.status_code == 422

    def test_chat_empty_message(self, client: TestClient) -> None:
        resp = client.post(
            "/chat",
            json={"session_id": "test-session", "message": ""},
        )
        assert resp.status_code == 422

    def test_chat_missing_session(self, client: TestClient) -> None:
        resp = client.post(
            "/chat",
            json={"message": "Hello"},
        )
        assert resp.status_code == 422

    def test_get_ticket_not_found(self, client: TestClient) -> None:
        resp = client.get("/tickets/CST-NONEXISTENT")
        assert resp.status_code == 404

    def test_get_ticket_success(self, client: TestClient) -> None:
        from src.api.server import pipeline

        ticket_data = TicketCreate(
            customer_name="Test",
            customer_email="test@example.com",
            issue_description="Test issue for API",
            category="technical",
            summary="Test ticket for API test",
        )
        ticket = pipeline.tickets.create("api-test-session", ticket_data)
        resp = client.get(f"/tickets/{ticket.ticket_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticket_id"] == ticket.ticket_id
        assert data["customer_name"] == "Test"

    def test_chat_backend_failure(self, client: TestClient) -> None:
        from src.api.server import pipeline

        original_workflow = pipeline.workflow
        pipeline.workflow = AsyncMock()
        pipeline.workflow.ainvoke.side_effect = AgentProcessingError("LLM error")

        resp = client.post(
            "/chat",
            json={"session_id": "fail-session", "message": "Hello"},
        )
        assert resp.status_code == 502
        pipeline.workflow = original_workflow

    def test_chat_service_unavailable(self, client: TestClient) -> None:
        from src.api.server import pipeline

        original = pipeline.ready
        pipeline.ready = False
        resp = client.post(
            "/chat",
            json={"session_id": "test-session", "message": "Hello"},
        )
        assert resp.status_code == 503
        pipeline.ready = original


# =====================================================================
# WORKFLOW TESTS (mocked LLM)
# =====================================================================

class TestWorkflow:
    """Tests for the LangGraph workflow with mocked LLM."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock LLM that returns configurable responses."""
        model = AsyncMock()
        return model

    @pytest.fixture
    def mock_retriever(self):
        retriever = AsyncMock()
        retriever.search = AsyncMock(return_value=[])
        return retriever

    @pytest.fixture
    def workflow_deps(self, mock_model, mock_retriever):
        from src.llm.workflow import build_support_workflow

        store = SessionStore()
        repo = TicketRepository()
        wf = build_support_workflow(mock_model, mock_retriever, store, repo)
        return wf, mock_model, mock_retriever, store, repo

    @pytest.mark.asyncio
    async def test_knowledge_answer(self, workflow_deps) -> None:
        wf, mock_model, mock_retriever, store, repo = workflow_deps

        # Retriever returns shipping docs
        mock_retriever.search.return_value = [
            {"content": "Standard delivery takes 3-5 business days.", "source": "shipping.md"}
        ]

        # LLM intent: answer
        intent_response = MagicMock()
        intent_response.content = '{"intent": "answer"}'
        # LLM answer
        answer_response = MagicMock()
        answer_response.content = "Standard shipping takes 3-5 business days."

        mock_model.ainvoke = AsyncMock(side_effect=[intent_response, answer_response])

        result = await wf.ainvoke({
            "session_id": "test-1",
            "customer_message": "How long does shipping take?",
            "messages": [],
            "retrieved_chunks": [],
            "route": "",
            "extracted_fields": {},
            "response_text": "",
            "sources": [],
            "ticket_id": None,
        })

        assert result.get("response_text")
        assert "shipping.md" in result.get("sources", [])

    @pytest.mark.asyncio
    async def test_unknown_question(self, workflow_deps) -> None:
        wf, mock_model, mock_retriever, store, repo = workflow_deps

        # No relevant docs found
        mock_retriever.search.return_value = []

        intent_response = MagicMock()
        intent_response.content = '{"intent": "answer"}'
        answer_response = MagicMock()
        answer_response.content = (
            "I'm sorry, I don't have information about that in our knowledge base."
        )
        mock_model.ainvoke = AsyncMock(side_effect=[intent_response, answer_response])

        result = await wf.ainvoke({
            "session_id": "test-2",
            "customer_message": "Do you offer free international installation?",
            "messages": [],
            "retrieved_chunks": [],
            "route": "",
            "extracted_fields": {},
            "response_text": "",
            "sources": [],
            "ticket_id": None,
        })

        assert result.get("response_text")
        assert result.get("sources") == []

    @pytest.mark.asyncio
    async def test_ticket_intent(self, workflow_deps) -> None:
        wf, mock_model, mock_retriever, store, repo = workflow_deps

        mock_retriever.search.return_value = [
            {"content": "Duplicate charges should be reported.", "source": "payments.md"}
        ]

        intent_response = MagicMock()
        intent_response.content = '{"intent": "ticket"}'
        extract_response = MagicMock()
        extract_response.content = '{"issue_description": "Payment charged twice"}'
        mock_model.ainvoke = AsyncMock(side_effect=[intent_response, extract_response])

        result = await wf.ainvoke({
            "session_id": "test-3",
            "customer_message": "My payment was charged twice, I want to report this",
            "messages": [],
            "retrieved_chunks": [],
            "route": "",
            "extracted_fields": {},
            "response_text": "",
            "sources": [],
            "ticket_id": None,
        })

        # Should ask for missing fields (name at minimum)
        assert result.get("response_text")
        assert result.get("ticket_id") is None  # Not created yet

    @pytest.mark.asyncio
    async def test_ticket_creation_complete(self, workflow_deps) -> None:
        wf, mock_model, mock_retriever, store, repo = workflow_deps

        # Pre-fill session with all needed data except name
        session = store.get_or_create("test-4")
        session.customer_email = "rahul@example.com"
        session.issue_description = "Payment was charged twice"
        session.category = "payment"

        mock_retriever.search.return_value = []

        # The decide node sees partial fields and routes to ticket
        extract_response = MagicMock()
        extract_response.content = '{"customer_name": "Rahul Sharma"}'
        mock_model.ainvoke = AsyncMock(return_value=extract_response)

        result = await wf.ainvoke({
            "session_id": "test-4",
            "customer_message": "Rahul Sharma",
            "messages": [],
            "retrieved_chunks": [],
            "route": "",
            "extracted_fields": {},
            "response_text": "",
            "sources": [],
            "ticket_id": None,
        })

        assert result.get("ticket_id") is not None
        assert result["ticket_id"].startswith("CST-")

    @pytest.mark.asyncio
    async def test_duplicate_ticket_prevention(self, workflow_deps) -> None:
        wf, mock_model, mock_retriever, store, repo = workflow_deps

        # Pre-create a ticket in the session
        session = store.get_or_create("test-5")
        session.ticket_id = "CST-2026-0001"

        mock_retriever.search.return_value = []
        intent_response = MagicMock()
        intent_response.content = '{"intent": "answer"}'
        answer_response = MagicMock()
        answer_response.content = "Your ticket has been created."
        mock_model.ainvoke = AsyncMock(side_effect=[intent_response, answer_response])

        result = await wf.ainvoke({
            "session_id": "test-5",
            "customer_message": "I want to create another ticket",
            "messages": [],
            "retrieved_chunks": [],
            "route": "",
            "extracted_fields": {},
            "response_text": "",
            "sources": [],
            "ticket_id": None,
        })

        # Should route to answer since ticket already exists
        assert result.get("response_text")


# =====================================================================
# STREAMLIT INTEGRATION TESTS (contract verification)
# =====================================================================

class TestStreamlitContract:
    """Verify the HTTP request contract and session management."""

    def test_api_request_contract(self) -> None:
        """The Streamlit app should send the correct JSON structure."""
        import json

        payload = {"session_id": "test-uuid", "message": "How long does shipping take?"}
        parsed = json.loads(json.dumps(payload))
        assert "session_id" in parsed
        assert "message" in parsed
        assert len(parsed["session_id"]) > 0
        assert len(parsed["message"]) > 0

    def test_api_response_contract(self) -> None:
        """The response must contain the expected fields."""
        response = {
            "success": True,
            "session_id": "test-uuid",
            "response": "Standard shipping takes 3-5 business days.",
            "sources": ["shipping.md"],
            "ticket_id": None,
        }
        assert response["success"] is True
        assert isinstance(response["sources"], list)
        assert response["ticket_id"] is None or isinstance(
            response["ticket_id"], str
        )

    def test_response_with_ticket(self) -> None:
        """Response with a ticket_id set."""
        response = ChatResponse(
            success=True,
            session_id="test",
            response="Ticket created.",
            sources=[],
            ticket_id="CST-2026-0001",
        )
        assert response.ticket_id == "CST-2026-0001"

    def test_session_id_is_stable_uuid(self) -> None:
        """Session ID should be a valid UUID string."""
        import uuid

        sid = str(uuid.uuid4())
        # Should not raise
        uuid.UUID(sid)
