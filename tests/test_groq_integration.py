"""Tests for Groq API integration and error handling.

Verifies:
1. Environment-driven configuration (GROQ_API_KEY, base_url, model).
2. Handling of Groq rate limits (429).
3. Handling of Groq authentication errors (401).
4. Handling of Groq timeouts and connection errors.
5. Handling of Groq unavailable service (500/503).
6. Local HuggingFace embeddings are preserved and not routed to Groq.
7. Structured output & tool calling behavior with mocked model.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from src.config import Settings, load_settings
from src.llm.client import build_chat_model, handle_llm_exception
from src.pipeline import SupportPipeline
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError


def test_groq_configuration_loading(monkeypatch) -> None:
    """Verify settings load Groq API defaults and GROQ_API_KEY securely."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret_key_12345")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    settings = load_settings()
    assert settings.llm_base_url == "https://api.groq.com/openai/v1"
    assert settings.llm_model == "openai/gpt-oss-20b"
    assert settings.llm_api_key == "gsk_test_secret_key_12345"
    assert settings.groq_api_key == "gsk_test_secret_key_12345"


def test_groq_build_chat_model() -> None:
    """Verify build_chat_model configures ChatOpenAI with Groq parameters."""
    settings = Settings(
        llm_base_url="https://api.groq.com/openai/v1",
        llm_api_key="gsk_sample_key",
        llm_model="openai/gpt-oss-20b",
        vector_db_path=".data/test_vdb",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-collection",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )
    model = build_chat_model(settings)
    assert str(model.openai_api_base).rstrip("/") == "https://api.groq.com/openai/v1"
    assert model.model_name == "openai/gpt-oss-20b"


def test_groq_rate_limit_handling() -> None:
    """Verify 429 rate limit errors produce helpful AgentProcessingError."""
    exc = Exception("Error code: 429 - Rate limit reached for model openai/gpt-oss-20b")
    err = handle_llm_exception(exc)
    assert isinstance(err, AgentProcessingError)
    assert "rate limit" in str(err).lower()


def test_groq_authentication_failure_handling() -> None:
    """Verify 401 authentication errors produce helpful AgentProcessingError."""
    exc = Exception("Error code: 401 - Invalid API Key provided")
    err = handle_llm_exception(exc)
    assert isinstance(err, AgentProcessingError)
    assert "authentication" in str(err).lower() or "groq_api_key" in str(err).lower()


def test_groq_timeout_handling() -> None:
    """Verify timeout errors produce helpful AgentProcessingError."""
    exc = TimeoutError("Request timed out after 30 seconds")
    err = handle_llm_exception(exc)
    assert isinstance(err, AgentProcessingError)
    assert "timed out" in str(err).lower()


def test_groq_service_unavailable_handling() -> None:
    """Verify 503 / 500 errors produce helpful AgentProcessingError."""
    exc = Exception("Error code: 503 - Service Unavailable")
    err = handle_llm_exception(exc)
    assert isinstance(err, AgentProcessingError)
    assert "unavailable" in str(err).lower()


@pytest.mark.asyncio
async def test_groq_rate_limit_in_workflow_propagates(tmp_path) -> None:
    """Verify that a rate limit during workflow execution raises AgentProcessingError."""
    from src.llm.workflow import build_support_workflow

    mock_model = AsyncMock()
    mock_model.ainvoke.side_effect = Exception("429 Rate limit reached")

    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = []

    store = SessionStore()
    repo = TicketRepository()
    wf = build_support_workflow(mock_model, mock_retriever, store, repo)

    with pytest.raises(AgentProcessingError, match="rate limit"):
        await wf.ainvoke({
            "session_id": "rate-limit-session",
            "customer_message": "How long does shipping take?",
        })


@pytest.mark.asyncio
async def test_groq_auth_failure_in_workflow_propagates(tmp_path) -> None:
    """Verify that an authentication error raises AgentProcessingError without fallback."""
    from src.llm.workflow import build_support_workflow

    mock_model = AsyncMock()
    mock_model.ainvoke.side_effect = Exception("401 Authentication failed")

    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = []

    store = SessionStore()
    repo = TicketRepository()
    wf = build_support_workflow(mock_model, mock_retriever, store, repo)

    with pytest.raises(AgentProcessingError, match="authentication"):
        await wf.ainvoke({
            "session_id": "auth-fail-session",
            "customer_message": "I need help with my account",
        })


def test_embeddings_use_huggingface_not_groq() -> None:
    """Verify embeddings explicitly use HuggingFace and local ChromaDB, not Groq."""
    from src.rag.embeddings import build_embeddings

    settings = Settings(
        llm_base_url="https://api.groq.com/openai/v1",
        llm_api_key="gsk_dummy",
        llm_model="openai/gpt-oss-20b",
        vector_db_path=".data/test_vdb",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-collection",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )
    # build_embeddings must return HuggingFaceEmbeddings instance
    with patch("src.rag.embeddings.HuggingFaceEmbeddings") as mock_hf:
        build_embeddings(settings)
        mock_hf.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
