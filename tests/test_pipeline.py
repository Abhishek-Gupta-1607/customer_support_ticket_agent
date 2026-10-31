from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Settings
from src.models import ChatResponse
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


def make_pipeline_settings(tmp_path: Path) -> Settings:
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


@pytest.mark.asyncio
async def test_uninitialized_pipeline_rejects_chat(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_pipeline_settings(tmp_path), tmp_path)
    with pytest.raises(ComponentNotReadyError):
        await pipeline.process("session-1", "hello")


@pytest.mark.asyncio
async def test_pipeline_rejects_blank_session_id(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_pipeline_settings(tmp_path), tmp_path)
    pipeline.ready = True
    pipeline.workflow = MagicMock()

    with pytest.raises(AgentProcessingError, match="Session ID"):
        await pipeline.process("   ", "hello")


@pytest.mark.asyncio
async def test_pipeline_rejects_blank_message(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_pipeline_settings(tmp_path), tmp_path)
    pipeline.ready = True
    pipeline.workflow = MagicMock()

    with pytest.raises(AgentProcessingError, match="Message"):
        await pipeline.process("session-1", "   ")


@pytest.mark.asyncio
async def test_pipeline_processes_successful_turn(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_pipeline_settings(tmp_path), tmp_path)
    pipeline.ready = True

    mock_wf = AsyncMock()
    mock_wf.ainvoke.return_value = {
        "response_text": "Standard delivery takes 3 to 5 business days.",
        "sources": ["shipping.md"],
        "ticket_id": None,
    }
    pipeline.workflow = mock_wf

    result = await pipeline.process("session-1", "How long does standard shipping take?")
    assert isinstance(result, ChatResponse)
    assert result.success is True
    assert result.session_id == "session-1"
    assert "shipping" in result.response.lower() or "3 to 5" in result.response
    assert result.sources == ["shipping.md"]
    assert result.ticket_id is None

    session = pipeline.sessions.get_or_create("session-1")
    assert len(session.history) == 2
    assert session.history[0]["role"] == "user"
    assert session.history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_pipeline_returns_valid_ticket_id(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_pipeline_settings(tmp_path), tmp_path)
    pipeline.ready = True

    from src.models import TicketCreate
    ticket = pipeline.tickets.create(
        "session-1",
        TicketCreate(
            customer_name="Alice",
            customer_email="alice@example.com",
            issue_description="Payment issue description",
            category="payment",
            summary="Payment problem summary",
        ),
    )

    mock_wf = AsyncMock()
    mock_wf.ainvoke.return_value = {
        "response_text": f"Your ticket {ticket.ticket_id} has been created.",
        "sources": [],
        "ticket_id": ticket.ticket_id,
    }
    pipeline.workflow = mock_wf

    result = await pipeline.process("session-1", "Alice")
    assert result.ticket_id == ticket.ticket_id


@pytest.mark.asyncio
async def test_pipeline_filters_unregistered_ticket_id(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_pipeline_settings(tmp_path), tmp_path)
    pipeline.ready = True

    mock_wf = AsyncMock()
    mock_wf.ainvoke.return_value = {
        "response_text": "Fake ticket created.",
        "sources": [],
        "ticket_id": "CST-FAKE-9999",
    }
    pipeline.workflow = mock_wf

    result = await pipeline.process("session-1", "Alice")
    assert result.ticket_id is None


@pytest.mark.asyncio
async def test_pipeline_handles_workflow_exception(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_pipeline_settings(tmp_path), tmp_path)
    pipeline.ready = True

    mock_wf = AsyncMock()
    mock_wf.ainvoke.side_effect = RuntimeError("Downstream failure")
    pipeline.workflow = mock_wf

    with pytest.raises(AgentProcessingError):
        await pipeline.process("session-1", "Hello")
