"""Direct verification of all eight required customer demonstration scenarios.

Scenarios:
1. Shipping question (How long does standard shipping take?)
2. Payment question (I have a pending payment. What does it mean?)
3. Unknown question (Do you offer free international installation?)
4. Ticket creation (My payment was charged twice. I want to report this.)
5. Multi-turn conversation (name -> email -> description -> category -> ticket)
6. Ticket retrieval (GET /tickets/{ticket_id})
7. Duplicate request (Retry completed ticket in same session)
8. Failure handling (Simulate unavailable LLM or backend)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.llm.workflow import build_support_workflow
from src.models import TicketCreate
from src.pipeline import SupportPipeline
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(tmp_path / "scenario_vdb"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="scenario-support",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


# ----------------------------------------------------------------------
# Scenario 1: Shipping question
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_1_shipping_question() -> None:
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = [
        {
            "content": (
                "Standard delivery usually takes three to five business days. "
                "Express delivery usually takes one to two business days."
            ),
            "source": "shipping.md",
        }
    ]

    mock_model = AsyncMock()
    # 1. intent: answer
    intent_resp = MagicMock()
    intent_resp.content = '{"intent": "answer"}'
    # 2. answer
    ans_resp = MagicMock()
    ans_resp.content = "Standard delivery usually takes three to five business days."
    mock_model.ainvoke.side_effect = [intent_resp, ans_resp]

    store = SessionStore()
    repo = TicketRepository()
    wf = build_support_workflow(mock_model, mock_retriever, store, repo)

    result = await wf.ainvoke({
        "session_id": "scen-1",
        "customer_message": "How long does standard shipping take?",
    })

    assert "three to five business days" in result["response_text"].lower() or "3 to 5" in result["response_text"]
    assert "shipping.md" in result["sources"]
    assert result["ticket_id"] is None


# ----------------------------------------------------------------------
# Scenario 2: Payment question
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_2_payment_question() -> None:
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = [
        {
            "content": (
                "A pending authorization may temporarily appear separately from "
                "the completed charge."
            ),
            "source": "payments.md",
        }
    ]

    mock_model = AsyncMock()
    intent_resp = MagicMock()
    intent_resp.content = '{"intent": "answer"}'
    ans_resp = MagicMock()
    ans_resp.content = (
        "A pending payment is an authorization hold that may appear separately "
        "from the completed charge."
    )
    mock_model.ainvoke.side_effect = [intent_resp, ans_resp]

    store = SessionStore()
    repo = TicketRepository()
    wf = build_support_workflow(mock_model, mock_retriever, store, repo)

    result = await wf.ainvoke({
        "session_id": "scen-2",
        "customer_message": "I have a pending payment. What does it mean?",
    })

    assert "pending" in result["response_text"].lower()
    assert "payments.md" in result["sources"]
    assert result["ticket_id"] is None


# ----------------------------------------------------------------------
# Scenario 3: Unknown question
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_3_unknown_question() -> None:
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = []  # No relevant context

    mock_model = AsyncMock()
    intent_resp = MagicMock()
    intent_resp.content = '{"intent": "answer"}'
    ans_resp = MagicMock()
    ans_resp.content = (
        "I'm sorry, our knowledge base does not contain information about "
        "free international installation."
    )
    mock_model.ainvoke.side_effect = [intent_resp, ans_resp]

    store = SessionStore()
    repo = TicketRepository()
    wf = build_support_workflow(mock_model, mock_retriever, store, repo)

    result = await wf.ainvoke({
        "session_id": "scen-3",
        "customer_message": "Do you offer free international installation?",
    })

    assert "does not contain" in result["response_text"].lower() or "not" in result["response_text"].lower()
    assert result["sources"] == []
    assert result["ticket_id"] is None


# ----------------------------------------------------------------------
# Scenario 4: Ticket creation intent detection & initial collection
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_4_ticket_creation_intent() -> None:
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = [
        {"content": "A confirmed duplicate charge should be reported.", "source": "payments.md"}
    ]

    mock_model = AsyncMock()
    intent_resp = MagicMock()
    intent_resp.content = '{"intent": "ticket"}'
    extract_resp = MagicMock()
    extract_resp.content = '{"issue_description": "My payment was charged twice", "category": "payment"}'
    mock_model.ainvoke.side_effect = [intent_resp, extract_resp]

    store = SessionStore()
    repo = TicketRepository()
    wf = build_support_workflow(mock_model, mock_retriever, store, repo)

    result = await wf.ainvoke({
        "session_id": "scen-4",
        "customer_message": "My payment was charged twice. I want to report this.",
    })

    # The agent recognizes the issue and asks for missing info (e.g. name)
    assert result["ticket_id"] is None
    assert "name" in result["response_text"].lower()
    session = store.get_or_create("scen-4")
    assert session.category == "payment"
    assert "charged twice" in session.issue_description.lower()


# ----------------------------------------------------------------------
# Scenario 5: Multi-turn conversation leading to ticket creation
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_5_multi_turn_ticket_creation() -> None:
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = []

    store = SessionStore()
    repo = TicketRepository()

    mock_model = AsyncMock()
    wf = build_support_workflow(mock_model, mock_retriever, store, repo)

    # Turn 1: Customer reports issue
    t1_intent = MagicMock(content='{"intent": "ticket"}')
    t1_extract = MagicMock(content='{"issue_description": "I want to report a payment issue", "category": "payment"}')
    mock_model.ainvoke.side_effect = [t1_intent, t1_extract]

    res1 = await wf.ainvoke({
        "session_id": "scen-5",
        "customer_message": "I want to report a payment issue.",
    })
    assert res1["ticket_id"] is None
    assert "name" in res1["response_text"].lower()

    # Turn 2: Customer provides name
    t2_extract = MagicMock(content='{"customer_name": "Rahul Sharma"}')
    mock_model.ainvoke.side_effect = [t2_extract]

    res2 = await wf.ainvoke({
        "session_id": "scen-5",
        "customer_message": "Rahul Sharma",
    })
    assert res2["ticket_id"] is None
    assert "email" in res2["response_text"].lower()

    # Turn 3: Customer provides email -> All 4 fields collected -> Ticket created!
    t3_extract = MagicMock(content='{"customer_email": "rahul@example.com"}')
    mock_model.ainvoke.side_effect = [t3_extract]

    res3 = await wf.ainvoke({
        "session_id": "scen-5",
        "customer_message": "rahul@example.com",
    })
    assert res3["ticket_id"] is not None
    assert res3["ticket_id"].startswith("CST-")
    assert "created successfully" in res3["response_text"].lower() or "ticket id" in res3["response_text"].lower()

    # Verify repository has it
    saved = repo.get(res3["ticket_id"])
    assert saved is not None
    assert saved.customer_name == "Rahul Sharma"
    assert saved.customer_email == "rahul@example.com"
    assert saved.category == "payment"


# ----------------------------------------------------------------------
# Scenario 6: Ticket retrieval via GET /tickets/{ticket_id}
# ----------------------------------------------------------------------
def test_scenario_6_ticket_retrieval() -> None:
    from src.api.server import app, pipeline

    pipeline.ready = True
    ticket_data = TicketCreate(
        customer_name="Rahul Sharma",
        customer_email="rahul@example.com",
        issue_description="Payment was charged twice on my credit card",
        category="payment",
        summary="Payment charged twice issue",
    )
    created = pipeline.tickets.create("scen-6", ticket_data)

    client = TestClient(app)
    resp = client.get(f"/tickets/{created.ticket_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticket_id"] == created.ticket_id
    assert data["customer_name"] == "Rahul Sharma"
    assert data["customer_email"] == "rahul@example.com"
    assert data["category"] == "payment"
    assert data["status"] == "open"


# ----------------------------------------------------------------------
# Scenario 7: Duplicate request in same session does not create new ticket
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_7_duplicate_ticket_prevention() -> None:
    store = SessionStore()
    repo = TicketRepository()

    # First ticket already created
    ticket_data = TicketCreate(
        customer_name="Rahul Sharma",
        customer_email="rahul@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Payment duplicate charge",
    )
    first_ticket = repo.create("scen-7", ticket_data)
    session = store.get_or_create("scen-7")
    session.ticket_id = first_ticket.ticket_id

    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = []
    mock_model = AsyncMock()
    ans_resp = MagicMock(content="Your ticket has already been created.")
    mock_model.ainvoke.return_value = ans_resp

    wf = build_support_workflow(mock_model, mock_retriever, store, repo)

    # Customer sends another ticket request in the same session
    res = await wf.ainvoke({
        "session_id": "scen-7",
        "customer_message": "My payment was charged twice. I want to report this.",
    })

    # No new ticket created, total tickets in repo remains 1
    assert len(list(repo.all())) == 1
    assert session.ticket_id == first_ticket.ticket_id


# ----------------------------------------------------------------------
# Scenario 8: Failure handling preserves conversation history
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_scenario_8_failure_handling(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = SupportPipeline(settings, tmp_path)
    pipeline.ready = True

    # Pre-populate session with earlier turn
    session = pipeline.sessions.get_or_create("scen-8")
    session.history.append({"role": "user", "content": "Hello"})
    session.history.append({"role": "assistant", "content": "How can I help you?"})

    # Simulate downstream model failure
    failing_wf = AsyncMock()
    failing_wf.ainvoke.side_effect = RuntimeError("OpenAI endpoint unreachable")
    pipeline.workflow = failing_wf

    with pytest.raises(AgentProcessingError):
        await pipeline.process("scen-8", "Can you help me with an issue?")

    # Prior chat history must remain intact
    assert len(session.history) >= 2
    assert session.history[0]["content"] == "Hello"
    assert session.history[1]["content"] == "How can I help you?"

@pytest.mark.asyncio
async def test_question_asked_during_ticket_collection() -> None:
    """Verify that asking a question mid-ticket-collection routes to answer, not stuck in ticket loop."""
    mock_retriever = AsyncMock()
    mock_retriever.search.return_value = [
        {"content": "A pending authorization may temporarily appear.", "source": "payments.md"}
    ]

    store = SessionStore()
    repo = TicketRepository()
    session = store.get_or_create("interrupted-session")
    session.issue_description = "My payment was charged twice. I want to report this."
    session.category = "payment"

    mock_model = AsyncMock()
    intent_resp = MagicMock(content='{"intent": "answer"}')
    ans_resp = MagicMock(content="A pending authorization is a temporary hold.")
    mock_model.ainvoke.side_effect = [intent_resp, ans_resp]

    wf = build_support_workflow(mock_model, mock_retriever, store, repo)

    result = await wf.ainvoke({
        "session_id": "interrupted-session",
        "customer_message": "I have a pending payment. What does it mean?",
    })

    assert result["ticket_id"] is None
    assert "pending authorization" in result["response_text"].lower()
    assert "payments.md" in result["sources"]