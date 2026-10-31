import pytest

from src.sessions.store import ConversationState, SessionStore


def test_session_reports_only_missing_ticket_fields() -> None:
    state = ConversationState(customer_name="Asha", customer_email="asha@example.com")
    assert state.missing_ticket_fields() == ["issue_description", "category"]


def test_empty_session_reports_all_missing_fields() -> None:
    state = ConversationState()
    assert state.missing_ticket_fields() == [
        "customer_name",
        "customer_email",
        "issue_description",
        "category",
    ]


def test_complete_session_reports_no_missing_fields() -> None:
    state = ConversationState(
        customer_name="Asha",
        customer_email="asha@example.com",
        issue_description="Payment charged twice",
        category="payment",
    )
    assert state.missing_ticket_fields() == []


def test_sessions_are_isolated_by_session_id() -> None:
    store = SessionStore()
    s1 = store.get_or_create("session-1")
    s2 = store.get_or_create("session-2")

    s1.customer_name = "Alice"
    s1.customer_email = "alice@example.com"

    assert s2.customer_name is None
    assert s2.customer_email is None


def test_same_session_id_returns_same_instance() -> None:
    store = SessionStore()
    s1 = store.get_or_create("session-1")
    s1.customer_name = "Bob"
    s2 = store.get_or_create("session-1")
    assert s2.customer_name == "Bob"


def test_blank_session_id_raises_value_error() -> None:
    store = SessionStore()
    with pytest.raises(ValueError, match="session_id must not be blank"):
        store.get_or_create("   ")


def test_conversation_history_preserves_order() -> None:
    state = ConversationState()
    state.history.append({"role": "user", "content": "How long is shipping?"})
    state.history.append({"role": "assistant", "content": "3-5 business days."})
    state.history.append({"role": "user", "content": "Thank you."})

    assert len(state.history) == 3
    assert state.history[0] == {"role": "user", "content": "How long is shipping?"}
    assert state.history[1] == {"role": "assistant", "content": "3-5 business days."}
    assert state.history[2] == {"role": "user", "content": "Thank you."}


def test_completed_session_retains_ticket_id() -> None:
    state = ConversationState()
    assert state.ticket_id is None
    state.ticket_id = "CST-2026-0001"
    assert state.ticket_id == "CST-2026-0001"
