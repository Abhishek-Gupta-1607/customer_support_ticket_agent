from src.models import TicketCreate
from src.tools.ticket_tool import TicketRepository, create_ticket_tool


def test_repository_prevents_duplicate_ticket_per_session() -> None:
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )

    first = repository.create("session-1", request)
    second = repository.create("session-1", request)

    assert first.ticket_id == second.ticket_id
    assert len(list(repository.all())) == 1


def test_unique_ids_across_different_sessions() -> None:
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )

    t1 = repository.create("session-1", request)
    t2 = repository.create("session-2", request)

    assert t1.ticket_id != t2.ticket_id
    assert len(list(repository.all())) == 2


def test_preservation_of_ticket_details() -> None:
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Rahul Sharma",
        customer_email="rahul@example.com",
        issue_description="Payment was charged twice on my card",
        category="payment",
        summary="Duplicate payment issue reported",
    )

    ticket = repository.create("session-1", request)
    assert ticket.customer_name == "Rahul Sharma"
    assert ticket.customer_email == "rahul@example.com"
    assert ticket.issue_description == "Payment was charged twice on my card"
    assert ticket.category == "payment"
    assert ticket.status == "open"
    assert ticket.created_at is not None


def test_retrieval_of_existing_ticket() -> None:
    repository = TicketRepository()
    request = TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Issue with order delivery",
        category="order",
        summary="Order delivery delayed",
    )

    created = repository.create("session-1", request)
    retrieved = repository.get(created.ticket_id)
    assert retrieved is not None
    assert retrieved.ticket_id == created.ticket_id


def test_unknown_ticket_returns_none() -> None:
    repository = TicketRepository()
    assert repository.get("CST-NONEXISTENT") is None


def test_ticket_tool_execution_and_idempotency() -> None:
    repository = TicketRepository()
    tool = create_ticket_tool(repository, "session-1")

    id1 = tool.invoke({
        "customer_name": "Alice Smith",
        "customer_email": "alice@example.com",
        "issue_description": "Cannot sign in to my account",
        "category": "account",
        "summary": "Account sign in problem",
    })
    id2 = tool.invoke({
        "customer_name": "Alice Smith",
        "customer_email": "alice@example.com",
        "issue_description": "Cannot sign in to my account",
        "category": "account",
        "summary": "Account sign in problem",
    })

    assert id1 == id2
    assert id1.startswith("CST-")
    assert repository.get(id1) is not None