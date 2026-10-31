from __future__ import annotations

import logging
from pathlib import Path

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import build_support_workflow
from src.models import ChatResponse
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError

logger = logging.getLogger(__name__)


class SupportPipeline:
    """Top-level binding for model, RAG, workflow, sessions, and ticket tool."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.model = build_chat_model(settings)
        self.retriever = KnowledgeRetriever(settings, documents_dir)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.workflow = None
        self.ready = False

    async def initialize(self) -> None:
        """Initialize shared components once during FastAPI startup."""
        # The order is intentional: retrieval must be ready before a workflow
        # capable of accepting traffic is exposed. If either step fails, leave
        # ``ready`` false and let the FastAPI lifespan fail clearly.
        await self.retriever.initialize()
        self.workflow = build_support_workflow(
            self.model,
            retriever=self.retriever,
            session_store=self.sessions,
            ticket_repo=self.tickets,
        )
        self.ready = True
        logger.info("Support pipeline initialized successfully.")

    async def process(self, session_id: str, message: str) -> ChatResponse:
        """Process one customer turn through the LangGraph workflow.

        Validates inputs, invokes the workflow asynchronously, preserves
        conversation state, and returns a structured ``ChatResponse``.
        """
        # ── guard ────────────────────────────────────────────────────────
        if not self.ready or self.workflow is None:
            raise ComponentNotReadyError("Support pipeline is not ready")

        # ── input validation ─────────────────────────────────────────────
        session_id = (session_id or "").strip()
        message = (message or "").strip()

        if not session_id:
            raise AgentProcessingError("Session ID must not be blank.")
        if not message:
            raise AgentProcessingError("Message must not be blank.")

        # ── session ──────────────────────────────────────────────────────
        session = self.sessions.get_or_create(session_id)
        session.history.append({"role": "user", "content": message})

        # ── workflow invocation ──────────────────────────────────────────
        initial_state = {
            "session_id": session_id,
            "customer_message": message,
            "messages": [],
            "retrieved_chunks": [],
            "route": "",
            "extracted_fields": {},
            "response_text": "",
            "sources": [],
            "ticket_id": None,
        }

        try:
            result = await self.workflow.ainvoke(initial_state)
        except AgentProcessingError:
            # Re-raise agent errors without wrapping
            raise
        except Exception as exc:
            logger.exception("Workflow invocation failed")
            from src.llm.client import handle_llm_exception
            raise handle_llm_exception(exc) from exc

        # ── output extraction ────────────────────────────────────────────
        response_text = (result.get("response_text") or "").strip()
        sources: list[str] = result.get("sources", [])
        ticket_id: str | None = result.get("ticket_id")

        if not response_text:
            response_text = (
                "I'm sorry, I was unable to process your request. "
                "Please try again or rephrase your question."
            )

        # Ensure sources contain only safe filenames
        safe_sources = [
            Path(s).name for s in sources if s
        ]

        # Validate ticket_id if present
        if ticket_id and not self.tickets.get(ticket_id):
            ticket_id = None

        # ── persist ──────────────────────────────────────────────────────
        session.history.append({"role": "assistant", "content": response_text})

        return ChatResponse(
            success=True,
            session_id=session_id,
            response=response_text,
            sources=safe_sources,
            ticket_id=ticket_id,
        )
