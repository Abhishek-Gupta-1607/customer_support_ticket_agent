from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from src.llm.prompts import ANSWER_TEMPLATE, SYSTEM_PROMPT
from src.models import TicketCategory, TicketCreate
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import ConversationState, SessionStore
from src.tools.ticket_tool import TicketRepository, create_ticket_tool
from src.utils.errors import AgentProcessingError

logger = logging.getLogger(__name__)

# Valid ticket categories for quick validation
_VALID_CATEGORIES = {"order", "payment", "account", "technical", "other"}

# Human-readable field names for follow-up prompts
_FIELD_PROMPTS: dict[str, str] = {
    "customer_name": "Could you please provide your full name?",
    "customer_email": "What is your email address so we can follow up on this issue?",
    "issue_description": "Could you describe the issue you're experiencing in more detail?",
    "category": (
        "What category does your issue fall under? "
        "Options: order, payment, account, technical, or other."
    ),
}

# Intent-detection system prompt
_INTENT_SYSTEM = """You are analyzing a customer support message to determine intent.

RULES:
- If the customer is asking a question about policies, shipping, payments, returns, accounts, or general information, respond with: {"intent": "answer"}
- If the customer is reporting a problem, complaint, requesting help with an unresolved issue, wants to create/file a support ticket, or wants to escalate something, respond with: {"intent": "ticket"}
- If the customer is providing personal information (name, email) as part of a ticket creation flow, respond with: {"intent": "ticket"}

Respond with ONLY the JSON object, nothing else."""

# Field-extraction system prompt
_EXTRACT_SYSTEM = """Extract customer information from the message below.
Return a JSON object with ONLY the fields that are EXPLICITLY stated by the customer.
Possible fields: "customer_name", "customer_email", "issue_description", "category".

For category, only use: "order", "payment", "account", "technical", or "other".
For email, only extract a valid email address.
For customer_name, only extract if the customer explicitly states their name.
For issue_description, extract if the customer describes their problem.

If a field is NOT explicitly mentioned, do NOT include it.
Respond with ONLY the JSON object, nothing else. Example: {"customer_name": "John Doe"}"""


class SupportWorkflowState(TypedDict, total=False):
    """Typed state shared by the supplied LangGraph node skeletons."""

    session_id: str
    customer_message: str
    messages: Annotated[list, add_messages]
    retrieved_chunks: list[dict[str, str]]
    route: str
    extracted_fields: dict[str, str]
    response_text: str
    sources: list[str]
    ticket_id: str | None


def build_support_workflow(
    model: BaseChatModel,
    retriever: KnowledgeRetriever | None = None,
    session_store: SessionStore | None = None,
    ticket_repo: TicketRepository | None = None,
):
    """Build the agent graph with injected runtime dependencies.

    Parameters
    ----------
    model:
        LangChain chat model adapter (e.g. ChatOpenAI).
    retriever:
        Initialized ``KnowledgeRetriever`` for RAG lookups.
    session_store:
        ``SessionStore`` for conversation state across turns.
    ticket_repo:
        ``TicketRepository`` for creating support tickets.
    """

    # ----- helpers --------------------------------------------------------

    def _parse_json(text: str) -> dict[str, Any]:
        """Best-effort JSON extraction from LLM output."""
        text = text.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to find JSON in markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try to find first { ... } block
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}

    def _build_history_context(session: ConversationState) -> str:
        """Build a short conversation-history string for LLM context."""
        if not session.history:
            return "No prior conversation."
        recent = session.history[-10:]  # last 10 messages
        lines = []
        for msg in recent:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _session_summary(session: ConversationState) -> str:
        """One-line summary of collected ticket fields."""
        parts = []
        if session.customer_name:
            parts.append(f"name={session.customer_name}")
        if session.customer_email:
            parts.append(f"email={session.customer_email}")
        if session.issue_description:
            parts.append(f"issue={session.issue_description[:60]}")
        if session.category:
            parts.append(f"category={session.category}")
        if session.ticket_id:
            parts.append(f"ticket_id={session.ticket_id}")
        return ", ".join(parts) if parts else "No fields collected yet."

    # ----- nodes ----------------------------------------------------------

    async def retrieve(state: SupportWorkflowState) -> SupportWorkflowState:
        """Retrieve relevant knowledge-base chunks for the customer query."""
        query = state.get("customer_message", "")
        if not query.strip() or retriever is None:
            return {"retrieved_chunks": []}

        try:
            chunks = await retriever.search(query)
        except Exception:
            logger.exception("Retrieval failed")
            chunks = []

        # Deduplicate by content
        seen: set[str] = set()
        unique: list[dict[str, str]] = []
        for chunk in chunks:
            c = chunk["content"]
            if c not in seen:
                seen.add(c)
                unique.append(chunk)

        return {"retrieved_chunks": unique}

    async def decide(state: SupportWorkflowState) -> SupportWorkflowState:
        """Classify customer intent and extract any provided fields."""
        message = state.get("customer_message", "")
        session_id = state.get("session_id", "")
        session = session_store.get_or_create(session_id) if session_store else ConversationState()

        # If a ticket was already created in this session, route to answer
        if session.ticket_id:
            return {"route": "answer", "extracted_fields": {}}

        # Check if we are mid-collection of a ticket
        has_partial = any([
            session.customer_name,
            session.customer_email,
            session.issue_description,
            session.category,
        ])

        # If mid-collection and customer is providing an answer (not asking a question),
        # continue collecting fields directly
        if has_partial and session.missing_ticket_fields() and not _is_clear_question(message):
            return {"route": "ticket", "extracted_fields": {}}

        # Otherwise, use LLM for intent classification
        history_ctx = _build_history_context(session)
        intent_prompt = (
            f"Conversation so far:\n{history_ctx}\n\n"
            f"Latest customer message: {message}"
        )

        try:
            response = await model.ainvoke([
                SystemMessage(content=_INTENT_SYSTEM),
                HumanMessage(content=intent_prompt),
            ])
            parsed = _parse_json(response.content if hasattr(response, 'content') else str(response))
            intent = parsed.get("intent", "answer")
        except Exception as exc:
            from src.llm.client import handle_llm_exception
            err = handle_llm_exception(exc)
            if any(term in str(err).lower() for term in ["rate limit", "authentication", "timed out", "unavailable", "connect"]):
                raise err
            logger.warning("Intent classification failed, falling back to heuristics: %s", exc)
            if has_partial and not _is_clear_question(message):
                intent = "ticket"
            else:
                intent = "answer"

        route = "ticket" if intent == "ticket" else "answer"
        logger.debug("Intent classified as '%s' for message: %s", route, message[:60])
        return {"route": route, "extracted_fields": {}}

    async def answer(state: SupportWorkflowState) -> SupportWorkflowState:
        """Generate a grounded answer using retrieved knowledge-base chunks."""
        message = state.get("customer_message", "")
        chunks = state.get("retrieved_chunks", [])
        session_id = state.get("session_id", "")
        session = session_store.get_or_create(session_id) if session_store else ConversationState()

        # Build context from retrieved chunks
        if chunks:
            context_parts = []
            source_set: set[str] = set()
            for chunk in chunks:
                context_parts.append(chunk["content"])
                source_set.add(chunk["source"])
            context = "\n\n".join(context_parts)
            sources = sorted(source_set)
        else:
            context = "No relevant documents found in the knowledge base."
            sources = []

        session_info = _session_summary(session)
        history_ctx = _build_history_context(session)

        user_prompt = ANSWER_TEMPLATE.format(
            context=context,
            session=f"{session_info}\n\nConversation history:\n{history_ctx}",
            message=message,
        )

        try:
            response = await model.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as exc:
            logger.exception("LLM answer generation failed")
            from src.llm.client import handle_llm_exception
            raise handle_llm_exception(exc)

        if not response_text or not response_text.strip():
            response_text = (
                "I'm sorry, I couldn't find relevant information in our knowledge base "
                "to answer your question. Could you please rephrase or ask about our "
                "shipping, payment, return, or account policies?"
            )

        return {
            "response_text": response_text.strip(),
            "sources": sources,
            "ticket_id": session.ticket_id,
        }

    async def collect_or_create(state: SupportWorkflowState) -> SupportWorkflowState:
        """Collect missing ticket fields or create the ticket when complete."""
        message = state.get("customer_message", "")
        session_id = state.get("session_id", "")
        session = session_store.get_or_create(session_id) if session_store else ConversationState()

        # If ticket already created, inform customer
        if session.ticket_id:
            return {
                "response_text": (
                    f"Your support ticket {session.ticket_id} has already been created. "
                    "Is there anything else I can help you with?"
                ),
                "sources": [],
                "ticket_id": session.ticket_id,
            }

        # Extract fields from the current message using LLM
        missing = session.missing_ticket_fields()

        try:
            extract_prompt = (
                f"Customer message: {message}\n\n"
                f"Fields still needed: {', '.join(missing)}\n"
                f"Already collected: {_session_summary(session)}"
            )
            response = await model.ainvoke([
                SystemMessage(content=_EXTRACT_SYSTEM),
                HumanMessage(content=extract_prompt),
            ])
            extracted = _parse_json(
                response.content if hasattr(response, 'content') else str(response)
            )
        except Exception as exc:
            from src.llm.client import handle_llm_exception
            err = handle_llm_exception(exc)
            if any(term in str(err).lower() for term in ["rate limit", "authentication", "timed out", "unavailable", "connect"]):
                raise err
            logger.warning("Field extraction failed, using heuristic fallback: %s", exc)
            extracted = {}

        # Also try simple heuristic extraction as fallback
        extracted = _heuristic_extract(message, missing, extracted)

        # Update session with extracted fields (preserve already established values)
        if "customer_name" in extracted and extracted["customer_name"]:
            session.customer_name = extracted["customer_name"].strip()
        if "customer_email" in extracted and extracted["customer_email"]:
            email_val = extracted["customer_email"].strip()
            # Basic email validation
            if "@" in email_val and "." in email_val:
                session.customer_email = email_val
        if not session.issue_description and "issue_description" in extracted and extracted["issue_description"]:
            desc = extracted["issue_description"].strip()
            if len(desc) >= 5:
                session.issue_description = desc
        if not session.category and "category" in extracted and extracted["category"]:
            cat = extracted["category"].strip().lower()
            if cat in _VALID_CATEGORIES:
                session.category = cat

        # If this is the first turn in ticket mode and no issue_description
        # was extracted, use the current message as description
        if not session.issue_description and message.strip():
            # Use the customer message as the issue description if it's
            # descriptive enough (more than just a name or email)
            if len(message.strip()) >= 5 and "@" not in message and not _looks_like_name_only(message):
                session.issue_description = message.strip()

        # Check if all fields are now present
        still_missing = session.missing_ticket_fields()

        if not still_missing:
            # All fields collected — create the ticket
            try:
                tool = create_ticket_tool(ticket_repo, session_id)
                summary = _generate_summary(session)
                ticket_id = tool.invoke({
                    "customer_name": session.customer_name,
                    "customer_email": session.customer_email,
                    "issue_description": session.issue_description,
                    "category": session.category,
                    "summary": summary,
                })
                session.ticket_id = ticket_id
                logger.info("Ticket %s created for session %s", ticket_id, session_id)
                return {
                    "response_text": (
                        f"Your support ticket has been created successfully!\n\n"
                        f"**Ticket ID:** {ticket_id}\n"
                        f"**Name:** {session.customer_name}\n"
                        f"**Email:** {session.customer_email}\n"
                        f"**Category:** {session.category}\n"
                        f"**Issue:** {session.issue_description}\n\n"
                        f"Our team will follow up with you shortly."
                    ),
                    "sources": [],
                    "ticket_id": ticket_id,
                }
            except ValidationError as exc:
                logger.warning("Ticket validation failed: %s", exc)
                # Reset the invalid field and ask again
                return {
                    "response_text": (
                        "Some of the information provided doesn't appear to be valid. "
                        "Could you please verify your details? " + _FIELD_PROMPTS.get(
                            still_missing[0] if still_missing else "customer_email",
                            "Please provide the missing information."
                        )
                    ),
                    "sources": [],
                    "ticket_id": None,
                }
            except Exception:
                logger.exception("Ticket creation failed")
                raise AgentProcessingError(
                    "Unable to create your support ticket. Please try again."
                )
        else:
            # Ask for the next missing field
            next_field = still_missing[0]
            prompt_text = _FIELD_PROMPTS.get(
                next_field,
                f"Could you please provide your {next_field.replace('_', ' ')}?",
            )

            # Build a contextual response
            collected_info = []
            if session.customer_name:
                collected_info.append(f"Name: {session.customer_name}")
            if session.customer_email:
                collected_info.append(f"Email: {session.customer_email}")
            if session.issue_description:
                collected_info.append(f"Issue: {session.issue_description}")
            if session.category:
                collected_info.append(f"Category: {session.category}")

            if collected_info:
                context_msg = (
                    "Thank you! I have the following information so far:\n"
                    + "\n".join(f"- {info}" for info in collected_info)
                    + f"\n\n{prompt_text}"
                )
            else:
                context_msg = (
                    "I'd be happy to help you create a support ticket. "
                    f"To get started, {prompt_text.lower()}"
                )

            return {
                "response_text": context_msg,
                "sources": [],
                "ticket_id": None,
            }

    def select_route(state: SupportWorkflowState) -> str:
        """Return the routing decision from the decide node."""
        route = state.get("route", "")
        if route not in ("answer", "ticket"):
            logger.warning("Unexpected route '%s', defaulting to 'answer'", route)
            return "answer"
        return route

    # ----- graph assembly -------------------------------------------------

    graph = StateGraph(SupportWorkflowState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("decide", decide)
    graph.add_node("answer", answer)
    graph.add_node("ticket", collect_or_create)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "decide",
        select_route,
        {"answer": "answer", "ticket": "ticket"},
    )
    graph.add_edge("answer", END)
    graph.add_edge("ticket", END)
    return graph.compile()


# -- private helpers -------------------------------------------------------


def _is_clear_question(message: str) -> bool:
    """Check if the message is clearly asking a question rather than reporting an issue or giving details."""
    msg = message.strip()
    if not msg:
        return False

    # Check if message ends with question mark
    if msg.endswith("?"):
        if not re.search(r"\b(create|file|open|raise|submit)\s+(a\s+)?ticket\b", msg, re.IGNORECASE):
            return True

    # Check common question starters
    question_prefixes = (
        "what", "how", "why", "when", "where", "who",
        "do you", "does", "can i", "could you", "is there", "are there",
        "tell me about", "explain"
    )
    msg_lower = msg.lower()
    if any(msg_lower.startswith(p) for p in question_prefixes):
        if not any(k in msg_lower for k in ["want to report", "file a ticket", "create a ticket", "unresolved issue"]):
            return True

    return False


def _heuristic_extract(
    message: str, missing: list[str], already: dict[str, str]
) -> dict[str, str]:
    """Apply simple heuristics to supplement LLM extraction."""
    result = dict(already)
    msg = message.strip()

    # Email extraction
    if "customer_email" in missing and "customer_email" not in result:
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", msg)
        if email_match:
            result["customer_email"] = email_match.group(0)

    # Category extraction
    if "category" in missing and "category" not in result:
        msg_lower = msg.lower()
        for cat in _VALID_CATEGORIES:
            if cat in msg_lower:
                result["category"] = cat
                break

    # Name extraction
    if "customer_name" in missing and "customer_name" not in result:
        name_match = re.search(r"(?:my name is|i am|name is|this is)\s+([A-Za-z\s'-]+)", msg, re.IGNORECASE)
        if name_match:
            result["customer_name"] = name_match.group(1).strip().title()
        elif _looks_like_name_only(msg, missing):
            result["customer_name"] = msg.strip().title()

    return result


def _looks_like_name_only(message: str, missing: list[str] | None = None) -> bool:
    """Check if the message looks like it's just a person's name."""
    msg = message.strip().rstrip(".")
    words = msg.split()
    if 1 <= len(words) <= 4:
        if not re.search(r"[\d@#$%^&*()!?/\\=<>~`_{}\[\]]", msg):
            lower_msg = msg.lower()
            excluded_keywords = ["http", "ticket", "issue", "order", "payment", "account", "problem", "charged", "shipping", "refund"]
            if not any(k in lower_msg for k in excluded_keywords):
                if missing and missing[0] == "customer_name":
                    return True
                if all(w[0].isupper() or w[0] == "'" for w in words if w):
                    return True
    return False


def _generate_summary(session: ConversationState) -> str:
    """Generate a short summary for the ticket from collected fields."""
    cat = session.category or "other"
    desc = session.issue_description or "Customer issue"
    # Truncate to fit the 160-char max
    summary = f"{cat.capitalize()} issue: {desc}"
    if len(summary) > 160:
        summary = summary[:157] + "..."
    if len(summary) < 5:
        summary = f"Support ticket for {session.customer_name or 'customer'}"
    return summary
