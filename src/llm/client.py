from __future__ import annotations

import logging
from langchain_openai import ChatOpenAI

from src.config import Settings
from src.utils.errors import AgentProcessingError

logger = logging.getLogger(__name__)


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Create the configured LangChain chat-model adapter for Groq API.

    Points to the Groq OpenAI-compatible endpoint (https://api.groq.com/openai/v1)
    and uses the configured model (default: openai/gpt-oss-20b).
    """
    api_key = settings.llm_api_key or "not-set"
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=api_key,
        model=settings.llm_model,
        temperature=0.2,
        max_retries=2,
        timeout=30.0,
    )


def handle_llm_exception(exc: Exception) -> AgentProcessingError:
    """Format Groq/OpenAI LLM exceptions into clear, descriptive AgentProcessingErrors."""
    exc_type = type(exc).__name__
    exc_str = str(exc)
    exc_lower = exc_str.lower()

    if "authentication" in exc_type.lower() or "401" in exc_str:
        return AgentProcessingError(
            "Groq API authentication failed. Please verify that GROQ_API_KEY is set and valid."
        )
    if "ratelimit" in exc_type.lower() or "429" in exc_str:
        return AgentProcessingError(
            "Groq API rate limit exceeded. Please wait a moment and try again."
        )
    if "timeout" in exc_type.lower() or "timed out" in exc_lower:
        return AgentProcessingError(
            "Groq API request timed out. Please try again."
        )
    if "connection" in exc_type.lower() or "connect" in exc_lower:
        return AgentProcessingError(
            "Unable to connect to Groq API. Please check your network connection."
        )
    if "internalserver" in exc_type.lower() or "500" in exc_str or "503" in exc_str or "unavailable" in exc_lower:
        return AgentProcessingError(
            "Groq API service is currently unavailable. Please try again later."
        )
    return AgentProcessingError(
        f"Groq API error: {exc_str}"
    )
