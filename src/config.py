from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Environment-controlled model and retrieval settings.

    Configured for Groq API as the primary LLM provider while preserving
    compatibility with any OpenAI-compatible endpoint.
    """
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    vector_db_path: str
    embedding_model: str
    rag_collection: str
    rag_top_k: int
    api_host: str
    api_port: int
    streamlit_host: str
    streamlit_port: int

    @property
    def groq_api_key(self) -> str:
        """Alias for the LLM API key."""
        return self.llm_api_key


def load_settings() -> Settings:
    """Load configuration from environment variables with Groq API defaults."""
    load_dotenv()
    # Priority for API key: GROQ_API_KEY, then LLM_API_KEY
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY", "")

    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_api_key=api_key,
        llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        vector_db_path=os.getenv("VECTOR_DB_PATH", ".data/vector_db"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        rag_collection=os.getenv("RAG_COLLECTION", "customer-support"),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
        streamlit_host=os.getenv("STREAMLIT_HOST", "127.0.0.1"),
        streamlit_port=int(os.getenv("STREAMLIT_PORT", "8501")),
    )
