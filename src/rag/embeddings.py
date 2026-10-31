import os

# Limit thread pools & buffer allocations to operate comfortably under 512MB RAM on free cloud tiers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    pass

from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Settings


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Create the configured local/open-source embedding adapter with minimal memory overhead."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
