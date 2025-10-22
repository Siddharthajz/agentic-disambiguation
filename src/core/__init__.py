"""
Core library for RAG-based disambiguation systems.

This module provides shared components for all RAG approaches:
- Base abstractions for retrievers and generators
- Data models (RetrievalResult, RAGResult)
- Caching layer
- Configuration management
"""

from .data_models import RetrievalResult, RAGResult
from .retrievers import (
    BaseRetriever,
    SparseRetriever,
    DenseRetriever,
    HybridRetriever,
    create_retriever
)
from .generators import BaseGenerator, OpenAIGenerator, HyDEGenerator
from .cache import RetrievalCache
from .config import RAGConfig

__all__ = [
    "RetrievalResult",
    "RAGResult",
    "BaseRetriever",
    "SparseRetriever",
    "DenseRetriever",
    "HybridRetriever",
    "create_retriever",
    "BaseGenerator",
    "OpenAIGenerator",
    "HyDEGenerator",
    "RetrievalCache",
    "RAGConfig",
]
