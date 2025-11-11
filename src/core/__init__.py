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
from .generators import BaseGenerator, OpenAIGenerator, HyDEGenerator, LlamaCppGenerator, LLAMA_CPP_AVAILABLE
from .cache import RetrievalCache
from .config import RAGConfig
from .output_utils import (
    get_model_name_from_config,
    get_organized_output_path,
    ensure_output_directory,
    load_existing_results,
    get_processed_question_ids,
    filter_unprocessed_data,
    merge_results
)

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
    "LlamaCppGenerator",
    "LLAMA_CPP_AVAILABLE",
    "RetrievalCache",
    "RAGConfig",
    "get_model_name_from_config",
    "get_organized_output_path",
    "ensure_output_directory",
    "load_existing_results",
    "get_processed_question_ids",
    "filter_unprocessed_data",
    "merge_results",
]
