"""
Data models for RAG pipeline.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class RetrievalResult:
    """Stores a single retrieval result."""
    doc_id: str
    title: str
    text: str
    score: float
    rank: int
    source: str = "sparse"  # "sparse", "dense", or "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RAGResult:
    """Stores the result of a RAG pipeline run."""
    question_id: str
    question: str
    retrieved_docs: List[Dict[str, Any]]
    generated_answer: str
    reference_data: Dict[str, Any]
    retrieval_time: float
    generation_time: float
    total_tokens: int
    evaluation: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None  # For approach-specific data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
