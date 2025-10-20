"""
Configuration management for RAG systems.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    # Retrieval settings
    retrieval_mode: str = "sparse"  # "sparse", "dense", "hybrid"
    sparse_index: str = "wikipedia-dpr"
    dense_index: str = "wikipedia-dpr-100w.bpr-single-nq"
    dense_encoder: str = "castorini/bpr-nq-question-encoder"
    top_k: int = 5

    # Generation settings
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 200
    temperature: float = 0.0

    # Hybrid retrieval settings
    hybrid_alpha: float = 0.5  # Weight for sparse vs dense

    # Performance settings
    use_cache: bool = True
    cache_dir: str = ".cache/retrieval"
    concurrency: int = 10
    batch_size: int = 10

    # Evaluation settings
    d_f1_threshold: float = 0.5

    # API settings
    openai_api_key: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "retrieval_mode": self.retrieval_mode,
            "sparse_index": self.sparse_index,
            "dense_index": self.dense_index,
            "dense_encoder": self.dense_encoder,
            "top_k": self.top_k,
            "llm_model": self.llm_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "hybrid_alpha": self.hybrid_alpha,
            "use_cache": self.use_cache,
            "concurrency": self.concurrency,
            "batch_size": self.batch_size,
            "d_f1_threshold": self.d_f1_threshold,
            "metadata": self.metadata
        }

    @classmethod
    def from_args(cls, args) -> "RAGConfig":
        """Create config from argparse arguments."""
        return cls(
            retrieval_mode=args.retrieval_mode,
            sparse_index=args.sparse_index,
            dense_index=args.dense_index,
            dense_encoder=args.dense_encoder,
            top_k=args.top_k,
            llm_model=args.model,
            max_tokens=args.max_tokens,
            temperature=getattr(args, 'temperature', 0.0),
            use_cache=getattr(args, 'use_cache', True),
            concurrency=args.concurrency,
            batch_size=getattr(args, 'batch_size', 10),
            d_f1_threshold=getattr(args, 'd_f1_threshold', 0.5)
        )
