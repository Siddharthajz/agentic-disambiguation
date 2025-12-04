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
    dense_index: str = "data/ambigqa_wiki.index"
    dense_encoder: str = "all-MiniLM-L6-v2"
    dense_metadata: str = "data/ambigqa_wiki_metadata.json"
    top_k: int = 5

    # Generation settings
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 200
    temperature: float = 0.0

    # Local LLM settings
    use_local_llm: bool = False
    local_model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    local_context_size: int = 4096
    local_gpu_layers: int = -1

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

    ambiguity_detection_method: str = "question"
    question_ambiguity_model_path: str = "models/distilbert-classifier"
    classifier_uncertainty_threshold: float = 0.6



    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, clean_for_mode: bool = False) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Args:
            clean_for_mode: If True, only include fields relevant to the retrieval mode
        """
        base_config = {
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "use_cache": self.use_cache,
            "concurrency": self.concurrency,
            "batch_size": self.batch_size,
            "d_f1_threshold": self.d_f1_threshold,
            "metadata": self.metadata
        }

        # Add LLM configuration based on whether using local or OpenAI
        if self.use_local_llm:
            base_config["use_local_llm"] = True
            base_config["local_model_path"] = self.local_model_path
            base_config["local_context_size"] = self.local_context_size
            base_config["local_gpu_layers"] = self.local_gpu_layers
        else:
            base_config["llm_model"] = self.llm_model

        if clean_for_mode:
            # Only include retrieval-specific fields based on mode
            if self.retrieval_mode == "sparse":
                base_config["sparse_index"] = self.sparse_index
            elif self.retrieval_mode == "dense":
                base_config["dense_index"] = self.dense_index
                base_config["dense_encoder"] = self.dense_encoder
                base_config["dense_metadata"] = self.dense_metadata
            elif self.retrieval_mode == "hybrid":
                base_config["sparse_index"] = self.sparse_index
                base_config["dense_index"] = self.dense_index
                base_config["dense_encoder"] = self.dense_encoder
                base_config["dense_metadata"] = self.dense_metadata
                base_config["hybrid_alpha"] = self.hybrid_alpha
        else:
            # Include all fields
            base_config["sparse_index"] = self.sparse_index
            base_config["dense_index"] = self.dense_index
            base_config["dense_encoder"] = self.dense_encoder
            base_config["dense_metadata"] = self.dense_metadata
            base_config["hybrid_alpha"] = self.hybrid_alpha

        return base_config

    @classmethod
    def from_args(cls, args) -> "RAGConfig":
        """Create config from argparse arguments."""
        return cls(
            retrieval_mode=args.retrieval_mode,
            sparse_index=args.sparse_index,
            dense_index=args.dense_index,
            dense_encoder=args.dense_encoder,
            dense_metadata=getattr(args, 'dense_metadata', 'data/ambigqa_wiki_metadata.json'),
            top_k=args.top_k,
            llm_model=args.model,
            max_tokens=args.max_tokens,
            temperature=getattr(args, 'temperature', 0.0),
            use_local_llm=getattr(args, 'use_local_llm', False),
            local_model_path=getattr(args, 'local_model_path', 'models/qwen2.5-3b-instruct-q4_k_m.gguf'),
            local_context_size=getattr(args, 'local_context_size', 4096),
            local_gpu_layers=getattr(args, 'local_gpu_layers', -1),
            use_cache=getattr(args, 'use_cache', True),
            concurrency=args.concurrency,
            batch_size=getattr(args, 'batch_size', 10),
            d_f1_threshold=getattr(args, 'd_f1_threshold', 0.5)
        )
