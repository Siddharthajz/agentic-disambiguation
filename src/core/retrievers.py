"""
Retriever components for RAG systems.

Provides modular retriever implementations:
- BaseRetriever: Abstract base class
- SparseRetriever: BM25 using PySerini/Lucene
- DenseRetriever: FAISS using sentence-transformers
- HybridRetriever: RRF fusion of sparse + dense
"""

import json
import time
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pyserini.search.lucene import LuceneSearcher

from .data_models import RetrievalResult
from .cache import RetrievalCache

logger = logging.getLogger(__name__)


def _extract_title_from_contents(contents: str) -> str:
    """
    Extract Wikipedia article title from PySerini document contents.

    The wikipedia-dpr index stores the title as the first line in quotes.
    Example: '"Three-point field goal"\nThe rest of the text...'

    Args:
        contents: Raw contents field from PySerini document

    Returns:
        Extracted title or 'Unknown' if extraction fails
    """
    if not contents:
        return 'Unknown'

    # Title is typically the first line in quotes
    first_line = contents.split('\n')[0].strip()

    # Remove surrounding quotes if present
    if first_line.startswith('"') and first_line.endswith('"'):
        return first_line[1:-1]

    # Fallback: return first line as-is
    return first_line if first_line else 'Unknown'


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    def __init__(self, top_k: int = 5, cache: Optional[RetrievalCache] = None):
        """
        Initialize retriever.

        Args:
            top_k: Number of documents to retrieve
            cache: Optional cache for results
        """
        self.top_k = top_k
        self.cache = cache

    @abstractmethod
    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve documents for a query.

        Args:
            query: Search query
            k: Number of documents to retrieve (overrides self.top_k)

        Returns:
            List of retrieval results
        """
        pass

    @abstractmethod
    def get_cache_params(self) -> Dict[str, Any]:
        """Get parameters for cache key generation."""
        pass

    def _check_cache(self, query: str, k: int, mode: str) -> Optional[List[RetrievalResult]]:
        """Check cache for results."""
        if self.cache is None:
            return None

        cached = self.cache.get(
            query=query,
            retrieval_mode=mode,
            top_k=k,
            **self.get_cache_params()
        )

        if cached:
            return [RetrievalResult(**doc) for doc in cached]
        return None

    def _save_cache(self, query: str, k: int, mode: str, results: List[RetrievalResult]):
        """Save results to cache."""
        if self.cache is None:
            return

        self.cache.set(
            query=query,
            retrieval_mode=mode,
            top_k=k,
            results=[r.to_dict() for r in results],
            **self.get_cache_params()
        )


class SparseRetriever(BaseRetriever):
    """BM25 retriever using PySerini/Lucene."""

    def __init__(
        self,
        index: str = "wikipedia-dpr",
        top_k: int = 5,
        cache: Optional[RetrievalCache] = None
    ):
        """
        Initialize sparse retriever.

        Args:
            index: PySerini prebuilt index name
            top_k: Number of documents to retrieve
            cache: Optional cache for results
        """
        super().__init__(top_k=top_k, cache=cache)
        self.index = index

        logger.info(f"Loading sparse retriever (BM25): {index}...")
        self.searcher = LuceneSearcher.from_prebuilt_index(index, True)
        logger.info("✓ Sparse retriever loaded")

    def get_cache_params(self) -> Dict[str, Any]:
        """Get cache parameters."""
        return {"index": self.index}

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve using BM25.

        Args:
            query: Search query
            k: Number of documents (overrides self.top_k)

        Returns:
            List of retrieval results
        """
        k = k or self.top_k

        # Check cache
        cached = self._check_cache(query, k, "sparse")
        if cached:
            return cached

        # Perform retrieval
        hits = self.searcher.search(query, k=k)
        results = []

        for rank, hit in enumerate(hits, 1):
            try:
                doc = self.searcher.doc(hit.docid)
                doc_dict = json.loads(doc.raw())
                contents = doc_dict.get('contents', '')
                results.append(RetrievalResult(
                    doc_id=hit.docid,
                    title=_extract_title_from_contents(contents),
                    text=contents,
                    score=hit.score,
                    rank=rank,
                    source="sparse"
                ))
            except Exception as e:
                logger.warning(f"Failed to parse document {hit.docid}: {e}")

        # Cache results
        self._save_cache(query, k, "sparse", results)

        return results


class DenseRetriever(BaseRetriever):
    """Dense retriever using FAISS and sentence-transformers."""

    def __init__(
        self,
        index: str = "ambigqa_wiki.index",
        encoder: str = "all-MiniLM-L6-v2",
        metadata_file: str = "ambigqa_wiki_metadata.json",
        top_k: int = 5,
        cache: Optional[RetrievalCache] = None
    ):
        """
        Initialize dense retriever.

        Args:
            index: Path to FAISS index file
            encoder: Sentence-transformers model name
            metadata_file: Path to metadata JSON file
            top_k: Number of documents to retrieve
            cache: Optional cache for results
        """
        super().__init__(top_k=top_k, cache=cache)
        self.index_path = index
        self.encoder_name = encoder
        self.metadata_path = metadata_file

        logger.info(f"Loading dense retriever (FAISS): {index}...")
        logger.info(f"Loading query encoder: {encoder}...")

        # Load FAISS index
        if not os.path.exists(index):
            raise FileNotFoundError(
                f"FAISS index not found at {index}. "
                f"Please run the index building script first: "
                f"python scripts/build_faiss_index.py --output-dir ./data"
            )
        self.index = faiss.read_index(index)
        logger.info(f"✓ FAISS index loaded with {self.index.ntotal} vectors")

        # Load metadata
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_file}. "
                f"Please run the index building script first: "
                f"python scripts/build_faiss_index.py --output-dir ./data"
            )
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logger.info(f"✓ Loaded metadata for {len(self.metadata)} documents")

        # Load sentence-transformers model
        self.encoder = SentenceTransformer(encoder)
        logger.info("✓ Dense retriever loaded")

    def get_cache_params(self) -> Dict[str, Any]:
        """Get cache parameters."""
        return {"index": self.index_path, "encoder": self.encoder_name}

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve using FAISS.

        Args:
            query: Search query
            k: Number of documents (overrides self.top_k)

        Returns:
            List of retrieval results
        """
        k = k or self.top_k

        # Check cache
        cached = self._check_cache(query, k, "dense")
        if cached:
            return cached

        # Encode query and normalize for cosine similarity
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Perform FAISS search
        scores, indices = self.index.search(query_embedding, k)

        # Convert to results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx < 0 or idx >= len(self.metadata):
                logger.warning(f"Invalid index {idx} returned by FAISS")
                continue

            doc_metadata = self.metadata[idx]
            results.append(RetrievalResult(
                doc_id=str(idx),
                title=doc_metadata.get('title', 'Unknown'),
                text=doc_metadata.get('text', ''),
                score=float(score),
                rank=rank,
                source="dense"
            ))

        # Cache results
        self._save_cache(query, k, "dense", results)

        return results


class HybridRetriever(BaseRetriever):
    """Hybrid retriever using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        sparse_retriever: SparseRetriever,
        dense_retriever: DenseRetriever,
        top_k: int = 5,
        rrf_k: int = 60,
        cache: Optional[RetrievalCache] = None
    ):
        """
        Initialize hybrid retriever.

        Args:
            sparse_retriever: BM25 retriever
            dense_retriever: FAISS retriever
            top_k: Number of documents to retrieve
            rrf_k: RRF constant (default 60 from original paper)
            cache: Optional cache for results
        """
        super().__init__(top_k=top_k, cache=cache)
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

    def get_cache_params(self) -> Dict[str, Any]:
        """Get cache parameters."""
        return {
            "sparse_index": self.sparse_retriever.index,
            "dense_index": self.dense_retriever.index_path,
            "dense_encoder": self.dense_retriever.encoder_name,
            "rrf_k": self.rrf_k
        }

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve using RRF fusion.

        Args:
            query: Search query
            k: Number of documents (overrides self.top_k)

        Returns:
            List of retrieval results ranked by RRF
        """
        k = k or self.top_k

        # Check cache
        cached = self._check_cache(query, k, "hybrid")
        if cached:
            return cached

        # Get results from both retrievers
        sparse_results = self.sparse_retriever.retrieve(query, k=k)
        dense_results = self.dense_retriever.retrieve(query, k=k)

        # RRF scoring
        rrf_scores = {}
        doc_map = {}

        # Process sparse results
        for res in sparse_results:
            rrf_scores[res.doc_id] = rrf_scores.get(res.doc_id, 0) + 1 / (self.rrf_k + res.rank)
            doc_map[res.doc_id] = res

        # Process dense results
        for res in dense_results:
            rrf_scores[res.doc_id] = rrf_scores.get(res.doc_id, 0) + 1 / (self.rrf_k + res.rank)
            if res.doc_id not in doc_map:
                doc_map[res.doc_id] = res

        # Sort by RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Create final results
        results = []
        for rank, doc_id in enumerate(sorted_doc_ids[:k], 1):
            original = doc_map[doc_id]
            results.append(RetrievalResult(
                doc_id=doc_id,
                title=original.title,
                text=original.text,
                score=rrf_scores[doc_id],
                rank=rank,
                source="hybrid"
            ))

        # Cache results
        self._save_cache(query, k, "hybrid", results)

        return results


def create_retriever(
    mode: str,
    sparse_index: str = "wikipedia-dpr",
    dense_index: str = "ambigqa_wiki.index",
    dense_encoder: str = "all-MiniLM-L6-v2",
    dense_metadata: str = "ambigqa_wiki_metadata.json",
    top_k: int = 5,
    cache: Optional[RetrievalCache] = None
) -> BaseRetriever:
    """
    Factory function to create retrievers.

    Args:
        mode: "sparse", "dense", or "hybrid"
        sparse_index: BM25 index name
        dense_index: Path to FAISS index file
        dense_encoder: Sentence-transformers model name
        dense_metadata: Path to metadata JSON file
        top_k: Number of documents to retrieve
        cache: Optional cache

    Returns:
        Configured retriever instance
    """
    if mode == "sparse":
        return SparseRetriever(index=sparse_index, top_k=top_k, cache=cache)

    elif mode == "dense":
        return DenseRetriever(
            index=dense_index,
            encoder=dense_encoder,
            metadata_file=dense_metadata,
            top_k=top_k,
            cache=cache
        )

    elif mode == "hybrid":
        sparse = SparseRetriever(index=sparse_index, top_k=top_k, cache=cache)
        dense = DenseRetriever(
            index=dense_index,
            encoder=dense_encoder,
            metadata_file=dense_metadata,
            top_k=top_k,
            cache=cache
        )
        return HybridRetriever(sparse, dense, top_k=top_k, cache=cache)

    else:
        raise ValueError(f"Invalid retrieval mode: {mode}")
