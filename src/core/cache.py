"""
Caching layer for retrieval results.

Caches retrieval results to avoid redundant searches.
BM25 and dense retrieval are deterministic, so same query = same results.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RetrievalCache:
    """Cache for retrieval results."""

    def __init__(self, cache_dir: str = ".cache/retrieval", enabled: bool = True):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Retrieval cache enabled: {self.cache_dir}")
        else:
            logger.info("Retrieval cache disabled")

    def _get_cache_key(self, query: str, retrieval_mode: str, top_k: int, **kwargs) -> str:
        """
        Generate cache key from query and retrieval parameters.

        Args:
            query: Search query
            retrieval_mode: "sparse", "dense", or "hybrid"
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters (e.g., index names)

        Returns:
            Cache key (hash)
        """
        # Create a deterministic string from all parameters
        params = {
            "query": query,
            "mode": retrieval_mode,
            "top_k": top_k,
            **kwargs
        }
        params_str = json.dumps(params, sort_keys=True)
        cache_key = hashlib.md5(params_str.encode()).hexdigest()
        return cache_key

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, query: str, retrieval_mode: str, top_k: int, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached retrieval results.

        Args:
            query: Search query
            retrieval_mode: Retrieval mode
            top_k: Number of documents
            **kwargs: Additional parameters

        Returns:
            Cached results or None if not found
        """
        if not self.enabled:
            return None

        cache_key = self._get_cache_key(query, retrieval_mode, top_k, **kwargs)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_data['results']
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None

        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None

    def set(self, query: str, retrieval_mode: str, top_k: int, results: List[Dict[str, Any]], **kwargs):
        """
        Cache retrieval results.

        Args:
            query: Search query
            retrieval_mode: Retrieval mode
            top_k: Number of documents
            results: Retrieval results to cache
            **kwargs: Additional parameters
        """
        if not self.enabled:
            return

        cache_key = self._get_cache_key(query, retrieval_mode, top_k, **kwargs)
        cache_path = self._get_cache_path(cache_key)

        try:
            cache_data = {
                "query": query,
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "params": kwargs,
                "results": results
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Cached results for query: {query[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def clear(self):
        """Clear all cached results."""
        if not self.enabled:
            return

        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info(f"Cleared cache: {self.cache_dir}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or not self.cache_dir.exists():
            return {"enabled": False, "num_entries": 0, "total_size": 0}

        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "enabled": True,
            "num_entries": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }
