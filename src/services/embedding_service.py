"""
Persistent embedding service for efficient training.

This service runs as a Ray actor, loads models once, and serves all training workers.
Avoids reloading 2.2GB of models on every episode.

Features:
- LRU cache with configurable size limit (prevents disk space issues)
- Automatic eviction of least-recently-used indices
- Disk usage monitoring
"""

import ray
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from collections import OrderedDict
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.tools.semantic_search import SemanticSearch


@ray.remote(num_cpus=2, num_gpus=0.1)  # Adjust GPU allocation as needed
class EmbeddingService:
    """
    Persistent Ray actor that handles all embedding operations.

    Models are loaded once and reused across all training episodes.
    """

    def __init__(
        self,
        embedding_model: str = "jinaai/jina-code-embeddings-0.5b",
        reranker_model: str = "jinaai/jina-reranker-v3",
        device: str = "cpu",  # or "cuda" if GPU available
        cache_dir: Optional[str] = None,
        max_indices: int = 50,  # Max number of indices to keep (LRU eviction)
        max_cache_size_gb: Optional[float] = None,  # Max total disk space (GB)
    ):
        """
        Initialize embedding service with persistent models.

        Args:
            max_indices: Maximum number of indices to keep loaded (LRU eviction)
            max_cache_size_gb: Maximum total disk space for cache (GB), None = unlimited
        """
        print(f"[EmbeddingService] Loading models on {device}...")

        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.device = device
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "swebench_indices")
        self.max_indices = max_indices
        self.max_cache_size_gb = max_cache_size_gb

        # Load models ONCE (this is the expensive part)
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.reranker = CrossEncoder(reranker_model, device=device) if reranker_model else None

        # LRU cache for loaded indices (OrderedDict maintains insertion order)
        self.indices_cache = OrderedDict()

        print(f"[EmbeddingService] Models loaded successfully!")
        print(f"[EmbeddingService] LRU cache: max {max_indices} indices")
        if max_cache_size_gb:
            print(f"[EmbeddingService] Disk limit: {max_cache_size_gb:.1f} GB")

    def _get_dir_size_mb(self, path: Path) -> float:
        """Get directory size in MB."""
        if not path.exists():
            return 0.0
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024 * 1024)

    def _get_total_cache_size_gb(self) -> float:
        """Get total cache size in GB."""
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            return 0.0
        total_mb = sum(
            self._get_dir_size_mb(d) for d in cache_path.iterdir() if d.is_dir()
        )
        return total_mb / 1024

    def _evict_lru_index(self):
        """Evict the least recently used index from cache and disk."""
        if not self.indices_cache:
            return

        # Remove from memory (FIFO in OrderedDict = LRU)
        lru_hash, lru_index = self.indices_cache.popitem(last=False)

        # Remove from disk
        persist_dir = Path(self.cache_dir) / lru_hash
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
            size_mb = self._get_dir_size_mb(persist_dir)
            print(f"[EmbeddingService] Evicted LRU index {lru_hash} ({size_mb:.1f} MB)")

    def _enforce_cache_limits(self):
        """Enforce cache size limits by evicting LRU indices."""
        # Enforce max_indices limit
        while len(self.indices_cache) >= self.max_indices:
            self._evict_lru_index()

        # Enforce max_cache_size_gb limit
        if self.max_cache_size_gb:
            while self._get_total_cache_size_gb() > self.max_cache_size_gb:
                self._evict_lru_index()
                if not self.indices_cache:
                    break  # Safety: don't infinite loop

    def get_or_load_index(self, repo_name: str, commit: str) -> SemanticSearch:
        """Get or load index for a specific repo+commit with LRU caching."""
        from src.mcp_server.semantic_search_server import get_repo_commit_hash

        repo_commit_hash = get_repo_commit_hash(repo_name, commit)

        # Check if already in cache (and move to end for LRU)
        if repo_commit_hash in self.indices_cache:
            # Move to end (most recently used)
            self.indices_cache.move_to_end(repo_commit_hash)
            return self.indices_cache[repo_commit_hash]

        # Not in cache - need to load
        persist_dir = Path(self.cache_dir) / repo_commit_hash

        if not persist_dir.exists():
            raise ValueError(
                f"Index not found for {repo_name}@{commit[:8]}. "
                f"Run pre-indexing first: python preindex_swebench.py --min-frequency 3"
            )

        # Enforce cache limits BEFORE loading new index
        self._enforce_cache_limits()

        # Load pre-computed index (fast, no model inference needed)
        index = SemanticSearch(
            collection_name=f"code_{repo_commit_hash}",
            persist_directory=str(persist_dir),
            embedding_model_name=self.embedding_model_name,
            reranker_model_name=self.reranker_model_name,
        )

        # Reuse our already-loaded models instead of loading new ones
        index.embedder = self.embedder
        index.reranker = self.reranker

        # Add to cache (at end = most recently used)
        self.indices_cache[repo_commit_hash] = index
        print(f"[EmbeddingService] Loaded index for {repo_name}@{commit[:8]} (cache: {len(self.indices_cache)}/{self.max_indices})")

        return self.indices_cache[repo_commit_hash]

    def search(
        self,
        query: str,
        repo_name: str,
        commit: str,
        n_results: int = 10,
    ) -> List[Dict]:
        """
        Perform semantic search using pre-loaded models and index.

        This is FAST because:
        1. Models already loaded (no 2.2GB reload)
        2. Code embeddings pre-computed (just query embedding + similarity)
        3. Reranker already loaded (no reload)
        """
        index = self.get_or_load_index(repo_name, commit)
        results = index.search(query, n_results=n_results)
        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query (useful for caching)."""
        return self.embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True)

    def embed_batch(self, queries: List[str]) -> np.ndarray:
        """Embed multiple queries efficiently."""
        return self.embedder.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def get_cache_stats(self) -> Dict:
        """Get statistics about loaded indices and disk usage."""
        total_cache_gb = self._get_total_cache_size_gb()
        return {
            "loaded_indices": len(self.indices_cache),
            "max_indices": self.max_indices,
            "indices": list(self.indices_cache.keys()),
            "total_cache_size_gb": round(total_cache_gb, 2),
            "max_cache_size_gb": self.max_cache_size_gb,
            "device": self.device,
            "embedding_model": self.embedding_model_name,
            "reranker_model": self.reranker_model_name,
        }


def get_embedding_service(
    device: str = "cpu",
    max_indices: int = 50,
    max_cache_size_gb: Optional[float] = None,
) -> ray.ObjectRef:
    """
    Get or create the shared embedding service with LRU caching.

    Call this once at training start to initialize the service.
    All workers will share this single instance.

    Args:
        device: Device for embeddings ('cpu' or 'cuda')
        max_indices: Maximum number of indices to keep (LRU eviction)
        max_cache_size_gb: Maximum total disk space for cache (GB), None = unlimited
    """
    try:
        # Try to get existing service
        service = ray.get_actor("embedding_service")
        print("[EmbeddingService] Using existing service")
    except ValueError:
        # Create new service
        print(f"[EmbeddingService] Creating new service on {device}")
        service = EmbeddingService.options(name="embedding_service").remote(
            device=device,
            max_indices=max_indices,
            max_cache_size_gb=max_cache_size_gb,
        )
        print("[EmbeddingService] Service created")

    return service