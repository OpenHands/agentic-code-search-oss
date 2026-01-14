"""
Two-phase embedding service for efficient indexing and retrieval.

PHASE 1 - INDEXING (GPU):
- Use GPU for fast embedding during index creation
- No training happens during this phase
- Maximum GPU utilization for indexing speed

PHASE 2 - RETRIEVAL (CPU):  
- Move models to CPU after indexing complete
- Free GPU for training
- Handle search queries on CPU (no GPU contention)

Workflow:
1. enter_indexing_phase() -> Load models on GPU
2. Create indices (GPU-accelerated)
3. enter_retrieval_phase() -> Move models to CPU
4. Training starts (GPU free for training)
5. Search queries (CPU-based, no contention)
"""

import ray
import shutil
import gc
from pathlib import Path
from typing import List, Dict, Optional
from collections import OrderedDict
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.tools.semantic_search import SemanticSearch


@ray.remote(num_cpus=4, num_gpus=1)  # Allocate 1 GPU for indexing phase
class EmbeddingService:
    """
    Two-phase embedding service for efficient indexing and retrieval.
    
    Phase 1 - INDEXING (GPU):
    - Load models on GPU for fast embedding
    - Create indices using GPU acceleration
    - No training happens during this phase
    
    Phase 2 - RETRIEVAL (CPU):
    - Move models to CPU after indexing
    - Free GPU for training
    - Handle search queries on CPU
    
    Key features:
    - GPU-accelerated indexing
    - CPU-based retrieval (no GPU contention)
    - Phase-aware model placement
    - LRU cache for indices
    - Batch-aware cleanup
    """

    def __init__(
        self,
        embedding_model: str = "jinaai/jina-code-embeddings-0.5b",
        reranker_model: str = "jinaai/jina-reranker-v3",
        cache_dir: Optional[str] = None,
        max_indices: int = 20,
        max_cache_size_gb: Optional[float] = None,
    ):
        """
        Initialize embedding service.

        Args:
            max_indices: Maximum number of indices to keep loaded (LRU eviction)
            max_cache_size_gb: Maximum total disk space for cache (GB), None = unlimited
        """
        print(f"[EmbeddingService] Initializing two-phase service...")

        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "swebench_indices")
        self.max_indices = max_indices
        self.max_cache_size_gb = max_cache_size_gb

        # Track current phase and device
        self.current_phase = None  # 'indexing', 'retrieval', or None
        self.current_device = None
        
        # Models start unloaded
        self.embedder = None
        self.reranker = None
        self.models_loaded = False

        # LRU cache for loaded indices
        self.indices_cache = OrderedDict()

        print(f"[EmbeddingService] Initialized")
        print(f"[EmbeddingService] LRU cache: max {max_indices} indices")
        if max_cache_size_gb:
            print(f"[EmbeddingService] Disk limit: {max_cache_size_gb:.1f} GB")
    
    def enter_indexing_phase(self):
        """
        Enter indexing phase: Load models on GPU for fast embedding.
        
        Call this BEFORE indexing a new batch.
        No training should happen during this phase.
        """
        if self.current_phase == 'indexing':
            print("[EmbeddingService] Already in indexing phase")
            return
        
        print("\n" + "="*80)
        print("[EmbeddingService] ⚡ ENTERING INDEXING PHASE (GPU)")
        print("="*80)
        
        import torch
        
        # Unload if already loaded on different device
        if self.models_loaded:
            self._unload_models()
        
        # Check GPU availability
        if not torch.cuda.is_available():
            print("[EmbeddingService] WARNING: No GPU available, using CPU")
            device = "cpu"
            torch.set_num_threads(8)
        else:
            device = "cuda"
            print(f"[EmbeddingService] Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Load models on GPU
        print(f"[EmbeddingService] Loading models on {device}...")
        self.embedder = SentenceTransformer(
            self.embedding_model_name, 
            device=device
        )
        
        if self.reranker_model_name:
            self.reranker = CrossEncoder(
                self.reranker_model_name, 
                device=device
            )
        else:
            self.reranker = None
        
        self.models_loaded = True
        self.current_device = device
        self.current_phase = 'indexing'
        
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[EmbeddingService] GPU memory after loading: {allocated:.2f}GB")
        
        print(f"[EmbeddingService] ✓ Models loaded on {device} for indexing")
        print("="*80 + "\n")
    
    def enter_retrieval_phase(self):
        """
        Enter retrieval phase: Move models to CPU to free GPU for training.
        
        Call this AFTER indexing is complete, BEFORE training starts.
        Models will be moved from GPU to CPU.
        """
        if self.current_phase == 'retrieval':
            print("[EmbeddingService] Already in retrieval phase")
            return
        
        print("\n" + "="*80)
        print("[EmbeddingService] 🔄 ENTERING RETRIEVAL PHASE (CPU)")
        print("="*80)
        
        import torch
        
        if not self.models_loaded:
            print("[EmbeddingService] Loading models on CPU...")
            # Load directly on CPU
            torch.set_num_threads(4)
            self.embedder = SentenceTransformer(
                self.embedding_model_name, 
                device="cpu"
            )
            if self.reranker_model_name:
                self.reranker = CrossEncoder(
                    self.reranker_model_name, 
                    device="cpu"
                )
        else:
            # Move existing models from GPU to CPU
            print(f"[EmbeddingService] Moving models from {self.current_device} to CPU...")
            
            self.embedder.to("cpu")
            if self.reranker:
                self.reranker.model.to("cpu")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                before_clear = torch.cuda.memory_allocated() / 1024**3
                torch.cuda.empty_cache()
                after_clear = torch.cuda.memory_allocated() / 1024**3
                freed = before_clear - after_clear
                print(f"[EmbeddingService] GPU memory freed: {freed:.2f}GB")
        
        self.models_loaded = True
        self.current_device = "cpu"
        self.current_phase = 'retrieval'
        
        # Also move any loaded indices to CPU
        for index_hash, index in self.indices_cache.items():
            if hasattr(index, 'embedder') and index.embedder is not None:
                index.embedder = self.embedder
            if hasattr(index, 'reranker') and index.reranker is not None:
                index.reranker = self.reranker
        
        print(f"[EmbeddingService] ✓ Models on CPU for retrieval (GPU freed for training)")
        print("="*80 + "\n")
    
    def _unload_models(self):
        """Completely unload models from memory."""
        import torch
        
        print(f"[EmbeddingService] Unloading models from {self.current_device}...")
        
        del self.embedder
        del self.reranker
        
        self.embedder = None
        self.reranker = None
        self.models_loaded = False
        self.current_device = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[EmbeddingService] Models unloaded")
    
    def offload_for_training(self):
        """
        Prepare for training: Move to CPU if needed, clear GPU cache.
        
        This is equivalent to enter_retrieval_phase() but more explicit.
        Call this RIGHT BEFORE training starts.
        """
        print("\n[EmbeddingService] 🎯 PREPARING FOR TRAINING")
        self.enter_retrieval_phase()
        
        import torch
        if torch.cuda.is_available():
            # Extra cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            # Report GPU memory
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[EmbeddingService] GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            print("[EmbeddingService] ✓ GPU ready for training\n")
    
    def ensure_models_loaded(self):
        """
        Ensure models are loaded on appropriate device.
        
        If no phase is set, defaults to retrieval (CPU).
        """
        if not self.models_loaded:
            if self.current_phase == 'indexing':
                self.enter_indexing_phase()
            else:
                # Default to retrieval phase (CPU)
                self.enter_retrieval_phase()

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
            size_mb = self._get_dir_size_mb(persist_dir)
            shutil.rmtree(persist_dir)
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
    
    def clear_all_indices(self):
        """Clear ALL loaded indices from memory (not disk)."""
        self.indices_cache.clear()
        gc.collect()
        print(f"[EmbeddingService] Cleared all loaded indices from memory")
    
    def cleanup_batch_indices(self, repo_hashes: List[str]):
        """
        Clean up indices for the given repo hashes.
        
        Args:
            repo_hashes: List of repository hashes to cleanup
        """
        # self.logger.info(f"[EmbeddingService] Cleaning up {len(repo_hashes)} indices")
        
        for repo_hash in repo_hashes:
            persist_dir = Path(self.cache_dir) / repo_hash
            
            if not persist_dir.exists():
                # self.logger.warning(f"[EmbeddingService] Index directory not found: {persist_dir}")
                continue
            
            try:
                # ✅ More robust deletion - handle locked files
                import time
                
                # First attempt: normal removal
                shutil.rmtree(persist_dir, ignore_errors=False)
                # self.logger.info(f"[EmbeddingService] ✓ Cleaned up index: {repo_hash}")
                
            except OSError as e:
                if e.errno == 39:  # Directory not empty
                    # self.logger.warning(f"[EmbeddingService] Directory not empty, trying force delete: {persist_dir}")
                    try:
                        # Second attempt: force delete with error handling
                        def handle_remove_error(func, path, exc_info):
                            """Error handler for shutil.rmtree"""
                            import stat
                            import os
                            
                            # Try to change permissions and retry
                            if not os.access(path, os.W_OK):
                                os.chmod(path, stat.S_IWUSR)
                                try:
                                    func(path)
                                except Exception:
                                    pass  # Ignore if still fails
                        
                        shutil.rmtree(persist_dir, onerror=handle_remove_error)
                        # self.logger.info(f"[EmbeddingService] ✓ Force-cleaned index: {repo_hash}")
                        
                    except Exception as e2:
                        pass
                        # self.logger.warning(f"[EmbeddingService] Could not fully delete {persist_dir}: {e2}")
                        # self.logger.warning(f"[EmbeddingService] This is non-fatal - continuing anyway")
                else:
                    pass
                    # self.logger.warning(f"[EmbeddingService] Error cleaning up {persist_dir}: {e}")
                    # self.logger.warning(f"[EmbeddingService] This is non-fatal - continuing anyway")
            
            except Exception as e:
                pass
                # self.logger.warning(f"[EmbeddingService] Unexpected error cleaning up {persist_dir}: {e}")
                # self.logger.warning(f"[EmbeddingService] This is non-fatal - continuing anyway")
        
        # self.logger.info(f"[EmbeddingService] Cleanup complete")

    def get_or_load_index(self, repo_name: str, commit: str, repo_path: Optional[str] = None) -> SemanticSearch:
        """
        Get or load index for a specific repo+commit with LRU caching.
        
        Args:
            repo_name: Repository name (e.g., "django/django")
            commit: Commit hash
            repo_path: Path to cloned repository (required for creating new indices)
        
        Uses current device (GPU for indexing, CPU for retrieval).
        """
        # Ensure models are loaded on appropriate device
        self.ensure_models_loaded()
        
        from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
        repo_commit_hash = get_repo_commit_hash(repo_name, commit)

        # Check if already in cache (and move to end for LRU)
        if repo_commit_hash in self.indices_cache:
            self.indices_cache.move_to_end(repo_commit_hash)
            return self.indices_cache[repo_commit_hash]

        persist_dir = Path(self.cache_dir) / repo_commit_hash
        ready_file = persist_dir / ".ready"

        # If index doesn't exist with .ready marker
        if not ready_file.exists():
            if repo_path is None:
                raise ValueError(
                    f"Index not found for {repo_name}@{commit[:8]} and no repo_path provided. "
                    f"Expected .ready marker at: {ready_file}\n"
                    f"Run pre-indexing first or provide repo_path to create index."
                )
            
            # CREATE NEW INDEX (only during indexing phase)
            if self.current_phase != 'indexing':
                raise ValueError(
                    f"Cannot create index during {self.current_phase} phase. "
                    f"Index for {repo_name}@{commit[:8]} must be pre-created. "
                    f"Switch to indexing phase first with enter_indexing_phase()."
                )
            
            print(f"[EmbeddingService] Creating NEW index for {repo_name}@{commit[:8]}...")
            print(f"[EmbeddingService]   Repo path: {repo_path}")
            print(f"[EmbeddingService]   Index dir: {persist_dir}")
            
            # Enforce cache limits BEFORE creating
            self._enforce_cache_limits()
            
            # Create new index
            index = SemanticSearch(
                collection_name=f"code_{repo_commit_hash}",
                persist_directory=str(persist_dir),
                embedding_model_name=self.embedding_model_name,
                reranker_model_name=self.reranker_model_name,
                device=self.current_device,
                num_threads=4 if self.current_device == "cpu" else 1,
            )
            
            # Reuse our models
            index.embedder = self.embedder
            index.reranker = self.reranker
            
            # Index the repository
            try:
                stats = index.index_code_files(repo_path, file_extensions=[".py"])
            except Exception as e:
                print(f"[EmbeddingService] ✗ Indexing failed: {e}")
                # Clean up partial index
                if persist_dir.exists():
                    shutil.rmtree(persist_dir)
                raise ValueError(f"Failed to index {repo_name}@{commit[:8]}: {e}") from e
            
            # Create .ready marker after successful indexing
            if stats["total_chunks"] > 0:
                ready_file.touch()
                print(f"[EmbeddingService] ✓ Created .ready marker ({stats['total_chunks']} chunks)")
            else:
                print(f"[EmbeddingService] ✗ WARNING: No chunks indexed!")
                # Clean up empty index
                if persist_dir.exists():
                    shutil.rmtree(persist_dir)
                raise ValueError(f"Failed to index {repo_name}@{commit[:8]} - no chunks created")
            
            # Add to cache
            self.indices_cache[repo_commit_hash] = index
            print(f"[EmbeddingService] Loaded NEW index for {repo_name}@{commit[:8]} on {self.current_device}")
            return index

        # Load existing index (with .ready marker present)
        self._enforce_cache_limits()
        
        index = SemanticSearch(
            collection_name=f"code_{repo_commit_hash}",
            persist_directory=str(persist_dir),
            embedding_model_name=self.embedding_model_name,
            reranker_model_name=self.reranker_model_name,
            device=self.current_device,
            num_threads=4 if self.current_device == "cpu" else 1,
            read_only=True,  # CRITICAL: Read-only mode during training
        )
        
        index.embedder = self.embedder
        index.reranker = self.reranker
        
        self.indices_cache[repo_commit_hash] = index
        print(f"[EmbeddingService] Loaded index for {repo_name}@{commit[:8]} on {self.current_device} (cache: {len(self.indices_cache)}/{self.max_indices})")
        
        return index

    def search(
        self,
        query: str,
        repo_name: str,
        commit: str,
        n_results: int = 10,
    ) -> List[Dict]:
        """
        Perform semantic search using current device (CPU during training).
        """
        index = self.get_or_load_index(repo_name, commit)
        results = index.search(query, n_results=n_results, use_reranker=False)
        return results

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query on current device."""
        self.ensure_models_loaded()
        return self.embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True)

    def embed_batch(self, queries: List[str]) -> np.ndarray:
        """Embed multiple queries efficiently on current device."""
        self.ensure_models_loaded()
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
            "current_phase": self.current_phase,
            "current_device": self.current_device,
            "embedding_model": self.embedding_model_name,
            "reranker_model": self.reranker_model_name,
            "models_loaded": self.models_loaded,
        }

@ray.remote(num_cpus=2, num_gpus=0.25)
class EmbeddingWorker:
    """Parallel indexing worker that uses a fraction of GPU."""
    
    def __init__(self, worker_id: int, embedding_model: str, cache_dir: str, num_threads: int = 4):

        import torch
        from pathlib import Path
        from loguru import logger  # Import logger here
        import sys
        self.logger = logger.bind(worker_id=worker_id)
        # Configure logger for this worker
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>Worker-{extra[worker_id]}</cyan> | <level>{message}</level>",
            level="INFO"
        )
        logger.configure(extra={"worker_id": worker_id})
        self.worker_id = worker_id
        self.cache_dir = Path(cache_dir)
        
        # Check GPU availability
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            
            print(f"[Worker {worker_id}] 🔥 GPU AVAILABLE")
            print(f"[Worker {worker_id}]   - Device count: {gpu_count}")
            print(f"[Worker {worker_id}]   - GPU name: {gpu_name}")
            print(f"[Worker {worker_id}]   - CUDA version: {torch.version.cuda}")
        else:
            device = "cpu"
            print(f"[Worker {worker_id}] ⚠️  NO GPU - using CPU")
        
        print(f"[Worker {worker_id}] Loading embedder on {device}...")
        
        # Load embedder
        self.embedder = SentenceTransformer(
            embedding_model,
            device=device,
            model_kwargs={'torch_dtype': torch.float16} if device == "cuda" else {}
        )
        
        # Verify embedder device
        print(f"[Worker {worker_id}] Embedder model device: {self.embedder.device}")
        
        if device == "cuda":
            # Print GPU memory usage
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Worker {worker_id}] GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        else:
            torch.set_num_threads(num_threads)
        
        print(f"[Worker {worker_id}] ✓ Ready on {device}")
        self.logger.info(f"Initialized on {device}")
    
    def index_repo(self, repo_name: str, commit: str, repo_path: str) -> dict:
        """Index a single repository."""
        import shutil
        import torch
        from pathlib import Path
        from src.tools.semantic_search import SemanticSearch
        from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
        
        try:
            self.logger.info(f"🔨 Starting indexing: {repo_name}@{commit[:7]}")
            
            device_type = self.embedder.device.type
            print(f"[Worker {self.worker_id}]   Device: {device_type}")
            
            if device_type == "cuda":
                gpu_id = self.embedder.device.index or 0
                before_mem = torch.cuda.memory_allocated(gpu_id) / 1024**3
                print(f"[Worker {self.worker_id}]   GPU {gpu_id} memory before: {before_mem:.2f}GB")
            
            repo_commit_hash = get_repo_commit_hash(repo_name, commit)
            index_path = self.cache_dir / repo_commit_hash
            ready_file = index_path / ".ready"
            
            print(f"[Worker {self.worker_id}]   Repo hash: {repo_commit_hash}")
            print(f"[Worker {self.worker_id}]   Index path: {index_path}")
            print(f"[Worker {self.worker_id}]   Ready file: {ready_file}")
            
            # Check if already indexed
            if ready_file.exists():
                print(f"[Worker {self.worker_id}] ✓ Already indexed (cached)")
                return {
                    'success': True,
                    'repo_name': repo_name,
                    'commit': commit,
                    'cached': True,
                    'worker_id': self.worker_id,
                    'device': device_type
                }
            
            # Create index directory
            index_path.mkdir(parents=True, exist_ok=True)
            print(f"[Worker {self.worker_id}]   Created index directory")
            
            # Create SemanticSearch with pre-loaded embedder
            print(f"[Worker {self.worker_id}]   Creating SemanticSearch with shared embedder...")
            index = SemanticSearch(
                collection_name=f"code_{repo_commit_hash}",
                persist_directory=str(index_path),
                device=device_type,
                max_chunk_size=512,
                embedder=self.embedder,  # Pass pre-loaded embedder
                reranker=None,
            )
            
            print(f"[Worker {self.worker_id}]   Indexing files from: {repo_path}")
            
            import time
            start_time = time.time()
            
            stats = index.index_code_files(
                repo_path,
                file_extensions=[".py"],
                batch_size=128,
            )
            
            index_time = time.time() - start_time
            
            print(f"[Worker {self.worker_id}]   Indexing complete. Chunks: {stats['total_chunks']}")
            
            if stats["total_chunks"] == 0:
                print(f"[Worker {self.worker_id}]   ✗ No chunks created, cleaning up...")
                if index_path.exists():
                    shutil.rmtree(index_path, ignore_errors=True)
                raise ValueError("No chunks indexed")
            
            # ✅ CRITICAL: CREATE .ready MARKER
            print(f"[Worker {self.worker_id}]   Creating .ready marker...")
            print(f"[Worker {self.worker_id}]   Ready file path: {ready_file}")
            print(f"[Worker {self.worker_id}]   Ready file parent exists: {ready_file.parent.exists()}")
            self.logger.info("Creating .ready marker...")
            # Create the .ready file
            ready_file.touch()
            import os
            os.sync()  # Force kernel to flush buffers to disk
            time.sleep(0.5)
            
            # ✅ VERIFY IT WAS CREATED
            if ready_file.exists():
                print(f"[Worker {self.worker_id}]   ✓✓✓ .ready marker CREATED and VERIFIED")
                self.logger.info("✓✓✓ .ready marker CREATED and VERIFIED")
                # Double-check with explicit open/close to force flush
                with open(ready_file, 'a') as f:
                    f.flush()
                    os.fsync(f.fileno())  # Force flush to disk
            else:
                print(f"[Worker {self.worker_id}]   ✗✗✗ WARNING: .ready marker NOT created!")
                self.logger.warning("✗✗✗ WARNING: .ready marker NOT created!")
            if device_type == "cuda":
                after_mem = torch.cuda.memory_allocated(gpu_id) / 1024**3
                mem_used = after_mem - before_mem
                print(f"[Worker {self.worker_id}]   GPU {gpu_id} memory after: {after_mem:.2f}GB (delta: +{mem_used:.2f}GB)")
            
            chunks_per_sec = stats['total_chunks'] / index_time if index_time > 0 else 0
            print(f"[Worker {self.worker_id}] ✓ Indexed in {index_time:.1f}s ({stats['total_chunks']} chunks, {chunks_per_sec:.1f} chunks/sec)")
            
            return {
                'success': True,
                'repo_name': repo_name,
                'commit': commit,
                'cached': False,
                'chunks': stats['total_chunks'],
                'worker_id': self.worker_id,
                'device': device_type,
                'index_time': index_time,
                'chunks_per_sec': chunks_per_sec,
                'repo_hash': repo_commit_hash,  # Add this for debugging
            }
            
        except Exception as e:
            print(f"[Worker {self.worker_id}] ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up partial index
            if 'index_path' in locals() and index_path.exists():
                print(f"[Worker {self.worker_id}]   Cleaning up partial index: {index_path}")
                shutil.rmtree(index_path, ignore_errors=True)
            
            return {
                'success': False,
                'repo_name': repo_name,
                'commit': commit,
                'error': str(e),
                'worker_id': self.worker_id,
                'device': device_type if hasattr(self, 'embedder') else 'unknown'
            }
        
def get_embedding_service(
    max_indices: int = 20,
    max_cache_size_gb: Optional[float] = None,
) -> ray.ObjectRef:
    """
    Get or create the shared two-phase embedding service.

    Call this once at training start to initialize the service.
    All workers will share this single instance.

    Args:
        max_indices: Maximum number of indices to keep (LRU eviction)
        max_cache_size_gb: Maximum total disk space for cache (GB), None = unlimited
    """
    try:
        # Try to get existing service
        service = ray.get_actor("embedding_service")
        print("[EmbeddingService] Using existing service")
    except ValueError:
        # Create new service
        print(f"[EmbeddingService] Creating new two-phase service")
        service = EmbeddingService.options(name="embedding_service").remote(
            max_indices=max_indices,
            max_cache_size_gb=max_cache_size_gb,
        )
        print("[EmbeddingService] Service created")

    return service