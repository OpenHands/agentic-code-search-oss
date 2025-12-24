"""
Batched training script for code search with two-phase GPU/CPU indexing.

This script extends the base PPO training with support for batched repository indexing,
allowing training on large datasets without running out of GPU memory or disk space.

Key features:
- Two-phase indexing: GPU for fast embedding creation, CPU for retrieval during training
- Batch processing: Index N repos at a time, train, then cleanup
- Memory efficient: Embeddings and training never compete for GPU
- Automatic cleanup: Remove old indices to free disk space

Usage:
    python src/train_batched.py --config-name batched_indexing
    sbatch scripts/train_with_batched_indexing.sh
"""

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
import asyncio
import ray
import subprocess
from datasets import load_dataset, concatenate_datasets
import time
import shutil
from loguru import logger
from typing import List, Dict, Any, Optional

from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg

from src.services.batched_index_manager import BatchedIndexManager

# NOTE: These imports assume the following files exist in your codebase:
# - src/generator/code_search_generator.py (from your existing code)
# - src/async_trainer.py (from your existing code)
# If running this fails, ensure these files are present
try:
    from src.generator.code_search_generator import CodeSearchGenerator
    from src.async_trainer import CustomFullyAsyncRayPPOTrainer as FullyAsyncRayPPOTrainer
except ImportError as e:
    logger.error(f"Could not import required modules: {e}")
    logger.error("Please ensure src/generator/code_search_generator.py and src/async_trainer.py exist")
    raise


class BatchedCodeSearchPPOExp(BasePPOExp):
    """Extended experiment class with two-phase batched indexing."""
    
    def __init__(self, cfg: DictConfig):
        self.batch_manager: Optional[BatchedIndexManager] = None
        self.embedding_service = None
        super().__init__(cfg)
        
        # Ensure train_ds is loaded - if parent didn't load it, load it here
        if not hasattr(self, 'train_ds') or self.train_ds is None:
            logger.info("[Init] Loading training dataset...")
            self._load_dataset()
    
    def _load_dataset(self):
        """Load training dataset if not already loaded by parent."""
        if hasattr(self, 'train_ds') and self.train_ds is not None:
            logger.info(f"[Init] Dataset already loaded: {len(self.train_ds)} samples")
            return
        
        # Load from config
        train_data_paths = self.cfg.data.train_data
        logger.info(f"[Init] Loading dataset from: {train_data_paths}")
        
        datasets = []
        for path in train_data_paths:
            logger.info(f"[Init] Loading: {path}")
            ds = load_dataset("parquet", data_files=path, split="train")
            datasets.append(ds)
        
        if len(datasets) == 1:
            self.train_ds = datasets[0]
        else:
            self.train_ds = concatenate_datasets(datasets)
        
        logger.info(f"[Init] ✓ Loaded {len(self.train_ds)} training samples")
    
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initialize CodeSearchGenerator with semantic search support."""
        semantic_search_cfg = cfg.get('semantic_search', OmegaConf.create({
            'enabled': True,
            'embedding_model': 'jinaai/jina-code-embeddings-0.5b',
            'reranker_model': None,
            'max_indices': 15
        }))
        generator = CodeSearchGenerator(
            model_name=cfg.trainer.policy.model.path,  # Add model_name
            generator_cfg=cfg.generator,
            semantic_search_cfg=semantic_search_cfg,
            skyrl_gym_cfg=OmegaConf.create({"max_env_workers": 0}),
            tokenizer=tokenizer,
            inference_engine_client=inference_engine_client,
        )
        return generator
    
    def setup_batched_indexing(self):
        """
        Initialize batched indexing components.
        
        Creates the EmbeddingService actor that will be used throughout training.
        Must be called after dataset is loaded.
        """
        if not self.cfg.get("batched_indexing", {}).get("enabled", False):
            logger.info("[Batched] Batched indexing disabled, using regular training")
            return
        
        logger.info("[Batched] Setting up batched indexing...")
        
        # Ensure dataset is loaded
        if not hasattr(self, 'train_ds') or self.train_ds is None:
            logger.info("[Batched] Dataset not loaded yet, loading now...")
            self._load_dataset()
        
        # Verify train_ds exists
        if not hasattr(self, 'train_ds') or self.train_ds is None:
            raise AttributeError("train_ds could not be loaded. Check your data config.")
        
        logger.info(f"[Batched] Training dataset size: {len(self.train_ds)} samples")
        
        # 1. Initialize batch manager
        batch_config = self.cfg.batched_indexing
        
        self.batch_manager = BatchedIndexManager(
            dataset=self.train_ds,
            batch_size=batch_config.batch_size,
            cache_dir=batch_config.cache_dir,
            repo_field=batch_config.get("repo_field", "repo"),  # Changed from "repo_name" to "repo"
            commit_field=batch_config.get("commit_field", "base_commit"),
        )
        
        # 2. Initialize EmbeddingService actor (CRITICAL!)
        from src.services.embedding_service import EmbeddingService
        
        semantic_config = self.cfg.get("semantic_search", {})
        
        # Check if actor already exists (e.g., from previous run)
        try:
            existing_actor = ray.get_actor("embedding_service")
            logger.warning("[Batched] Found existing embedding_service actor, killing it...")
            ray.kill(existing_actor)
            time.sleep(2)  # Give it time to cleanup
        except ValueError:
            # Actor doesn't exist, which is expected
            pass
        
        # Create the actor with proper configuration
        logger.info("[Batched] Creating EmbeddingService actor...")
        self.embedding_service = EmbeddingService.options(
            name="embedding_service",  # Named actor for global access
            num_cpus=4,  # CPU cores for processing
            num_gpus=0,  # Don't reserve GPU - will borrow temporarily during indexing
            lifetime="detached",  # Keep alive across batches
            max_restarts=-1,  # Auto-restart on failure
            max_task_retries=-1  # Retry failed tasks
        ).remote(
            embedding_model=semantic_config.get("embedding_model", "jinaai/jina-code-embeddings-0.5b"),
            reranker_model=semantic_config.get("reranker_model"),
            cache_dir=batch_config.cache_dir,
            max_indices=semantic_config.get("max_indices", 15),
        )
        
        # Verify actor is accessible
        try:
            test_actor = ray.get_actor("embedding_service")
            logger.info("[Batched] ✓ EmbeddingService actor initialized and accessible")
        except ValueError as e:
            logger.error(f"[Batched] ✗ Failed to verify embedding_service actor: {e}")
            raise
        
        logger.info(f"[Batched] Total batches: {self.batch_manager.num_batches}")
        logger.info(f"[Batched] Batch size: {batch_config.batch_size} repos/batch")
        logger.info(f"[Batched] Cache dir: {batch_config.cache_dir}")
        logger.info(f"[Batched] Repos dir: {batch_config.repos_dir}")
    
    def clone_batch_repos(self, batch_repos: List[tuple]) -> tuple[List[tuple], List[tuple]]:
        """
        Clone repositories for a batch.
        
        Args:
            batch_repos: List of (repo_name, commit) tuples to clone
            
        Returns:
            (successful_clones, failed_clones) where each is a list of (repo_name, commit, [error])
        """
        logger.info(f"[Cloning] Starting clone for {len(batch_repos)} repositories...")
        
        repos_dir = Path(self.cfg.batched_indexing.repos_dir)
        repos_dir.mkdir(parents=True, exist_ok=True)
        
        successful = []
        failed = []
        
        for repo_name, commit in batch_repos:
            logger.info(f"[Cloning] Processing {repo_name}@{commit[:7]}")
            
            # Create directory name: owner__repo__{commit[:8]}
            # This ensures uniqueness per commit
            dir_name = f"{repo_name.replace('/', '__')}__{commit[:8]}"
            repo_path = repos_dir / dir_name
            
            # Check if already cloned
            if repo_path.exists() and (repo_path / ".git").exists():
                # Verify it's at the right commit
                import subprocess
                try:
                    result = subprocess.run(
                        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=10
                    )
                    actual_commit = result.stdout.strip()
                    
                    if actual_commit == commit:
                        logger.info(f"[Cloning] ✓ {repo_name}@{commit[:7]} already cloned")
                        successful.append((repo_name, commit))
                        continue
                    else:
                        logger.warning(f"[Cloning] Commit mismatch for {repo_name}, re-cloning")
                        logger.warning(f"[Cloning]   Expected: {commit[:7]}, Found: {actual_commit[:7]}")
                        shutil.rmtree(repo_path)
                except Exception as e:
                    logger.warning(f"[Cloning] Could not verify commit for {repo_name}: {e}")
                    shutil.rmtree(repo_path)
            
            # Clone the repository
            try:
                import subprocess
                
                logger.info(f"[Cloning] Cloning {repo_name} to {repo_path}")
                
                # Clone with depth 1 for speed
                subprocess.run(
                    [
                        "git", "clone",
                        "--quiet",
                        f"https://github.com/{repo_name}.git",
                        str(repo_path)
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                # Checkout the specific commit
                logger.info(f"[Cloning] Checking out commit {commit[:7]}")
                subprocess.run(
                    ["git", "-C", str(repo_path), "checkout", "--quiet", commit],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Verify checkout
                result = subprocess.run(
                    ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10
                )
                actual_commit = result.stdout.strip()
                
                if actual_commit != commit:
                    raise RuntimeError(
                        f"Checkout verification failed: expected {commit}, got {actual_commit}"
                    )
                
                logger.info(f"[Cloning] ✓ Successfully cloned {repo_name}@{commit[:7]}")
                successful.append((repo_name, commit))
                
            except subprocess.TimeoutExpired as e:
                error_msg = f"Timeout after {e.timeout}s"
                logger.error(f"[Cloning] ✗ {error_msg}")
                failed.append((repo_name, commit, error_msg))
                
                # Clean up partial clone
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
                    
            except subprocess.CalledProcessError as e:
                error_msg = f"Git error: {e.stderr if e.stderr else str(e)}"
                logger.error(f"[Cloning] ✗ {error_msg}")
                failed.append((repo_name, commit, error_msg))
                
                # Clean up partial clone
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[Cloning] ✗ Unexpected error: {error_msg}")
                failed.append((repo_name, commit, error_msg))
                
                # Clean up partial clone
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
        
        # Report results
        logger.info("\n" + "="*80)
        logger.info(f"[Cloning] Batch cloning complete")
        logger.info(f"  Successful: {len(successful)}/{len(batch_repos)}")
        logger.info(f"  Failed:     {len(failed)}/{len(batch_repos)}")
        logger.info("="*80)
        
        if failed:
            logger.warning(f"\n[Cloning] Failed to clone {len(failed)} repositories:")
            for repo_name, commit, error in failed:
                logger.warning(f"  ✗ {repo_name}@{commit[:7]}: {error}")
        
        return successful, failed

    def index_batch_repos_parallel(self, batch_repos: List[tuple], max_retries: int = 2, num_workers: int = 4):
        """
        Index a batch of repositories in parallel using Ray tasks.
        
        Args:
            batch_repos: List of (repo_name, commit) tuples to index
            max_retries: Number of times to retry failed indices
            num_workers: Number of parallel indexing workers
        """
        logger.info(f"[Indexing] Starting PARALLEL indexing for {len(batch_repos)} repositories with {num_workers} workers...")
        
        repos_dir = Path(self.cfg.batched_indexing.repos_dir)
        cache_dir = Path(self.cfg.batched_indexing.cache_dir)
        
        # Get embedding service actor
        try:
            embedding_service = ray.get_actor("embedding_service")
        except ValueError as e:
            logger.error("[Indexing] embedding_service actor not found!")
            raise
        
        # Enter indexing phase (models on GPU)
        logger.info("[Indexing] Entering GPU indexing phase...")
        ray.get(embedding_service.enter_indexing_phase.remote())
        
        # Define indexing task
        @ray.remote(num_cpus=2)
        def index_single_repo(repo_name: str, commit: str, repos_dir: str, cache_dir: str, max_retries: int):
            """Index a single repository with retry logic."""
            import time
            import shutil
            from pathlib import Path
            from loguru import logger
            
            from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
            
            repos_dir = Path(repos_dir)
            cache_dir = Path(cache_dir)
            
            repo_commit_hash = get_repo_commit_hash(repo_name, commit)
            
            # Find the cloned repo directory
            dir_name = f"{repo_name.replace('/', '__')}__{commit[:8]}"
            repo_path = repos_dir / dir_name
            
            if not repo_path.exists() or not repo_path.is_dir():
                return {
                    'success': False,
                    'repo_name': repo_name,
                    'commit': commit,
                    'error': "Repository not cloned"
                }
            
            # Check if already indexed
            index_path = cache_dir / repo_commit_hash
            ready_file = index_path / ".ready"
            
            if ready_file.exists():
                return {
                    'success': True,
                    'repo_name': repo_name,
                    'commit': commit,
                    'cached': True
                }
            
            # Try indexing with retries
            for attempt in range(max_retries):
                try:
                    # Get embedding service actor
                    embedding_service = ray.get_actor("embedding_service")
                    
                    # Call embedding service to index
                    ray.get(embedding_service.get_or_load_index.remote(
                        repo_name=repo_name,
                        commit=commit,
                        repo_path=str(repo_path)
                    ))
                    
                    # Verify .ready marker was created
                    if ready_file.exists():
                        return {
                            'success': True,
                            'repo_name': repo_name,
                            'commit': commit,
                            'cached': False,
                            'attempts': attempt + 1
                        }
                    else:
                        last_error = "Missing .ready marker"
                        
                except Exception as e:
                    last_error = str(e)
                    
                    # Clean up partial index before retry
                    if index_path.exists():
                        shutil.rmtree(index_path, ignore_errors=True)
                    
                    # Wait before retry
                    if attempt < max_retries - 1:
                        time.sleep(5)
            
            return {
                'success': False,
                'repo_name': repo_name,
                'commit': commit,
                'error': last_error,
                'attempts': max_retries
            }
        
        # Launch parallel indexing tasks
        logger.info(f"[Indexing] Launching {len(batch_repos)} parallel indexing tasks...")
        
        futures = []
        for repo_name, commit in batch_repos:
            future = index_single_repo.remote(
                repo_name, commit, str(repos_dir), str(cache_dir), max_retries
            )
            futures.append(future)
        
        # Wait for all tasks to complete with progress
        from tqdm import tqdm
        results = []
        
        with tqdm(total=len(futures), desc="Indexing repos") as pbar:
            while futures:
                # Wait for next task to complete
                done, futures = ray.wait(futures, num_returns=1, timeout=1.0)
                
                for done_future in done:
                    result = ray.get(done_future)
                    results.append(result)
                    pbar.update(1)
                    
                    if result['success']:
                        cached = result.get('cached', False)
                        status = "cached" if cached else "indexed"
                        logger.info(f"[Indexing] ✓ {result['repo_name']}@{result['commit'][:7]} ({status})")
                    else:
                        logger.error(f"[Indexing] ✗ {result['repo_name']}@{result['commit'][:7]}: {result['error']}")
        
        # Separate successful and failed
        successful = [(r['repo_name'], r['commit']) for r in results if r['success']]
        failed = [(r['repo_name'], r['commit'], r['error']) for r in results if not r['success']]
        
        # Get indexing stats
        try:
            stats = ray.get(embedding_service.get_cache_stats.remote())
            logger.info(f"[Indexing] Cache stats: {stats}")
        except Exception as e:
            logger.warning(f"[Indexing] Could not get cache stats: {e}")
            stats = {"loaded_indices": "unknown"}
        
        # Report results
        logger.info("\n" + "="*80)
        logger.info(f"[Indexing] Parallel batch indexing complete")
        logger.info(f"  Successful: {len(successful)}/{len(batch_repos)}")
        logger.info(f"  Failed:     {len(failed)}/{len(batch_repos)}")
        if isinstance(stats.get('loaded_indices'), int):
            logger.info(f"  Active indices: {stats['loaded_indices']}")
        logger.info("="*80)
        
        if failed:
            logger.warning(f"\n[Indexing] Failed to index {len(failed)} repositories:")
            for repo_name, commit, error in failed:
                logger.warning(f"  ✗ {repo_name}@{commit[:7]}: {error}")
        
        return successful, failed
    def index_batch_repos(self, batch_repos: List[tuple], max_retries: int = 2):
        """
        Index a batch of repositories using GPU with retry logic.
        
        Args:
            batch_repos: List of (repo_name, commit) tuples to index
            max_retries: Number of times to retry failed indices
        """
        logger.info(f"[Indexing] Starting indexing for {len(batch_repos)} repositories...")
        
        repos_dir = Path(self.cfg.batched_indexing.repos_dir)
        cache_dir = Path(self.cfg.batched_indexing.cache_dir)
        
        # Get embedding service actor
        try:
            embedding_service = ray.get_actor("embedding_service")
        except ValueError as e:
            logger.error("[Indexing] embedding_service actor not found!")
            raise
        
        # Enter indexing phase (models on GPU)
        logger.info("[Indexing] Entering GPU indexing phase...")
        ray.get(embedding_service.enter_indexing_phase.remote())
        
        # Track success/failure
        successful = []
        failed = []
        
        # Index each repo with retry logic
        for repo_name, commit in batch_repos:
            logger.info(f"[Indexing] Processing {repo_name}@{commit[:7]}")
            
            from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
            repo_commit_hash = get_repo_commit_hash(repo_name, commit)
            
            # Find the cloned repo directory using the naming convention
            # Format: owner__repo__{commit[:8]}
            dir_name = f"{repo_name.replace('/', '__')}__{commit[:8]}"
            repo_path = repos_dir / dir_name
            
            if not repo_path.exists() or not repo_path.is_dir():
                logger.error(f"[Indexing] Repository not found at: {repo_path}")
                failed.append((repo_name, commit, "Repository not cloned"))
                continue
            
            logger.info(f"[Indexing] Found repo at: {repo_path}")
            
            # Check if already indexed with .ready marker
            index_path = cache_dir / repo_commit_hash
            ready_file = index_path / ".ready"
            
            if ready_file.exists():
                logger.info(f"[Indexing] ✓ {repo_name}@{commit[:7]} already indexed")
                successful.append((repo_name, commit))
                continue
            
            # Try indexing with retries
            indexed = False
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"[Indexing] Attempt {attempt + 1}/{max_retries} for {repo_name}@{commit[:7]}")
                    logger.info(f"[Indexing]   Repo path: {repo_path}")
                    logger.info(f"[Indexing]   Index path: {index_path}")
                    
                    # Call embedding service to index - PASS REPO_PATH
                    ray.get(embedding_service.get_or_load_index.remote(
                        repo_name=repo_name,
                        commit=commit,
                        repo_path=str(repo_path)  # CRITICAL: Pass the repo path
                    ))
                    
                    # Verify .ready marker was created
                    if ready_file.exists():
                        logger.info(f"[Indexing] ✓ Successfully indexed {repo_name}@{commit[:7]}")
                        indexed = True
                        successful.append((repo_name, commit))
                        break
                    else:
                        logger.warning(f"[Indexing] Index created but .ready marker missing for {repo_name}@{commit[:7]}")
                        last_error = "Missing .ready marker"
                        
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"[Indexing] ✗ Attempt {attempt + 1} failed for {repo_name}@{commit[:7]}")
                    logger.error(f"[Indexing] Error: {e}")
                    import traceback
                    logger.error(f"[Indexing] Traceback:\n{traceback.format_exc()}")
                    
                    # Clean up partial index before retry
                    if index_path.exists():
                        logger.info(f"[Indexing] Cleaning up partial index at {index_path}")
                        shutil.rmtree(index_path, ignore_errors=True)
                    
                    # Wait before retry
                    if attempt < max_retries - 1:
                        import time
                        logger.info(f"[Indexing] Waiting 5 seconds before retry...")
                        time.sleep(5)
            
            if not indexed:
                logger.error(f"[Indexing] ✗ Failed to index {repo_name}@{commit[:7]} after {max_retries} attempts")
                logger.error(f"[Indexing] Last error: {last_error}")
                failed.append((repo_name, commit, last_error))
        
        # Get indexing stats
        try:
            stats = ray.get(embedding_service.get_cache_stats.remote())
            logger.info(f"[Indexing] Cache stats: {stats}")
        except Exception as e:
            logger.warning(f"[Indexing] Could not get cache stats: {e}")
            stats = {"loaded_indices": "unknown"}
        
        # Report results
        logger.info("\n" + "="*80)
        logger.info(f"[Indexing] Batch indexing complete")
        logger.info(f"  Successful: {len(successful)}/{len(batch_repos)}")
        logger.info(f"  Failed:     {len(failed)}/{len(batch_repos)}")
        if isinstance(stats.get('loaded_indices'), int):
            logger.info(f"  Active indices: {stats['loaded_indices']}")
        logger.info("="*80)
        
        if successful:
            logger.info(f"\n[Indexing] Successfully indexed:")
            for repo_name, commit in successful:
                logger.info(f"  ✓ {repo_name}@{commit[:7]}")
        
        if failed:
            logger.warning(f"\n[Indexing] Failed to index {len(failed)} repositories:")
            for repo_name, commit, error in failed:
                logger.warning(f"  ✗ {repo_name}@{commit[:7]}: {error}")
            
            # Decide what to do with failures
            if self.cfg.batched_indexing.get("skip_failed_indices", True):
                if successful:
                    logger.warning(f"[Indexing] Continuing with {len(successful)} successfully indexed repos")
                    logger.warning(f"[Indexing] Episodes using failed repos will be filtered out")
                else:
                    logger.error(f"[Indexing] No repositories were successfully indexed!")
            else:
                raise RuntimeError(
                    f"Failed to index {len(failed)} repositories. "
                    f"Set batched_indexing.skip_failed_indices=true to continue anyway."
                )
        
        return successful, failed


    def run_batch(self, batch_idx: int):
        """Run training for a single batch of repositories."""
        logger.info(f"\n{'='*80}")
        logger.info(f"[Batch {batch_idx}/{self.batch_manager.num_batches-1}] Starting...")
        logger.info(f"{'='*80}")
        from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
        batch_repos = self.batch_manager.get_batch_repos(batch_idx)
        logger.info(f"[Batch {batch_idx}] Repos in batch: {[f'{r}@{c[:7]}' for r, c in batch_repos]}")
        
        # Phase 0: CLONING
        logger.info(f"[Batch {batch_idx}] Phase 0: Cloning repositories...")
        successful_clones, failed_clones = self.clone_batch_repos(batch_repos)
        
        if not successful_clones:
            logger.error(f"[Batch {batch_idx}] ✗ No repositories were successfully cloned!")
            if failed_clones:
                logger.error(f"[Batch {batch_idx}] Clone failures:")
                for repo_name, commit, error in failed_clones:
                    logger.error(f"  - {repo_name}@{commit[:7]}: {error}")
            raise RuntimeError(f"Batch {batch_idx} failed - no repos cloned")
        
        repos_to_index = successful_clones
        
        # Phase 1: INDEXING (GPU) - Use parallel GPU workers
        logger.info(f"[Batch {batch_idx}] Phase 1: Indexing on GPU (parallel)...")
        logger.info(f"[Batch {batch_idx}] Creating GPU indexing workers...")
        
        from src.services.embedding_service import EmbeddingWorker
        
        # Create GPU workers for parallel indexing
        num_workers = self.cfg.batched_indexing.get("num_index_workers", 4)
        semantic_config = self.cfg.get("semantic_search", {})
        
        try:
            workers = [
                EmbeddingWorker.remote(
                    worker_id=i,
                    embedding_model=semantic_config.get("embedding_model", "jinaai/jina-code-embeddings-0.5b"),
                    cache_dir=str(self.cfg.batched_indexing.cache_dir),
                )
                for i in range(num_workers)
            ]
            
            logger.info(f"[Batch {batch_idx}] Created {num_workers} GPU workers")
            
        except Exception as e:
            logger.error(f"[Batch {batch_idx}] Failed to create workers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Distribute indexing tasks across workers
        repos_dir = Path(self.cfg.batched_indexing.repos_dir)
        futures = []
        task_mapping = {}  # Track which future corresponds to which repo
        
        for i, (repo_name, commit) in enumerate(repos_to_index):
            # Find repo path
            dir_name = f"{repo_name.replace('/', '__')}__{commit[:8]}"
            repo_path = repos_dir / dir_name
            
            if not repo_path.exists():
                logger.warning(f"[Batch {batch_idx}] Repo not found: {repo_path}")
                continue
            
            # Assign to worker (round-robin)
            worker = workers[i % num_workers]
            future = worker.index_repo.remote(repo_name, commit, str(repo_path))
            futures.append(future)
            task_mapping[future] = (repo_name, commit)
        
        logger.info(f"[Batch {batch_idx}] Launched {len(futures)} indexing tasks")
        
        if len(futures) == 0:
            logger.error(f"[Batch {batch_idx}] No indexing tasks were launched!")
            logger.error(f"[Batch {batch_idx}] Check that repos were cloned to correct locations")
            raise RuntimeError(f"Batch {batch_idx} failed - no indexing tasks")
        
        # Wait for completion with progress and detailed error reporting
        from tqdm import tqdm
        results = []
        errors = []
        
        with tqdm(total=len(futures), desc="Indexing repos") as pbar:
            remaining_futures = list(futures)
            
            while remaining_futures:
                done, remaining_futures = ray.wait(remaining_futures, num_returns=1, timeout=5.0)
                
                for done_future in done:
                    try:
                        result = ray.get(done_future)
                        results.append(result)
                        pbar.update(1)
                        
                        repo_name, commit = task_mapping[done_future]
                        
                        if result['success']:
                            logger.info(f"[Batch {batch_idx}] ✓ {repo_name}@{commit[:7]} - {result.get('chunks', 0)} chunks")
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            logger.error(f"[Batch {batch_idx}] ✗ {repo_name}@{commit[:7]} - {error_msg}")
                            errors.append((repo_name, commit, error_msg))
                            
                    except Exception as e:
                        repo_name, commit = task_mapping[done_future]
                        logger.error(f"[Batch {batch_idx}] ✗ {repo_name}@{commit[:7]} - Exception: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        errors.append((repo_name, commit, str(e)))
                        pbar.update(1)
        
        # Process results
        successful_indices = [(r['repo_name'], r['commit']) for r in results if r.get('success', False)]
        failed_indices = [(r['repo_name'], r['commit'], r.get('error', 'Unknown')) for r in results if not r.get('success', False)]
        
        logger.info(f"\n[Batch {batch_idx}] Indexing results:")
        logger.info(f"  Tasks launched: {len(futures)}")
        logger.info(f"  Results received: {len(results)}")
        logger.info(f"  Successful: {len(successful_indices)}")
        logger.info(f"  Failed: {len(failed_indices)}")
        
        # Detailed error reporting
        if errors:
            logger.error(f"\n[Batch {batch_idx}] Indexing errors:")
            for repo_name, commit, error in errors:
                logger.error(f"  - {repo_name}@{commit[:7]}: {error}")
        
        if failed_indices:
            logger.warning(f"\n[Batch {batch_idx}] Failed indices:")
            for repo_name, commit, error in failed_indices:
                logger.warning(f"  - {repo_name}@{commit[:7]}: {error}")
        
        # Phase 2: CLEANUP INDEXING WORKERS (Free GPU!)
        logger.info(f"[Batch {batch_idx}] Phase 2: Cleaning up GPU indexing workers...")
        
        for worker in workers:
            try:
                ray.kill(worker)
            except Exception as e:
                logger.warning(f"[Batch {batch_idx}] Could not kill worker: {e}")
        
        # Force cleanup
        import time
        time.sleep(2)
        
        logger.info(f"[Batch {batch_idx}] ✓ GPU indexing workers terminated, GPU freed")
        
        if not successful_indices:
            logger.error(f"[Batch {batch_idx}] ✗ No repositories were successfully indexed!")
            logger.error(f"[Batch {batch_idx}] Total repos to index: {len(repos_to_index)}")
            logger.error(f"[Batch {batch_idx}] Tasks launched: {len(futures)}")
            logger.error(f"[Batch {batch_idx}] Check logs above for specific errors")
            raise RuntimeError(f"Batch {batch_idx} failed - no indices created")
        
        # Phase 3: RETRIEVAL SETUP (CPU)
        logger.info(f"[Batch {batch_idx}] Phase 3: Setting up CPU retrieval...")
        
        # NOW create/get the embedding service actor for retrieval
        # This will be on CPU
        try:
            embedding_service = ray.get_actor("embedding_service")
        except ValueError:
            # Create it if it doesn't exist
            from src.services.embedding_service import EmbeddingService
            
            logger.info(f"[Batch {batch_idx}] Creating EmbeddingService actor for retrieval...")
            embedding_service = EmbeddingService.options(
                name="embedding_service",
                num_cpus=4,
                num_gpus=0,  # NO GPU for retrieval
                lifetime="detached",
            ).remote(
                embedding_model=semantic_config.get("embedding_model", "jinaai/jina-code-embeddings-0.5b"),
                reranker_model=semantic_config.get("reranker_model"),
                cache_dir=str(self.cfg.batched_indexing.cache_dir),
                max_indices=semantic_config.get("max_indices", 15),
            )
        
        # Enter retrieval phase (CPU)
        ray.get(embedding_service.enter_retrieval_phase.remote())
        
        stats = ray.get(embedding_service.get_cache_stats.remote())
        logger.info(f"[Batch {batch_idx}] ✓ Retrieval ready on CPU")
        logger.info(f"[Batch {batch_idx}] Current phase: {stats['current_phase']}, device: {stats['current_device']}")
        
        batch_dataset = self.batch_manager.get_batch_dataset(batch_idx)

        logger.info(f"[Batch {batch_idx}] Original batch dataset size: {len(batch_dataset)}")
        logger.info(f"[Batch {batch_idx}] Sample repos in batch dataset:")
        for i, instance in enumerate(batch_dataset.select(range(min(3, len(batch_dataset))))):
            repo_name = instance[self.batch_manager.repo_field]
            commit = instance[self.batch_manager.commit_field]
            repo_hash = get_repo_commit_hash(repo_name, commit)
            logger.info(f"  [{i}] {repo_name}@{commit[:7]} -> hash: {repo_hash}")

        # Combine all failed repos
        all_failed_repos = set()
        for repo_name, commit, _ in failed_clones + failed_indices:
            all_failed_repos.add((repo_name, commit))

        logger.info(f"[Batch {batch_idx}] Failed repos to filter out: {len(all_failed_repos)}")

        if all_failed_repos:
            from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
            failed_hashes = {
                get_repo_commit_hash(repo_name, commit) 
                for repo_name, commit in all_failed_repos
            }
            
            logger.info(f"[Batch {batch_idx}] Failed hashes: {failed_hashes}")
            
            def keep_instance(instance):
                instance_hash = get_repo_commit_hash(
                    instance[self.batch_manager.repo_field],
                    instance[self.batch_manager.commit_field]
                )
                keep = instance_hash not in failed_hashes
                if not keep:
                    logger.debug(f"[Batch {batch_idx}] Filtering out {instance[self.batch_manager.repo_field]}@{instance[self.batch_manager.commit_field][:7]} (hash: {instance_hash})")
                return keep
            
            original_size = len(batch_dataset)
            batch_dataset = batch_dataset.filter(keep_instance)
            filtered_size = len(batch_dataset)
            
            logger.warning(f"[Batch {batch_idx}] Filtered dataset: {original_size} -> {filtered_size} instances")
            
            # Log successful hashes
            logger.info(f"[Batch {batch_idx}] Successfully indexed repos:")
            for repo_name, commit in successful_indices:
                repo_hash = get_repo_commit_hash(repo_name, commit)
                logger.info(f"  ✓ {repo_name}@{commit[:7]} -> hash: {repo_hash}")
            
            # Verify all instances in filtered dataset have successful indices
            logger.info(f"[Batch {batch_idx}] Verifying filtered dataset...")
            successful_hashes = {
                get_repo_commit_hash(repo_name, commit)
                for repo_name, commit in successful_indices
            }
            
            for i, instance in enumerate(batch_dataset):
                instance_hash = get_repo_commit_hash(
                    instance[self.batch_manager.repo_field],
                    instance[self.batch_manager.commit_field]
                )
                if instance_hash not in successful_hashes:
                    logger.error(f"[Batch {batch_idx}] ✗ Instance {i} has hash {instance_hash} which is NOT in successful indices!")
                else:
                    logger.debug(f"[Batch {batch_idx}] ✓ Instance {i} has hash {instance_hash} which IS in successful indices")

        self.train_ds = batch_dataset
        logger.info(f"[Batch {batch_idx}] Final training dataset size: {len(batch_dataset)} instances")

        # Phase 4: TRAINING (GPU for training, CPU for retrieval)
        logger.info(f"[Batch {batch_idx}] Phase 4: Training (GPU) with CPU retrieval...")
        logger.info(f"[Batch {batch_idx}] GPU is 100% available for training")
        
        start_time = time.time()
        
        super(BatchedCodeSearchPPOExp, self).run()
        
        train_time = time.time() - start_time
        logger.info(f"[Batch {batch_idx}] Training completed in {train_time:.2f}s")
        
        # Phase 5: CLEANUP
        if self.cfg.batched_indexing.get("cleanup_between_batches", True):
            logger.info(f"[Batch {batch_idx}] Phase 5: Cleaning up batch...")
            
            
            # Cleanup indices
            successful_hashes = [
                get_repo_commit_hash(repo_name, commit)
                for repo_name, commit in successful_indices
            ]
            
            ray.get(embedding_service.cleanup_batch_indices.remote(successful_hashes))
            
            # Cleanup cloned repos
            repos_dir = Path(self.cfg.batched_indexing.repos_dir)
            cleaned_repos = 0
            for repo_name, commit in successful_clones:
                dir_name = f"{repo_name.replace('/', '__')}__{commit[:8]}"
                repo_path = repos_dir / dir_name
                
                if repo_path.exists():
                    try:
                        shutil.rmtree(repo_path)
                        cleaned_repos += 1
                    except Exception as e:
                        logger.warning(f"Could not remove {repo_path}: {e}")
            
            logger.info(f"[Batch {batch_idx}] Cleaned up {len(successful_hashes)} indices, {cleaned_repos} repos")
        
        logger.info(f"[Batch {batch_idx}] ✓ Complete")

    def run(self):
        """
        Main training loop with batched indexing support.
        
        If batched_indexing.enabled=false, falls back to regular training.
        Otherwise, runs training in batches with GPU/CPU phase separation.
        """
        # Check if batched indexing is enabled
        if not self.cfg.get("batched_indexing", {}).get("enabled", False):
            logger.info("[Training] Using regular training (batched indexing disabled)")
            # Use parent's run method
            super().run()
            return
        
        logger.info("[Training] Using batched training mode")
        
        # Setup batched indexing (will load dataset if needed)
        self.setup_batched_indexing()
        
        # Run training in batches
        num_batches = self.batch_manager.num_batches
        logger.info(f"[Training] Starting training across {num_batches} batches")
        
        for batch_idx in range(num_batches):
            try:
                self.run_batch(batch_idx)
                
                # Log progress
                progress = self.batch_manager.get_progress()
                logger.info(f"[Progress] Completed {progress['completed_batches']}/{progress['total_batches']} "
                           f"batches ({progress['progress_percent']:.1f}%)")
                
            except Exception as e:
                logger.error(f"[Batch {batch_idx}] ✗ Failed with error: {e}")
                logger.exception("Full traceback:")
                
                # Decide whether to continue or abort
                if self.cfg.get("batched_indexing", {}).get("continue_on_batch_failure", False):
                    logger.warning(f"[Batch {batch_idx}] Continuing to next batch...")
                    continue
                else:
                    logger.error("[Training] Aborting due to batch failure")
                    raise
        
        logger.info("[Training] ✓ All batches complete")
        
        # Cleanup embedding service actor
        try:
            embedding_service = ray.get_actor("embedding_service")
            ray.kill(embedding_service)
            logger.info("[Cleanup] ✓ EmbeddingService actor terminated")
        except ValueError:
            logger.warning("[Cleanup] embedding_service actor not found (may have already been killed)")
        except Exception as e:
            logger.warning(f"[Cleanup] Could not terminate embedding_service: {e}")


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    """Ray remote entry point for training."""
    exp = BatchedCodeSearchPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point with two-phase batched indexing."""
    # Validate configuration
    validate_cfg(cfg)

    # Setup rewards
    if hasattr(cfg.generator, "reward"):
        with open(cfg.generator.reward, "r") as f:
            reward_cfg = OmegaConf.load(f)
        cfg.generator.reward = reward_cfg.reward
    else:
        with open_dict(cfg):
            cfg.generator.reward = [
                {"fn": "multilevel_localization_f1_reward"},
            ]

    # Setup two-phase semantic search
    semantic_search_cfg = cfg.get('semantic_search', OmegaConf.create({'enabled': False}))
    
    if semantic_search_cfg.enabled:
        logger.info("\n" + "="*80)
        logger.info("Initializing Two-Phase Semantic Search")
        logger.info("="*80)
        
        # Check batched indexing configuration
        batched_config = cfg.get('batched_indexing', {})
        if batched_config.get('enabled', False):
            cache_dir = Path(batched_config.cache_dir)
            logger.info(f"[TwoPhase] Batched indexing enabled (batch_size={batched_config.batch_size})")
            logger.info(f"[TwoPhase] Cache directory: {cache_dir}")
        else:
            cache_dir = Path("/data/user_data/sanidhyv/.cache/swebench_indices")
        
        # Check for existing indices
        if cache_dir.exists():
            num_indices = len(list(cache_dir.iterdir()))
            if num_indices > 0:
                logger.info(f"[TwoPhase] Found {num_indices} existing indices")
        
        logger.info(f"[TwoPhase] Two-phase architecture ready")
        logger.info("="*80 + "\n")
    
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    
    from skyrl_train.utils import prepare_runtime_environment
    from skyrl_train.utils.ppo_utils import sync_registries

    # Prepare environment variables
    env_vars = prepare_runtime_environment(cfg)

    # Define exclusions for Ray runtime environment
    excludes = [
        # Checkpoints and models
        "ckpts/",
        "*.ckpt",
        "*.pth",
        "*.pt",
        "*.safetensors",
        "*.bin",
        
        
        # Logs
        "logs/",
        "*.log",
        "*.out",
        "*.err",
        
        # Caches and temp
        ".cache/",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        ".venv/",
        "venv/",
        "env/",
        "ray_temp*/",
        "ray_spill/",
        
        # Trajectories
        "trajectories/",
        
        # Git
        ".git/",
        
        # Hydra outputs
        "outputs/",
        "multirun/",
    ]

    # Initialize Ray with runtime environment
    ray.init(
        runtime_env={
            "env_vars": env_vars,
            "excludes": excludes,  # Add this
        }
    )

    # Sync registries
    sync_registries()
    
    # Run training
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()