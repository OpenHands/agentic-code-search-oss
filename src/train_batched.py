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
        self.current_batch_dataset = None
        self._batched_mode = cfg.get("batched_indexing", {}).get("enabled", False)
        self._shared_tracker = None
        super().__init__(cfg)
        
        # After parent init, load full dataset for batch manager
        if self._batched_mode and (not hasattr(self, 'train_ds') or self.train_ds is None):
            logger.info("[Init] Loading full dataset for batch manager...")
            self._load_dataset()
    def get_tracker(self):
        """
        Override to reuse tracker across batches in batched mode.
        
        In batched training, we create one tracker for all batches.
        Otherwise, use parent's behavior.
        """
        if self._batched_mode and hasattr(self, '_shared_tracker') and self._shared_tracker is not None:
            logger.info("[get_tracker] Reusing shared tracker for batched training")
            return self._shared_tracker
        else:
            # Normal behavior: create new tracker
            return super().get_tracker()
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        """Get the async trainer - CRITICAL for reward tracking!"""
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )
    
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

    def _setup_trainer(self):
        """Override to inject batch dataset before trainer initialization."""
        if self._batched_mode and self.current_batch_dataset is not None:
            from skyrl_train.dataset import PromptDataset
            from skyrl_train.utils.trainer_utils import build_dataloader
            
            batch_idx = getattr(self, '_current_batch_idx', 0)
            logger.info(f"[_setup_trainer] Injecting batch {batch_idx} dataset: {len(self.current_batch_dataset)} instances")
            
            # Validate dataset structure
            if len(self.current_batch_dataset) > 0:
                first_item = self.current_batch_dataset[0]
                if "prompt" not in first_item:
                    raise ValueError(
                        f"Batch dataset missing 'prompt' key. "
                        f"Available keys: {list(first_item.keys())}"
                    )
            
            # Create PromptDataset wrapper without file loading
            prompts_dataset = object.__new__(PromptDataset)
            prompts_dataset.tokenizer = self.tokenizer
            prompts_dataset.max_prompt_length = self.cfg.trainer.max_prompt_length
            prompts_dataset.prompt_key = "prompt"
            prompts_dataset.env_class_key = "env_class"
            prompts_dataset.num_workers = 8
            prompts_dataset.datasets = None
            prompts_dataset.dataframe = self.current_batch_dataset
            
            # Replace the dataset
            self.train_dataset = prompts_dataset
            
            logger.info(f"[_setup_trainer] ✓ Batch {batch_idx} dataset ready: {len(prompts_dataset)} instances")
            
            assert len(prompts_dataset) >= self.cfg.trainer.train_batch_size, \
                f"Batch too small: {len(prompts_dataset)} < {self.cfg.trainer.train_batch_size}"
        
        # Create the trainer (will use shared tracker via get_tracker() override)
        trainer = super()._setup_trainer()
        
        # ✅ Rebuild the dataloader with the new batch dataset
        if self._batched_mode and self.current_batch_dataset is not None:
            batch_idx = getattr(self, '_current_batch_idx', 0)
            
            logger.info(f"[_setup_trainer] Rebuilding dataloader for batch {batch_idx}")
            
            # Rebuild dataloader
            trainer.train_dataloader = build_dataloader(
                self.cfg, 
                trainer.train_dataset, 
                is_train=True, 
                is_fully_async=True
            )
            
            # Update async dataloader wrapper
            trainer.async_train_dataloader._train_dataloader = trainer.train_dataloader
            trainer.async_train_dataloader._train_dataloader_initial_state = trainer.train_dataloader.state_dict()
            trainer.async_train_dataloader._effective_dataloader_length = (
                len(trainer.train_dataloader) // trainer.mini_batch_size * trainer.mini_batch_size
            )
            trainer.async_train_dataloader._iter = enumerate(trainer.train_dataloader)
            
            # Update epoch/step calculations
            trainer.num_steps_per_epoch = len(trainer.train_dataloader) // trainer.mini_batch_size
            trainer.total_training_steps = trainer.num_steps_per_epoch * self.cfg.trainer.epochs
            
            # Enable single-epoch mode for batched training
            trainer.enable_single_epoch_mode()
            
            logger.info(f"[_setup_trainer] ✓ Dataloader rebuilt:")
            logger.info(f"  - Batches per epoch: {trainer.num_steps_per_epoch}")
            logger.info(f"  - Epochs: {self.cfg.trainer.epochs}")
            logger.info(f"  - Total training steps: {trainer.total_training_steps}")
            
            # ✅ Log batch info to WandB (without changing run name)
            if hasattr(trainer, 'tracker') and trainer.tracker is not None:
                trainer.tracker.log({'batch/current_batch': batch_idx}, step=trainer.global_step)
                logger.info(f"[_setup_trainer] ✓ Logged batch {batch_idx} to WandB")
        
        return trainer
    
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initialize CodeSearchGenerator with semantic search support."""
        semantic_search_cfg = cfg.get('semantic_search', OmegaConf.create({
            'enabled': True,
            'embedding_model': 'jinaai/jina-code-embeddings-0.5b',
            'reranker_model': None,
            'max_indices': 15
        }))
        generator = CodeSearchGenerator(
            model_name=cfg.trainer.policy.model.path,
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
            repo_field=batch_config.get("repo_field", "repo"),
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
            dir_name = f"{repo_name.replace('/', '__')}__{commit[:8]}"
            repo_path = repos_dir / dir_name
            
            # Check if already cloned
            if repo_path.exists() and (repo_path / ".git").exists():
                # Verify it's at the right commit
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
                        shutil.rmtree(repo_path)
                except Exception as e:
                    logger.warning(f"[Cloning] Could not verify commit for {repo_name}: {e}")
                    shutil.rmtree(repo_path)
            
            # Clone the repository
            try:
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
                
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
                    
            except subprocess.CalledProcessError as e:
                error_msg = f"Git error: {e.stderr if e.stderr else str(e)}"
                logger.error(f"[Cloning] ✗ {error_msg}")
                failed.append((repo_name, commit, error_msg))
                
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[Cloning] ✗ Unexpected error: {error_msg}")
                failed.append((repo_name, commit, error_msg))
                
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

    def run_batch(self, batch_idx: int):
        """Run training for a single batch of repositories."""
        from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[Batch {batch_idx}/{self.batch_manager.num_batches-1}] Starting...")
        logger.info(f"{'='*80}")
        
        batch_repos = self.batch_manager.get_batch_repos(batch_idx)
        logger.info(f"[Batch {batch_idx}] Repos in batch: {[f'{r}@{c[:7]}' for r, c in batch_repos]}")
        
        # ✅ CRITICAL: If this is not batch 0, force cleanup of previous training state
        if batch_idx > 0:
            logger.info(f"[Batch {batch_idx}] Releasing previous batch's training resources...")
            
            # Force garbage collection
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Give Ray time to cleanup
            import time
            time.sleep(5)
            
            # Log available resources
            try:
                available = ray.available_resources()
                logger.info(f"[Batch {batch_idx}] Available before indexing: GPU={available.get('GPU', 0):.1f}, CPU={available.get('CPU', 0):.1f}")
            except:
                pass
        
        # Phase 0: CLONING
        logger.info(f"[Batch {batch_idx}] Phase 0: Cloning repositories...")
        successful_clones, failed_clones = self.clone_batch_repos(batch_repos)
        
        if not successful_clones:
            logger.error(f"[Batch {batch_idx}] ✗ No repositories were successfully cloned!")
            raise RuntimeError(f"Batch {batch_idx} failed - no repos cloned")
        
        # Phase 1: INDEXING (GPU)
        logger.info(f"[Batch {batch_idx}] Phase 1: Indexing on GPU (parallel)...")
        
        from src.services.embedding_service import EmbeddingWorker
        
        num_workers = self.cfg.batched_indexing.get("num_index_workers", 4)
        semantic_config = self.cfg.get("semantic_search", {})
        
        workers = []
        results = []
        errors = []
        futures = []
        task_mapping = {}
        
        try:
            # ✅ Check GPU availability BEFORE creating workers
            available = ray.available_resources()
            available_gpus = available.get('GPU', 0)
            logger.info(f"[Batch {batch_idx}] Available GPUs before worker creation: {available_gpus}")
            
            if available_gpus < 0.5:  # Need at least 0.5 GPU total
                logger.error(f"[Batch {batch_idx}] ✗ Insufficient GPU resources: {available_gpus} GPUs available")
                logger.error(f"[Batch {batch_idx}] Waiting 30s for GPU cleanup...")
                import time
                time.sleep(30)
                
                available = ray.available_resources()
                available_gpus = available.get('GPU', 0)
                logger.info(f"[Batch {batch_idx}] GPUs after wait: {available_gpus}")
                
                if available_gpus < 0.5:
                    raise RuntimeError(f"Cannot create indexing workers - only {available_gpus} GPUs available")
            
            # Create workers
            logger.info(f"[Batch {batch_idx}] Creating {num_workers} GPU workers...")
            workers = [
                EmbeddingWorker.remote(
                    worker_id=i,
                    embedding_model=semantic_config.get("embedding_model", "jinaai/jina-code-embeddings-0.5b"),
                    cache_dir=str(self.cfg.batched_indexing.cache_dir),
                )
                for i in range(num_workers)
            ]
            
            logger.info(f"[Batch {batch_idx}] ✓ Created {num_workers} GPU workers")
            
            # Distribute tasks
            repos_dir = Path(self.cfg.batched_indexing.repos_dir)
            
            for i, (repo_name, commit) in enumerate(successful_clones):
                dir_name = f"{repo_name.replace('/', '__')}__{commit[:8]}"
                repo_path = repos_dir / dir_name
                
                if not repo_path.exists():
                    logger.warning(f"[Batch {batch_idx}] Repo not found: {repo_path}")
                    continue
                
                worker = workers[i % num_workers]
                future = worker.index_repo.remote(repo_name, commit, str(repo_path))
                futures.append(future)
                task_mapping[future] = (repo_name, commit)
            
            logger.info(f"[Batch {batch_idx}] Launched {len(futures)} indexing tasks")
            
            # Wait for completion with timeout
            from tqdm import tqdm
            import time
            
            indexing_start_time = time.time()
            max_indexing_time = 1800  # 30 min
            last_progress_time = time.time()
            last_progress_count = 0
            
            with tqdm(total=len(futures), desc="Indexing repos") as pbar:
                remaining_futures = list(futures)
                
                while remaining_futures:
                    elapsed = time.time() - indexing_start_time
                    if elapsed > max_indexing_time:
                        logger.error(f"[Batch {batch_idx}] ⏱️  TIMEOUT after {elapsed:.0f}s")
                        logger.error(f"[Batch {batch_idx}] Completed: {len(results)}/{len(futures)}")
                        
                        for future in remaining_futures:
                            try:
                                ray.cancel(future)
                            except:
                                pass
                            repo_name, commit = task_mapping[future]
                            errors.append((repo_name, commit, "Indexing timeout"))
                        
                        break
                    
                    done, remaining_futures = ray.wait(remaining_futures, num_returns=1, timeout=5.0)
                    
                    for done_future in done:
                        try:
                            result = ray.get(done_future, timeout=10.0)
                            results.append(result)
                            pbar.update(1)
                            
                            last_progress_time = time.time()
                            last_progress_count = len(results)
                            
                            repo_name, commit = task_mapping[done_future]
                            
                            if result.get('success', False):
                                logger.info(f"[Batch {batch_idx}] ✓ {repo_name}@{commit[:7]}")
                            else:
                                logger.error(f"[Batch {batch_idx}] ✗ {repo_name}@{commit[:7]}: {result.get('error', 'Unknown')}")
                                errors.append((repo_name, commit, result.get('error', 'Unknown')))
                                
                        except Exception as e:
                            repo_name, commit = task_mapping[done_future]
                            logger.error(f"[Batch {batch_idx}] ✗ {repo_name}@{commit[:7]}: {str(e)[:200]}")
                            errors.append((repo_name, commit, str(e)[:200]))
                            pbar.update(1)
                            last_progress_time = time.time()
                            last_progress_count = len(results)
            
        finally:
            # ✅ ALWAYS cleanup workers
            logger.info(f"[Batch {batch_idx}] Phase 2: Cleaning up {len(workers)} GPU workers...")
            
            for i, worker in enumerate(workers):
                try:
                    ray.kill(worker, no_restart=True)
                except Exception as e:
                    logger.warning(f"[Batch {batch_idx}] Could not kill worker {i}: {e}")
            
            # Force GPU cleanup
            import time
            import gc
            import torch
            
            logger.info(f"[Batch {batch_idx}] Forcing GPU memory release...")
            time.sleep(3)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(2)
            
            available = ray.available_resources()
            logger.info(f"[Batch {batch_idx}] ✓ After cleanup: GPU={available.get('GPU', 0):.1f}, CPU={available.get('CPU', 0):.1f}")
        
        # Process results
        successful_indices = [(r['repo_name'], r['commit']) for r in results if r.get('success', False)]
        
        logger.info(f"[Batch {batch_idx}] Indexing results: {len(successful_indices)}/{len(futures)} successful")
        
        if not successful_indices:
            logger.error(f"[Batch {batch_idx}] ✗ No indices created!")
            raise RuntimeError(f"Batch {batch_idx} failed - no indices created")
        
        # Verify .ready markers
        cache_dir = Path(self.cfg.batched_indexing.cache_dir)
        missing = []
        for repo_name, commit in successful_indices:
            repo_hash = get_repo_commit_hash(repo_name, commit)
            if not (cache_dir / repo_hash / ".ready").exists():
                missing.append((repo_name, commit))
        
        if missing:
            logger.warning(f"[Batch {batch_idx}] Missing {len(missing)} .ready markers, waiting 5s...")
            import time
            time.sleep(5)
            # Remove still-missing from successful_indices
            for repo_name, commit in missing:
                repo_hash = get_repo_commit_hash(repo_name, commit)
                if not (cache_dir / repo_hash / ".ready").exists():
                    successful_indices.remove((repo_name, commit))
        
        logger.info(f"[Batch {batch_idx}] ✓ Verified {len(successful_indices)} indices")
        
        # Phase 3: Setup retrieval
        logger.info(f"[Batch {batch_idx}] Phase 3: Setting up CPU retrieval...")
        
        try:
            embedding_service = ray.get_actor("embedding_service")
        except ValueError:
            from src.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService.options(
                name="embedding_service",
                num_cpus=4,
                num_gpus=0,
                lifetime="detached",
            ).remote(
                embedding_model=semantic_config.get("embedding_model"),
                cache_dir=str(self.cfg.batched_indexing.cache_dir),
                max_indices=semantic_config.get("max_indices", 15),
            )
        
        ray.get(embedding_service.enter_retrieval_phase.remote())
        logger.info(f"[Batch {batch_idx}] ✓ Retrieval ready on CPU")
        
        # Filter dataset
        batch_dataset = self.batch_manager.get_batch_dataset(batch_idx)
        
        all_failed = set(failed_clones) | set((r, c) for r, c, _ in errors)
        if all_failed:
            failed_hashes = {get_repo_commit_hash(r, c) for r, c in all_failed}
            batch_dataset = batch_dataset.filter(
                lambda x: get_repo_commit_hash(x[self.batch_manager.repo_field], x[self.batch_manager.commit_field]) not in failed_hashes
            )
        
        self.current_batch_dataset = batch_dataset
        logger.info(f"[Batch {batch_idx}] Training dataset: {len(batch_dataset)} instances")
        
        # Phase 4: Training
        logger.info(f"[Batch {batch_idx}] Phase 4: Training...")
        
        start_time = time.time()
        self._current_batch_idx = batch_idx
        trainer = self._setup_trainer()
        asyncio.run(trainer.train())
        
        train_time = time.time() - start_time
        logger.info(f"[Batch {batch_idx}] ✓ Training complete ({train_time:.0f}s)")
        
        self.current_batch_dataset = None
        
        # Phase 5: Cleanup
        if self.cfg.batched_indexing.get("cleanup_between_batches", True):
            logger.info(f"[Batch {batch_idx}] Phase 5: Cleanup...")
            
            successful_hashes = [get_repo_commit_hash(r, c) for r, c in successful_indices]
            try:
                ray.get(embedding_service.cleanup_batch_indices.remote(successful_hashes))
            except Exception as e:
                logger.warning(f"[Batch {batch_idx}] Cleanup warning: {e}")
            
            # Cleanup repos
            repos_dir = Path(self.cfg.batched_indexing.repos_dir)
            for repo_name, commit in successful_clones:
                repo_path = repos_dir / f"{repo_name.replace('/', '__')}__{commit[:8]}"
                if repo_path.exists():
                    try:
                        shutil.rmtree(repo_path)
                    except:
                        pass
        
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
            trainer = self._setup_trainer()
            asyncio.run(trainer.train())
            return
        
        logger.info("[Training] Using batched training mode")
        
        # Setup batched indexing (will load dataset if needed)
        self.setup_batched_indexing()
        
        from skyrl_train.utils.tracking import Tracking
        
        logger.info("[Training] Creating shared WandB tracker for all batches...")
        self._shared_tracker = Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=self.cfg.trainer.logger,
            config=self.cfg,
        )
        logger.info("[Training] ✓ Shared tracker created")
        
        # Run training in batches
        num_batches = self.batch_manager.num_batches
        logger.info(f"[Training] Starting training across {num_batches} batches")
        
        try:
            for batch_idx in range(num_batches):
                try:
                    self.run_batch(batch_idx)
                    
                    # Simple progress logging
                    completed = batch_idx + 1
                    progress_percent = (completed / num_batches) * 100
                    logger.info(f"[Progress] Completed {completed}/{num_batches} "
                               f"batches ({progress_percent:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"[Batch {batch_idx}] ✗ Failed with error: {e}")
                    logger.error("Full traceback:")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    if self.cfg.get("batched_indexing", {}).get("continue_on_batch_failure", False):
                        logger.warning(f"[Batch {batch_idx}] Continuing to next batch...")
                        continue
                    else:
                        logger.error("[Training] Aborting due to batch failure")
                        raise
            
            logger.info("[Training] ✓ All batches complete!")
            
        finally:
            # ✅ Always finish tracker, even if training fails
            if hasattr(self, '_shared_tracker') and self._shared_tracker is not None:
                try:
                    # Use the proper finish method from Track3ing class
                    if hasattr(self._shared_tracker, 'logger') and 'wandb' in self._shared_tracker.logger:
                        self._shared_tracker.logger['wandb'].finish(exit_code=0)
                    logger.info("[Training] ✓ WandB run finished")
                except Exception as e:
                    logger.warning(f"[Training] Error finishing tracker: {e}")
            
            # Cleanup embedding service actor
            try:
                embedding_service = ray.get_actor("embedding_service")
                ray.kill(embedding_service)
                logger.info("[Cleanup] ✓ EmbeddingService actor terminated")
            except ValueError:
                logger.warning("[Cleanup] embedding_service actor not found")
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

    # Setup rewards - CRITICAL for reward tracking!
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
            "excludes": excludes,
        }
    )

    # Sync registries
    sync_registries()
    
    # Run training
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()