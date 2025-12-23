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

    def index_batch_repos(self, batch_repos: List[tuple]):
        """
        Index a batch of repositories using GPU.
        
        Args:
            batch_repos: List of (repo_name, commit) tuples to index
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
        
        # Index each repo
        for repo_name, commit in batch_repos:
            logger.info(f"[Indexing] Processing {repo_name}@{commit}")
            
            # Import here to avoid circular dependency
            from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
            
            repo_commit_hash = get_repo_commit_hash(repo_name, commit)
            repo_path = repos_dir / repo_name.replace("/", "__")
            
            if not repo_path.exists():
                logger.warning(f"[Indexing] Repository not found: {repo_path}")
                logger.info(f"[Indexing] Skipping {repo_name} (not cloned)")
                continue
            
            # Index using embedding service
            try:
                ray.get(embedding_service.get_or_load_index.remote(
                    repo_name=repo_name,
                    commit=commit,
                    repo_path=str(repo_path)
                ))
                logger.info(f"[Indexing] ✓ Indexed {repo_name}@{commit}")
            except Exception as e:
                logger.error(f"[Indexing] ✗ Failed to index {repo_name}: {e}")
                # Don't raise - continue with other repos
                continue
        
        # Get indexing stats
        stats = ray.get(embedding_service.get_cache_stats.remote())
        logger.info(f"[Indexing] ✓ Completed. Active indices: {stats['loaded_indices']}")

    def run_batch(self, batch_idx: int):
        """
        Run training for a single batch of repositories.
        
        Phases:
        1. INDEXING: Use GPU to create embeddings for batch repos
        2. OFFLOAD: Move embeddings to CPU, free GPU memory
        3. TRAINING: GPU for training, CPU for retrieval
        4. CLEANUP: Remove batch indices to free disk space
        
        Args:
            batch_idx: Index of the batch to process
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"[Batch {batch_idx}/{self.batch_manager.num_batches-1}] Starting...")
        logger.info(f"{'='*80}")
        
        # Get the embedding service actor (should already exist from setup)
        try:
            embedding_service = ray.get_actor("embedding_service")
        except ValueError as e:
            logger.error(f"[Batch {batch_idx}] ✗ embedding_service actor not found!")
            logger.error("Did you forget to call setup_batched_indexing()?")
            raise
        
        # Phase 1: INDEXING (GPU)
        logger.info(f"[Batch {batch_idx}] Phase 1: Indexing repositories on GPU...")
        
        batch_repos = self.batch_manager.get_batch_repos(batch_idx)
        logger.info(f"[Batch {batch_idx}] Repos to index: {[f'{r}@{c[:7]}' for r, c in batch_repos]}")
        
        self.index_batch_repos(batch_repos)
        
        # Phase 2: OFFLOAD (GPU -> CPU)
        logger.info(f"[Batch {batch_idx}] Phase 2: Offloading embeddings to CPU...")
        ray.get(embedding_service.offload_for_training.remote())
        
        # Get stats after offloading
        stats = ray.get(embedding_service.get_cache_stats.remote())
        logger.info(f"[Batch {batch_idx}] ✓ Models moved to CPU")
        logger.info(f"[Batch {batch_idx}] Active indices: {stats['loaded_indices']}")
        
        # Update dataset for this batch
        batch_dataset = self.batch_manager.get_batch_dataset(batch_idx)
        self.train_ds = batch_dataset
        logger.info(f"[Batch {batch_idx}] Training dataset size: {len(batch_dataset)} instances")
        
        # Phase 3: TRAINING (GPU) + RETRIEVAL (CPU)
        logger.info(f"[Batch {batch_idx}] Phase 3: Training on GPU (retrieval on CPU)...")
        start_time = time.time()
        
        # Use parent's run method which sets up and runs the trainer
        # The parent class (BasePPOExp) has the trainer setup logic
        super(BatchedCodeSearchPPOExp, self).run()
        
        train_time = time.time() - start_time
        logger.info(f"[Batch {batch_idx}] Training completed in {train_time:.2f}s")
        
        # Phase 4: CLEANUP
        if self.cfg.batched_indexing.get("cleanup_between_batches", True):
            logger.info(f"[Batch {batch_idx}] Phase 4: Cleaning up batch indices...")
            
            # Get repo commit hashes to cleanup
            from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
            required_hashes = [
                get_repo_commit_hash(repo_name, commit)
                for repo_name, commit in batch_repos
            ]
            
            cleanup_stats = ray.get(
                embedding_service.cleanup_batch_indices.remote(required_hashes)
            )
            logger.info(f"[Batch {batch_idx}] Removed {cleanup_stats.get('removed_count', 0)} indices")
            logger.info(f"[Batch {batch_idx}] Remaining indices: {cleanup_stats.get('remaining_count', 0)}")
        
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