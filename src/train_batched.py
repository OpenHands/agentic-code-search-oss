"""
Batched training script for code search with two-phase GPU/CPU indexing.

Simplified approach: Index → Train → Cleanup → Repeat
No custom trainer needed - just manage batches externally.
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
from src.services.batch_state_manager import BatchStateManager
try:
    from src.generator.code_search_generator import CodeSearchGenerator
    from src.async_trainer import CustomFullyAsyncRayPPOTrainer as FullyAsyncRayPPOTrainer
except ImportError as e:
    logger.error(f"Could not import required modules: {e}")
    raise


class BatchedCodeSearchPPOExp(BasePPOExp):
    """Extended experiment class with batched indexing - NO custom trainer needed."""
    
    def __init__(self, cfg: DictConfig):
        self.batch_manager: Optional[BatchedIndexManager] = None
        self.embedding_service = None
        self.current_batch_dataset = None
        self.batch_state_manager: Optional[BatchStateManager] = None
        self._batched_mode = cfg.get("batched_indexing", {}).get("enabled", False)
        self._shared_tracker = None
        
        super().__init__(cfg)
        
        if self._batched_mode and (not hasattr(self, 'train_ds') or self.train_ds is None):
            logger.info("[Init] Loading full dataset for batch manager...")
            self._load_dataset()
    
    def get_tracker(self):
        """Reuse tracker across batches."""
        if self._batched_mode and hasattr(self, '_shared_tracker') and self._shared_tracker is not None:
            return self._shared_tracker
        else:
            return super().get_tracker()
    
    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, 
                    inference_engine_client, generator, colocate_pg):
        """Use base FullyAsyncRayPPOTrainer - no customization needed!"""
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
        """Load training dataset if not already loaded."""
        if hasattr(self, 'train_ds') and self.train_ds is not None:
            return
        
        train_data_paths = self.cfg.data.train_data
        datasets = []
        for path in train_data_paths:
            ds = load_dataset("parquet", data_files=path, split="train")
            datasets.append(ds)
        
        self.train_ds = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
        logger.info(f"[Init] Loaded {len(self.train_ds)} training samples")
    
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initialize CodeSearchGenerator with semantic search support."""
        semantic_search_cfg = cfg.get('semantic_search', OmegaConf.create({
            'enabled': True,
            'embedding_model': 'jinaai/jina-code-embeddings-0.5b',
            'reranker_model': None,
            'max_indices': 50
        }))
        return CodeSearchGenerator(
            model_name=cfg.trainer.policy.model.path,
            generator_cfg=cfg.generator,
            semantic_search_cfg=semantic_search_cfg,
            skyrl_gym_cfg=OmegaConf.create({"max_env_workers": 0}),
            tokenizer=tokenizer,
            inference_engine_client=inference_engine_client,
        )
    
    def setup_batched_indexing(self):
        """
        Initialize batched indexing components with state management.
        
        Creates:
        - BatchManager for dataset organization
        - BatchStateManager for checkpoint/resume support
        - EmbeddingService actor for CPU retrieval
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
        
        logger.info(f"[Batched] Total batches: {self.batch_manager.num_batches}")
        logger.info(f"[Batched] Batch size: {batch_config.batch_size} repos/batch")
        logger.info(f"[Batched] Cache dir: {batch_config.cache_dir}")
        logger.info(f"[Batched] Repos dir: {batch_config.repos_dir}")
        
        # 2. Initialize batch state manager for checkpointing/resume
        from src.services.batch_state_manager import BatchStateManager
        
        state_file = Path(self.cfg.trainer.ckpt_path) / "batch_state.json"
        self.batch_state_manager = BatchStateManager(
            state_file=state_file,
            total_batches=self.batch_manager.num_batches
        )
        
        # Print current progress
        self.batch_state_manager.print_summary()
        
        # 3. Initialize EmbeddingService actor (for retrieval during training)
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
            num_gpus=0,  # Don't reserve GPU - will use CPU for retrieval
            lifetime="detached",  # Keep alive across batches
            max_restarts=-1,  # Auto-restart on failure
            max_task_retries=-1  # Retry failed tasks
        ).remote(
            embedding_model=semantic_config.get("embedding_model", "jinaai/jina-code-embeddings-0.5b"),
            reranker_model=semantic_config.get("reranker_model"),
            cache_dir=batch_config.cache_dir,
            max_indices=semantic_config.get("max_indices", 50),
        )
        
        # Verify actor is accessible
        try:
            test_actor = ray.get_actor("embedding_service")
            logger.info("[Batched] ✓ EmbeddingService actor initialized and accessible")
        except ValueError as e:
            logger.error(f"[Batched] ✗ Failed to verify embedding_service actor: {e}")
            raise
        
        logger.info("[Batched] ✓ Batched indexing setup complete\n")
    
    def clone_batch_repos(self, batch_repos: List[tuple]) -> tuple[List[tuple], List[tuple]]:
        """Clone repositories for a batch."""
        logger.info(f"[Cloning] Starting clone for {len(batch_repos)} repositories...")
        
        repos_dir = Path(self.cfg.batched_indexing.repos_dir)
        repos_dir.mkdir(parents=True, exist_ok=True)
        
        successful = []
        failed = []
        
        for repo_name, commit in batch_repos:
            dir_name = f"{repo_name.replace('/', '__')}__{commit[:8]}"
            repo_path = repos_dir / dir_name
            
            # Check if already cloned at correct commit
            if repo_path.exists() and (repo_path / ".git").exists():
                try:
                    result = subprocess.run(
                        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                        capture_output=True, text=True, check=True, timeout=10
                    )
                    if result.stdout.strip() == commit:
                        logger.info(f"[Cloning] ✓ {repo_name}@{commit[:7]} already cloned")
                        successful.append((repo_name, commit))
                        continue
                    shutil.rmtree(repo_path)
                except:
                    shutil.rmtree(repo_path)
            
            # Clone
            try:
                subprocess.run(
                    ["git", "clone", "--quiet", f"https://github.com/{repo_name}.git", str(repo_path)],
                    check=True, capture_output=True, timeout=300
                )
                subprocess.run(
                    ["git", "-C", str(repo_path), "checkout", "--quiet", commit],
                    check=True, capture_output=True, timeout=60
                )
                
                logger.info(f"[Cloning] ✓ {repo_name}@{commit[:7]}")
                successful.append((repo_name, commit))
                
            except Exception as e:
                logger.error(f"[Cloning] ✗ {repo_name}@{commit[:7]}: {e}")
                failed.append((repo_name, commit, str(e)))
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
        
        logger.info(f"[Cloning] Complete: {len(successful)}/{len(batch_repos)} successful")
        return successful, failed
    
    def run_batch(self, batch_idx: int):
        """
        Run training for a single batch with full state tracking.
        
        Phases:
        0. Clone repositories
        1. Index on GPU (parallel workers)
        2. Setup CPU retrieval
        3. Prepare training dataset
        4. Train (new trainer per batch)
        5. Cleanup
        
        Args:
            batch_idx: Index of batch to process (0-indexed)
        """
        from src.mcp_server.training_semantic_search_server import get_repo_commit_hash
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[Batch {batch_idx}/{self.batch_manager.num_batches-1}] Starting...")
        logger.info(f"{'='*80}")
        
        # ✅ Mark batch as started in state manager
        self.batch_state_manager.start_batch(batch_idx)
        
        try:
            batch_repos = self.batch_manager.get_batch_repos(batch_idx)
            logger.info(f"[Batch {batch_idx}] Repos in batch: {len(batch_repos)}")
            
            # ===================================================================
            # PHASE 0: CLONING
            # ===================================================================
            logger.info(f"[Batch {batch_idx}] Phase 0: Cloning repositories...")
            successful_clones, failed_clones = self.clone_batch_repos(batch_repos)
            
            if not successful_clones:
                error_msg = f"No repositories were successfully cloned (0/{len(batch_repos)})"
                logger.error(f"[Batch {batch_idx}] ✗ {error_msg}")
                if failed_clones:
                    logger.error(f"[Batch {batch_idx}] Clone failures:")
                    for repo_name, commit, error in failed_clones:
                        logger.error(f"  - {repo_name}@{commit[:7]}: {error}")
                raise RuntimeError(error_msg)
            
            logger.info(f"[Batch {batch_idx}] ✓ Cloned {len(successful_clones)}/{len(batch_repos)} repos")
            repos_to_index = successful_clones
            
            # ===================================================================
            # PHASE 1: INDEXING (GPU) - Parallel Workers
            # ===================================================================
            logger.info(f"[Batch {batch_idx}] Phase 1: Indexing on GPU (parallel)...")
            
            from src.services.embedding_service import EmbeddingWorker
            
            num_workers = self.cfg.batched_indexing.get("num_index_workers", 4)
            semantic_config = self.cfg.get("semantic_search", {})
            repos_dir = Path(self.cfg.batched_indexing.repos_dir)
            
            workers = []
            futures = []
            task_mapping = {}
            results = []
            errors = []
            
            try:
                # Create GPU workers
                logger.info(f"[Batch {batch_idx}] Creating {num_workers} GPU indexing workers...")
                workers = [
                    EmbeddingWorker.remote(
                        worker_id=i,
                        embedding_model=semantic_config.get("embedding_model", "jinaai/jina-code-embeddings-0.5b"),
                        cache_dir=str(self.cfg.batched_indexing.cache_dir),
                    )
                    for i in range(num_workers)
                ]
                
                logger.info(f"[Batch {batch_idx}] ✓ Created {num_workers} GPU workers")
                
                # Distribute indexing tasks across workers
                for i, (repo_name, commit) in enumerate(repos_to_index):
                    # Find repo path
                    dir_name = f"{repo_name.replace('/', '__')}__{commit[:8]}"
                    repo_path = repos_dir / dir_name
                    
                    if not repo_path.exists():
                        logger.warning(f"[Batch {batch_idx}] Repo not found: {repo_path}")
                        errors.append((repo_name, commit, "Repo directory not found"))
                        continue
                    
                    # Assign to worker (round-robin)
                    worker = workers[i % num_workers]
                    future = worker.index_repo.remote(repo_name, commit, str(repo_path))
                    futures.append(future)
                    task_mapping[future] = (repo_name, commit)
                
                logger.info(f"[Batch {batch_idx}] Launched {len(futures)} indexing tasks")
                
                if len(futures) == 0:
                    raise RuntimeError(f"No indexing tasks were launched!")
                
                # Wait for completion with timeout and progress tracking
                from tqdm import tqdm
                
                indexing_start_time = time.time()
                max_indexing_time = 1800  # 30 minutes max
                last_progress_time = time.time()
                last_progress_count = 0
                
                with tqdm(total=len(futures), desc="Indexing repos") as pbar:
                    remaining_futures = list(futures)
                    
                    while remaining_futures:
                        # Check global timeout
                        elapsed = time.time() - indexing_start_time
                        if elapsed > max_indexing_time:
                            logger.error(f"[Batch {batch_idx}] ⏱️  INDEXING TIMEOUT after {elapsed:.0f}s")
                            logger.error(f"[Batch {batch_idx}] Completed: {len(results)}/{len(futures)}")
                            logger.error(f"[Batch {batch_idx}] Stuck on: {len(remaining_futures)} repos")
                            
                            # Log which repos are stuck (first 5)
                            for future in remaining_futures[:5]:
                                repo_name, commit = task_mapping[future]
                                logger.error(f"  - STUCK: {repo_name}@{commit[:7]}")
                            
                            # Cancel remaining futures
                            for future in remaining_futures:
                                try:
                                    ray.cancel(future)
                                except:
                                    pass
                                repo_name, commit = task_mapping[future]
                                errors.append((repo_name, commit, "Indexing timeout"))
                            
                            break
                        
                        # Check for stalled progress (no progress in 5 minutes)
                        time_since_progress = time.time() - last_progress_time
                        if time_since_progress > 300 and len(results) == last_progress_count:
                            logger.warning(f"[Batch {batch_idx}] ⚠️  No progress in {time_since_progress:.0f}s")
                            logger.warning(f"[Batch {batch_idx}] Completed: {len(results)}/{len(futures)}")
                            logger.warning(f"[Batch {batch_idx}] Possibly stuck on:")
                            for future in remaining_futures[:3]:
                                repo_name, commit = task_mapping[future]
                                logger.warning(f"  - {repo_name}@{commit[:7]}")
                        
                        # Wait for next completion
                        done, remaining_futures = ray.wait(remaining_futures, num_returns=1, timeout=5.0)
                        
                        for done_future in done:
                            try:
                                result = ray.get(done_future, timeout=10.0)
                                results.append(result)
                                pbar.update(1)
                                
                                # Update progress tracking
                                last_progress_time = time.time()
                                last_progress_count = len(results)
                                
                                repo_name, commit = task_mapping[done_future]
                                
                                if result.get('success', False):
                                    chunks = result.get('chunks', 0)
                                    logger.info(f"[Batch {batch_idx}] ✓ {repo_name}@{commit[:7]} - {chunks} chunks")
                                else:
                                    error_msg = result.get('error', 'Unknown error')
                                    logger.error(f"[Batch {batch_idx}] ✗ {repo_name}@{commit[:7]} - {error_msg}")
                                    errors.append((repo_name, commit, error_msg))
                                    
                            except ray.exceptions.RayTaskError as e:
                                repo_name, commit = task_mapping[done_future]
                                error_str = str(e)[:500]
                                logger.error(f"[Batch {batch_idx}] ✗ {repo_name}@{commit[:7]} - Ray task error")
                                errors.append((repo_name, commit, f"Ray task error: {error_str}"))
                                pbar.update(1)
                                last_progress_time = time.time()
                                last_progress_count = len(results)
                                
                            except Exception as e:
                                repo_name, commit = task_mapping[done_future]
                                logger.error(f"[Batch {batch_idx}] ✗ {repo_name}@{commit[:7]} - Exception: {e}")
                                errors.append((repo_name, commit, str(e)[:200]))
                                pbar.update(1)
                                last_progress_time = time.time()
                                last_progress_count = len(results)
            
            finally:
                # ✅ GUARANTEED CLEANUP - Always runs, even if error
                logger.info(f"[Batch {batch_idx}] Phase 2: Cleaning up GPU indexing workers...")
                logger.info(f"[Batch {batch_idx}] Killing {len(workers)} GPU workers...")
                
                for i, worker in enumerate(workers):
                    try:
                        ray.kill(worker, no_restart=True)
                        logger.debug(f"[Batch {batch_idx}] ✓ Killed worker {i}")
                    except Exception as e:
                        logger.warning(f"[Batch {batch_idx}] Could not kill worker {i}: {e}")
                
                # Wait for cleanup
                logger.info(f"[Batch {batch_idx}] Waiting for Ray to release GPU resources...")
                time.sleep(5)
                
                # Check available resources
                try:
                    available_resources = ray.available_resources()
                    gpu_available = available_resources.get('GPU', 0)
                    cpu_available = available_resources.get('CPU', 0)
                    logger.info(f"[Batch {batch_idx}] Available after cleanup: GPU={gpu_available}, CPU={cpu_available}")
                except Exception as e:
                    logger.warning(f"[Batch {batch_idx}] Could not check resources: {e}")
                
                logger.info(f"[Batch {batch_idx}] ✓ GPU indexing workers terminated, GPU freed")
            
            # Process indexing results
            successful_indices = [
                (r['repo_name'], r['commit']) 
                for r in results if r.get('success', False)
            ]
            failed_indices = [
                (r['repo_name'], r['commit'], r.get('error', 'Unknown')) 
                for r in results if not r.get('success', False)
            ]
            
            logger.info(f"\n[Batch {batch_idx}] Indexing results:")
            logger.info(f"  Tasks launched:   {len(futures)}")
            logger.info(f"  Results received: {len(results)}")
            logger.info(f"  Successful:       {len(successful_indices)}")
            logger.info(f"  Failed:           {len(failed_indices)}")
            
            # Handle incomplete results
            if len(results) < len(futures):
                logger.warning(f"[Batch {batch_idx}] ⚠️  Not all indexing tasks completed!")
                logger.warning(f"[Batch {batch_idx}] Expected: {len(futures)}, Got: {len(results)}")
            
            # Detailed error reporting
            if errors:
                logger.error(f"\n[Batch {batch_idx}] Indexing errors (showing first 10):")
                for repo_name, commit, error in errors[:10]:
                    logger.error(f"  - {repo_name}@{commit[:7]}: {error[:200]}")
                if len(errors) > 10:
                    logger.error(f"  ... and {len(errors) - 10} more errors")
            
            # ✅ Save indexing results to state
            self.batch_state_manager.update_batch_indexing(
                batch_idx,
                successful_repos=successful_indices,
                failed_repos=failed_clones + failed_indices
            )
            
            # Verify .ready markers exist
            cache_dir = Path(self.cfg.batched_indexing.cache_dir)
            verified_indices = []
            missing_indices = []

            for repo_name, commit in successful_indices:
                repo_hash = get_repo_commit_hash(repo_name, commit)
                ready_file = cache_dir / repo_hash / ".ready"
                
                if ready_file.exists():
                    verified_indices.append((repo_name, commit))
                else:
                    logger.error(f"[Batch {batch_idx}] ✗ Missing .ready marker for {repo_name}@{commit[:7]} (hash: {repo_hash})")
                    missing_indices.append((repo_name, commit))

            if missing_indices:
                logger.error(f"[Batch {batch_idx}] ✗ {len(missing_indices)} indices missing .ready markers!")
                logger.error(f"[Batch {batch_idx}] Waiting 5s for filesystem sync...")
                time.sleep(5)
                
                # Check again after sync
                still_missing = []
                for repo_name, commit in missing_indices:
                    repo_hash = get_repo_commit_hash(repo_name, commit)
                    ready_file = cache_dir / repo_hash / ".ready"
                    if ready_file.exists():
                        verified_indices.append((repo_name, commit))
                    else:
                        still_missing.append((repo_name, commit))
                
                if still_missing:
                    logger.error(f"[Batch {batch_idx}] ⚠️  Still missing .ready markers after sync:")
                    for repo_name, commit in still_missing[:5]:
                        logger.error(f"  - {repo_name}@{commit[:7]}")
                    logger.error(f"[Batch {batch_idx}] Marking these repos as failed...")
                    
                    # Move to failed list
                    for repo_name, commit in still_missing:
                        errors.append((repo_name, commit, "Missing .ready marker after sync"))
                        if (repo_name, commit) in verified_indices:
                            verified_indices.remove((repo_name, commit))

            logger.info(f"[Batch {batch_idx}] ✓ Verified {len(verified_indices)} indices with .ready markers")
            
            if not verified_indices:
                error_msg = "No repositories were successfully indexed with verified .ready markers"
                logger.error(f"[Batch {batch_idx}] ✗ {error_msg}")
                raise RuntimeError(error_msg)
            
            # ===================================================================
            # PHASE 3: RETRIEVAL SETUP (CPU)
            # ===================================================================
            logger.info(f"[Batch {batch_idx}] Phase 3: Setting up CPU retrieval...")
            
            # Get embedding service (should already exist from setup)
            try:
                embedding_service = ray.get_actor("embedding_service")
            except ValueError:
                logger.error("[Batch {batch_idx}] embedding_service actor not found!")
                raise RuntimeError("EmbeddingService actor missing - check setup_batched_indexing()")
            
            # Enter retrieval phase (models on CPU)
            ray.get(embedding_service.enter_retrieval_phase.remote())
            
            stats = ray.get(embedding_service.get_cache_stats.remote())
            logger.info(f"[Batch {batch_idx}] ✓ Retrieval ready on {stats['current_device']}")
            logger.info(f"[Batch {batch_idx}] Phase: {stats['current_phase']}")
            
            # ===================================================================
            # PHASE 4: PREPARE TRAINING DATASET
            # ===================================================================
            logger.info(f"[Batch {batch_idx}] Phase 4: Preparing training dataset...")
            
            batch_dataset = self.batch_manager.get_batch_dataset(batch_idx)
            logger.info(f"[Batch {batch_idx}] Original batch dataset size: {len(batch_dataset)}")

            # Filter out failed repos
            all_failed_repos = set()
            for repo_name, commit, _ in failed_clones + errors:
                all_failed_repos.add((repo_name, commit))

            logger.info(f"[Batch {batch_idx}] Failed repos to filter out: {len(all_failed_repos)}")

            if all_failed_repos:
                failed_hashes = {
                    get_repo_commit_hash(repo_name, commit) 
                    for repo_name, commit in all_failed_repos
                }
                
                def keep_instance(instance):
                    instance_hash = get_repo_commit_hash(
                        instance[self.batch_manager.repo_field],
                        instance[self.batch_manager.commit_field]
                    )
                    return instance_hash not in failed_hashes
                
                original_size = len(batch_dataset)
                batch_dataset = batch_dataset.filter(keep_instance)
                filtered_size = len(batch_dataset)
                
                logger.info(f"[Batch {batch_idx}] Filtered dataset: {original_size} -> {filtered_size} instances")
            
            if len(batch_dataset) == 0:
                error_msg = "No training instances remaining after filtering failed repos"
                logger.error(f"[Batch {batch_idx}] ✗ {error_msg}")
                raise RuntimeError(error_msg)
            
            logger.info(f"[Batch {batch_idx}] ✓ Training dataset ready: {len(batch_dataset)} instances")
            
            # ===================================================================
            # PHASE 5: TRAINING
            # ===================================================================
            logger.info(f"[Batch {batch_idx}] Phase 5: Training (GPU) with CPU retrieval...")
            
            # Create dataset wrapper
            from skyrl_train.dataset import PromptDataset
            prompts_dataset = object.__new__(PromptDataset)
            prompts_dataset.tokenizer = self.tokenizer
            prompts_dataset.max_prompt_length = self.cfg.trainer.max_prompt_length
            prompts_dataset.prompt_key = "prompt"
            prompts_dataset.env_class_key = "env_class"
            prompts_dataset.num_workers = 8
            prompts_dataset.datasets = None
            prompts_dataset.dataframe = batch_dataset

            # Set as current dataset
            self.train_dataset = prompts_dataset

            # Create fresh trainer for this batch
            logger.info(f"[Batch {batch_idx}] Creating new trainer...")
            trainer = self._setup_trainer()

            # ✅ Enable batched mode to skip epoch-end validation
            trainer.enable_batched_mode()

            # Log batch info
            if hasattr(trainer, 'tracker') and trainer.tracker is not None:
                trainer.tracker.log({
                    'batch/current_batch': batch_idx,
                    'batch/training_instances': len(batch_dataset),
                    'batch/indexed_repos': len(successful_indices),
                }, step=trainer.global_step)

            # Update state
            self.batch_state_manager.update_batch_training(
                batch_idx,
                training_instances=len(batch_dataset),
                training_steps=trainer.num_steps_per_epoch
            )

            # Run training
            start_time = time.time()
            logger.info(f"[Batch {batch_idx}] Training for {trainer.num_steps_per_epoch} steps...")

            asyncio.run(trainer.train())  # Runs cleanly, no assertion error

            train_time = time.time() - start_time
            logger.info(f"[Batch {batch_idx}] ✓ Training complete in {train_time:.1f}s")

            # Get final global_step
            final_global_step = trainer.global_step - 1

            # Mark batch as complete
            self.batch_state_manager.complete_batch(batch_idx, final_global_step)
            # ===================================================================
            # PHASE 6: CLEANUP
            # ===================================================================
            if self.cfg.batched_indexing.get("cleanup_between_batches", True):
                logger.info(f"[Batch {batch_idx}] Phase 6: Cleaning up batch...")
                
                # Cleanup indices
                verified_hashes = [
                    get_repo_commit_hash(repo_name, commit)
                    for repo_name, commit in verified_indices
                ]
                
                try:
                    ray.get(embedding_service.cleanup_batch_indices.remote(verified_hashes))
                    logger.info(f"[Batch {batch_idx}] ✓ Cleaned up {len(verified_hashes)} indices")
                except Exception as e:
                    logger.warning(f"[Batch {batch_idx}] Cleanup warning (non-fatal): {e}")
                
                # Cleanup cloned repos
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
                
                logger.info(f"[Batch {batch_idx}] Cleaned up {cleaned_repos} repos")
            
            logger.info(f"[Batch {batch_idx}] ✓ Complete\n")
            
        except Exception as e:
            # ✅ Mark batch as failed in state
            self.batch_state_manager.fail_batch(batch_idx, str(e))
            raise
    
    def run(self):
        """Main training loop with resume support."""
        if not self.cfg.get("batched_indexing", {}).get("enabled", False):
            logger.info("[Training] Regular training (batched disabled)")
            trainer = super()._setup_trainer()
            asyncio.run(trainer.train())
            return
        
        logger.info("[Training] Batched training mode")
        self.setup_batched_indexing()
        
        # ✅ Determine starting batch (resume support)
        start_batch = self.batch_state_manager.get_resume_batch()
        
        if start_batch is None:
            logger.info("[Training] All batches already complete!")
            return
        
        if start_batch > 0:
            logger.info(f"[Training] ⚡ RESUMING from batch {start_batch}")
            logger.info(f"[Training] Batches 0-{start_batch-1} already complete")
        
        # ✅ Create shared tracker
        from skyrl_train.utils.tracking import Tracking
        logger.info("[Training] Creating shared tracker...")
        self._shared_tracker = Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=self.cfg.trainer.logger,
            config=self.cfg,
        )
        
        num_batches = self.batch_manager.num_batches
        logger.info(f"[Training] Processing batches {start_batch} to {num_batches-1}\n")
        
        try:
            for batch_idx in range(start_batch, num_batches):  # ✅ Start from resume point
                try:
                    self.run_batch(batch_idx)
                    
                    # Print progress
                    summary = self.batch_state_manager.get_progress_summary()
                    logger.info(
                        f"[Progress] {summary['completed_batches']}/{num_batches} "
                        f"batches complete ({summary['progress_percent']:.1f}%)\n"
                    )
                    
                except Exception as e:
                    logger.error(f"[Batch {batch_idx}] Failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    if not self.cfg.get("batched_indexing", {}).get("continue_on_batch_failure", False):
                        raise
                    logger.warning(f"[Batch {batch_idx}] Continuing to next batch...")
            
            logger.info("[Training] ✓ All batches complete!")
            
            # ✅ Print final summary
            self.batch_state_manager.print_summary()
        
        finally:
            # Finish tracker
            if hasattr(self, '_shared_tracker') and self._shared_tracker is not None:
                try:
                    if 'wandb' in self._shared_tracker.logger:
                        self._shared_tracker.logger['wandb'].finish(exit_code=0)
                    logger.info("[Training] ✓ WandB finished")
                except Exception as e:
                    logger.warning(f"Tracker finish error: {e}")
            
            # Cleanup embedding service
            try:
                embedding_service = ray.get_actor("embedding_service")
                ray.kill(embedding_service)
                logger.info("[Cleanup] ✓ Embedding service terminated")
            except:
                pass


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    """Ray remote entry point."""
    exp = BatchedCodeSearchPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    validate_cfg(cfg)

    # Setup rewards
    if hasattr(cfg.generator, "reward"):
        with open(cfg.generator.reward, "r") as f:
            reward_cfg = OmegaConf.load(f)
        cfg.generator.reward = reward_cfg.reward
    else:
        with open_dict(cfg):
            cfg.generator.reward = [{"fn": "multilevel_localization_f1_reward"}]

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    
    from skyrl_train.utils import prepare_runtime_environment
    from skyrl_train.utils.ppo_utils import sync_registries

    env_vars = prepare_runtime_environment(cfg)
    
    excludes = [
        "ckpts/", "*.ckpt", "*.pth", "*.pt", "*.safetensors", "*.bin",
        "logs/", "*.log", "*.out", "*.err",
        ".cache/", "__pycache__/", "*.pyc", ".pytest_cache/",
        ".venv/", "venv/", "env/", "ray_temp*/", "ray_spill/",
        "trajectories/", ".git/", "outputs/", "multirun/",
    ]

    ray.init(runtime_env={"env_vars": env_vars, "excludes": excludes})
    sync_registries()
    
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()