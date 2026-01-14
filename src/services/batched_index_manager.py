"""
Batched Index Manager for training with limited memory.

This manager handles:
1. Loading repos in batches (e.g., 10-20 repos at a time)
2. Pre-indexing only the current batch
3. Training on that batch
4. Cleaning up indices before loading the next batch
5. Cycling through all repos in the dataset
"""

import shutil
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict
import hashlib

from datasets import Dataset
from loguru import logger


def get_repo_commit_hash(repo_name: str, commit: str) -> str:
    """Get unique hash for (repo, commit) pair."""
    key = f"{repo_name}:{commit}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class BatchedIndexManager:
    """
    Manages batched indexing for training.
    
    Key features:
    - Indices only current batch of repos
    - Cleans up after each batch to free disk/memory
    - Tracks progress across batches
    - Supports resuming from checkpoint
    """
    
    def __init__(
        self,
        dataset: Dataset,
        cache_dir: Path,
        batch_size: int = 15,
        repo_field: str = "repo",
        commit_field: str = "base_commit",
    ):
        """
        Initialize batched index manager.
        
        Args:
            dataset: Full dataset to train on
            cache_dir: Directory to store indices
            batch_size: Number of unique repos per batch
            repo_field: Field name for repository
            commit_field: Field name for commit hash
        """
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.repo_field = repo_field
        self.commit_field = commit_field
        
        # Group instances by (repo, commit)
        self.repo_commit_to_instances = defaultdict(list)
        for idx, instance in enumerate(dataset):
            key = (instance[repo_field], instance[commit_field])
            self.repo_commit_to_instances[key].append(idx)
        
        # Get unique repo-commit pairs
        self.unique_repo_commits = list(self.repo_commit_to_instances.keys())
        
        # Calculate batches
        self.num_batches = (len(self.unique_repo_commits) + batch_size - 1) // batch_size
        
        logger.info(f"[BatchedIndexManager] Dataset: {len(dataset)} instances")
        logger.info(f"[BatchedIndexManager] Unique repos: {len(self.unique_repo_commits)}")
        logger.info(f"[BatchedIndexManager] Batch size: {batch_size} repos/batch")
        logger.info(f"[BatchedIndexManager] Total batches: {self.num_batches}")
        
        # Track current state
        self.current_batch_idx = 0
        self.indexed_hashes: Set[str] = set()
    
    def get_batch_repos(self, batch_idx: int) -> List[tuple]:
        """Get list of (repo, commit) pairs for a batch."""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.unique_repo_commits))
        return self.unique_repo_commits[start_idx:end_idx]
    
    def get_batch_instances(self, batch_idx: int) -> List[int]:
        """Get list of dataset indices for a batch."""
        batch_repos = self.get_batch_repos(batch_idx)
        instance_indices = []
        for repo_commit in batch_repos:
            instance_indices.extend(self.repo_commit_to_instances[repo_commit])
        return instance_indices
    
    def get_batch_dataset(self, batch_idx: int) -> Dataset:
        """Get subset of dataset for a batch."""
        instance_indices = self.get_batch_instances(batch_idx)
        return self.dataset.select(instance_indices)
    
    def get_required_indices(self, batch_idx: int) -> Set[str]:
        """Get set of index hashes required for a batch."""
        batch_repos = self.get_batch_repos(batch_idx)
        return {
            get_repo_commit_hash(repo, commit) 
            for repo, commit in batch_repos
        }
    
    def cleanup_batch_indices(self, batch_idx: int):
        """Clean up indices for a specific batch to free disk space."""
        batch_repos = self.get_batch_repos(batch_idx)
        
        cleaned_count = 0
        freed_mb = 0.0
        
        for repo, commit in batch_repos:
            repo_hash = get_repo_commit_hash(repo, commit)
            index_dir = self.cache_dir / repo_hash
            
            if index_dir.exists():
                # Calculate size before deletion
                size_mb = sum(
                    f.stat().st_size 
                    for f in index_dir.rglob("*") 
                    if f.is_file()
                ) / (1024 * 1024)
                
                # Remove directory
                shutil.rmtree(index_dir)
                
                cleaned_count += 1
                freed_mb += size_mb
                
                # Remove from tracked set
                if repo_hash in self.indexed_hashes:
                    self.indexed_hashes.remove(repo_hash)
        
        logger.info(
            f"[BatchedIndexManager] Cleaned batch {batch_idx}: "
            f"{cleaned_count} indices, {freed_mb:.1f} MB freed"
        )
    
    def cleanup_all_indices(self):
        """Clean up ALL indices to start fresh."""
        if not self.cache_dir.exists():
            return
        
        total_size = sum(
            f.stat().st_size 
            for f in self.cache_dir.rglob("*") 
            if f.is_file()
        ) / (1024 * 1024)
        
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.indexed_hashes.clear()
        
        logger.info(
            f"[BatchedIndexManager] Cleaned all indices: {total_size:.1f} MB freed"
        )
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about current cache state."""
        if not self.cache_dir.exists():
            return {
                "total_indices": 0,
                "cache_size_mb": 0.0,
                "indexed_hashes": []
            }
        
        indices = list(self.cache_dir.iterdir())
        total_size = sum(
            f.stat().st_size 
            for d in indices 
            for f in d.rglob("*") 
            if d.is_dir() and f.is_file()
        ) / (1024 * 1024)
        
        return {
            "total_indices": len(indices),
            "cache_size_mb": total_size,
            "indexed_hashes": list(self.indexed_hashes),
        }
    
    def advance_batch(self):
        """Move to next batch, cleaning up previous batch."""
        if self.current_batch_idx >= self.num_batches:
            logger.warning("[BatchedIndexManager] Already at last batch")
            return False
        
        # Clean up current batch before advancing
        if self.current_batch_idx > 0:
            self.cleanup_batch_indices(self.current_batch_idx - 1)
        
        self.current_batch_idx += 1
        
        batch_repos = self.get_batch_repos(self.current_batch_idx)
        logger.info(
            f"[BatchedIndexManager] Advanced to batch {self.current_batch_idx}/{self.num_batches} "
            f"({len(batch_repos)} repos)"
        )
        
        return True
    
    def get_progress(self) -> Dict:
        """Get current training progress."""
        current_instances = len(self.get_batch_instances(self.current_batch_idx))
        total_instances = len(self.dataset)
        instances_so_far = sum(
            len(self.get_batch_instances(i)) 
            for i in range(self.current_batch_idx)
        )
        
        return {
            "current_batch": self.current_batch_idx,
            "total_batches": self.num_batches,
            "current_batch_instances": current_instances,
            "instances_completed": instances_so_far,
            "total_instances": total_instances,
            "progress_pct": (instances_so_far / total_instances) * 100,
        }