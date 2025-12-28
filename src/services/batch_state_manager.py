"""
Batch state manager for checkpointing and resuming batched training.

Tracks:
- Which batch we're on
- Which batches are complete
- Which repos were successfully indexed per batch
- Allows resuming from crashes
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from loguru import logger
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class BatchState:
    """State for a single batch."""
    batch_idx: int
    status: str  # 'pending', 'indexing', 'training', 'complete', 'failed'
    successful_repos: List[Tuple[str, str]]  # [(repo_name, commit), ...]
    failed_repos: List[Tuple[str, str, str]]  # [(repo_name, commit, error), ...]
    training_instances: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    training_steps: Optional[int] = None
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        # Convert lists back to tuples
        data['successful_repos'] = [tuple(r) for r in data['successful_repos']]
        data['failed_repos'] = [tuple(r) for r in data['failed_repos']]
        return cls(**data)


class BatchStateManager:
    """
    Manages batch training state for checkpointing and resuming.
    
    State file structure:
    {
        "current_batch": 0,
        "total_batches": 139,
        "last_completed_batch": -1,
        "batches": {
            "0": {
                "batch_idx": 0,
                "status": "complete",
                "successful_repos": [["owner/repo", "commit"], ...],
                "failed_repos": [["owner/repo", "commit", "error"], ...],
                "training_instances": 15,
                "start_time": "2025-12-26T10:00:00",
                "end_time": "2025-12-26T10:15:00",
                "training_steps": 4
            },
            ...
        },
        "global_step": 5,
        "last_updated": "2025-12-26T10:15:00"
    }
    """
    
    def __init__(self, state_file: Path, total_batches: int):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.total_batches = total_batches
        
        # Load or initialize state
        if self.state_file.exists():
            self.load()
            logger.info(f"[BatchState] Loaded existing state from {self.state_file}")
            logger.info(f"[BatchState] Current batch: {self.state['current_batch']}")
            logger.info(f"[BatchState] Last completed: {self.state['last_completed_batch']}")
        else:
            self.state = {
                "current_batch": 0,
                "total_batches": total_batches,
                "last_completed_batch": -1,
                "batches": {},
                "global_step": 0,
                "last_updated": datetime.now().isoformat(),
            }
            self.save()
            logger.info(f"[BatchState] Initialized new state at {self.state_file}")
    
    def load(self):
        """Load state from disk."""
        with open(self.state_file, 'r') as f:
            self.state = json.load(f)
        
        # Validate
        if self.state['total_batches'] != self.total_batches:
            logger.warning(
                f"[BatchState] Total batches mismatch! "
                f"State has {self.state['total_batches']}, but dataset has {self.total_batches}"
            )
    
    def save(self):
        """Save state to disk atomically."""
        self.state['last_updated'] = datetime.now().isoformat()
        
        # Write to temp file first, then atomic rename
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        
        # Atomic rename
        temp_file.replace(self.state_file)
    
    def get_resume_batch(self) -> int:
        """
        Determine which batch to start from.
        
        Returns:
            Batch index to resume from (0-indexed)
        """
        last_completed = self.state['last_completed_batch']
        
        # If we've never completed a batch, start from 0
        if last_completed == -1:
            logger.info("[BatchState] No completed batches, starting from batch 0")
            return 0
        
        # Otherwise, start from the next batch after last completed
        next_batch = last_completed + 1
        
        if next_batch >= self.total_batches:
            logger.info("[BatchState] All batches complete!")
            return None
        
        logger.info(f"[BatchState] Resuming from batch {next_batch} (last completed: {last_completed})")
        return next_batch
    
    def start_batch(self, batch_idx: int):
        """Mark batch as started."""
        self.state['current_batch'] = batch_idx
        self.state['batches'][str(batch_idx)] = {
            "batch_idx": batch_idx,
            "status": "indexing",
            "successful_repos": [],
            "failed_repos": [],
            "training_instances": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "training_steps": None,
        }
        self.save()
        logger.info(f"[BatchState] Started batch {batch_idx}")
    
    def update_batch_indexing(
        self, 
        batch_idx: int, 
        successful_repos: List[Tuple[str, str]], 
        failed_repos: List[Tuple[str, str, str]]
    ):
        """Update batch with indexing results."""
        batch_key = str(batch_idx)
        if batch_key not in self.state['batches']:
            self.start_batch(batch_idx)
        
        self.state['batches'][batch_key]['status'] = 'training'
        self.state['batches'][batch_key]['successful_repos'] = successful_repos
        self.state['batches'][batch_key]['failed_repos'] = failed_repos
        self.save()
        logger.info(f"[BatchState] Batch {batch_idx}: indexed {len(successful_repos)} repos")
    
    def update_batch_training(self, batch_idx: int, training_instances: int, training_steps: int):
        """Update batch with training info."""
        batch_key = str(batch_idx)
        self.state['batches'][batch_key]['training_instances'] = training_instances
        self.state['batches'][batch_key]['training_steps'] = training_steps
        self.save()
    
    def complete_batch(self, batch_idx: int, global_step: int):
        """Mark batch as complete."""
        batch_key = str(batch_idx)
        self.state['batches'][batch_key]['status'] = 'complete'
        self.state['batches'][batch_key]['end_time'] = datetime.now().isoformat()
        self.state['last_completed_batch'] = batch_idx
        self.state['global_step'] = global_step
        self.save()
        logger.info(f"[BatchState] ✓ Batch {batch_idx} complete (global_step={global_step})")
    
    def fail_batch(self, batch_idx: int, error: str):
        """Mark batch as failed."""
        batch_key = str(batch_idx)
        if batch_key in self.state['batches']:
            self.state['batches'][batch_key]['status'] = 'failed'
            self.state['batches'][batch_key]['error'] = error
            self.state['batches'][batch_key]['end_time'] = datetime.now().isoformat()
        self.save()
        logger.error(f"[BatchState] ✗ Batch {batch_idx} failed: {error}")
    
    def get_batch_state(self, batch_idx: int) -> Optional[Dict]:
        """Get state for a specific batch."""
        return self.state['batches'].get(str(batch_idx))
    
    def get_completed_batches(self) -> List[int]:
        """Get list of completed batch indices."""
        completed = []
        for batch_idx_str, batch_data in self.state['batches'].items():
            if batch_data['status'] == 'complete':
                completed.append(int(batch_idx_str))
        return sorted(completed)
    
    def get_progress_summary(self) -> Dict:
        """Get progress summary."""
        completed = self.get_completed_batches()
        failed = [
            int(idx) for idx, data in self.state['batches'].items() 
            if data['status'] == 'failed'
        ]
        
        total_successful_repos = sum(
            len(data['successful_repos']) 
            for data in self.state['batches'].values()
            if data['status'] == 'complete'
        )
        
        return {
            'total_batches': self.total_batches,
            'completed_batches': len(completed),
            'failed_batches': len(failed),
            'current_batch': self.state['current_batch'],
            'last_completed_batch': self.state['last_completed_batch'],
            'global_step': self.state['global_step'],
            'progress_percent': 100 * len(completed) / self.total_batches,
            'total_indexed_repos': total_successful_repos,
        }
    
    def print_summary(self):
        """Print progress summary."""
        summary = self.get_progress_summary()
        
        logger.info("\n" + "="*80)
        logger.info("BATCH TRAINING PROGRESS")
        logger.info("="*80)
        logger.info(f"Total batches:        {summary['total_batches']}")
        logger.info(f"Completed:            {summary['completed_batches']} ({summary['progress_percent']:.1f}%)")
        logger.info(f"Failed:               {summary['failed_batches']}")
        logger.info(f"Current batch:        {summary['current_batch']}")
        logger.info(f"Last completed:       {summary['last_completed_batch']}")
        logger.info(f"Global training step: {summary['global_step']}")
        logger.info(f"Total indexed repos:  {summary['total_indexed_repos']}")
        logger.info("="*80 + "\n")