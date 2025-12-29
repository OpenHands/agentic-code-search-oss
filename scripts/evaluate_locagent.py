"""
Evaluate LocAgent localization results using our evaluation metrics.

This script evaluates the LocAgent baseline on SWE-Bench Lite using:
- F1 scores at file, module, and entity/function levels
- Acc@k metrics (accuracy at top-k predictions)
- Top-k thresholded F1 scores (using only top-k predictions)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rewards.file_localization.file_localization import compute_file_f1_score


def load_locagent_results(filepath: str) -> list:
    """Load LocAgent localization results from JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_acc_at_k(predictions: list, ground_truth: set, k: int) -> float:
    """Compute Acc@k: 1 if any GT item is in top-k predictions, else 0."""
    if not ground_truth:
        return 0.0
    top_k = set(predictions[:k])
    return 1.0 if top_k & ground_truth else 0.0


def compute_recall_at_k(predictions: list, ground_truth: set, k: int) -> float:
    """Compute Recall@k: fraction of GT items found in top-k predictions."""
    if not ground_truth:
        return 0.0
    top_k = set(predictions[:k])
    return len(top_k & ground_truth) / len(ground_truth)


def evaluate_locagent(results: list, top_k_values: list = [1, 3, 5, 10]) -> dict:
    """
    Evaluate LocAgent results at file, module, and entity levels.
    
    Returns dict with:
    - F1 scores (using all predictions)
    - Acc@k metrics for various k values
    - Top-k thresholded F1 scores (using only top-k predictions)
    - per_instance: Detailed per-instance results
    """
    file_scores = []
    module_scores = []
    entity_scores = []
    per_instance = []
    
    # Acc@k tracking
    file_acc_at_k = {k: [] for k in top_k_values}
    module_acc_at_k = {k: [] for k in top_k_values}
    entity_acc_at_k = {k: [] for k in top_k_values}
    
    # Recall@k tracking
    file_recall_at_k = {k: [] for k in top_k_values}
    module_recall_at_k = {k: [] for k in top_k_values}
    entity_recall_at_k = {k: [] for k in top_k_values}
    
    # Top-k thresholded F1 tracking
    file_f1_at_k = {k: [] for k in top_k_values}
    module_f1_at_k = {k: [] for k in top_k_values}
    entity_f1_at_k = {k: [] for k in top_k_values}
    
    # Track instances with valid ground truth
    valid_file_count = 0
    valid_module_count = 0
    valid_entity_count = 0
    
    for result in results:
        instance_id = result['instance_id']
        
        # Get predictions (assumed to be ordered by relevance/confidence)
        pred_files = result.get('found_files', [])
        pred_modules = result.get('found_modules', [])
        pred_entities = result.get('found_entities', [])
        
        # Get ground truth from meta_data
        gt_file_changes = result.get('meta_data', {}).get('gt_file_changes', [])
        
        gt_files = []
        gt_modules = []
        gt_entities = []
        
        for change in gt_file_changes:
            if 'file' in change:
                gt_files.append(change['file'])
            if 'changes' in change:
                edited_modules = change['changes'].get('edited_modules', [])
                if edited_modules:
                    gt_modules.extend(edited_modules)
                edited_entities = change['changes'].get('edited_entities', [])
                if edited_entities:
                    gt_entities.extend(edited_entities)
        
        gt_files = set(gt_files)
        gt_modules = set(gt_modules)
        gt_entities = set(gt_entities)
        
        # Compute F1 scores (using all predictions)
        file_f1 = compute_file_f1_score(pred_files, gt_files)
        module_f1 = compute_file_f1_score(pred_modules, gt_modules) if gt_modules else None
        entity_f1 = compute_file_f1_score(pred_entities, gt_entities) if gt_entities else None
        
        # Track F1 scores
        file_scores.append(file_f1)
        valid_file_count += 1
        
        if module_f1 is not None:
            module_scores.append(module_f1)
            valid_module_count += 1
            
        if entity_f1 is not None:
            entity_scores.append(entity_f1)
            valid_entity_count += 1
        
        # Compute Acc@k, Recall@k, and top-k thresholded F1 for each k
        instance_metrics = {
            'instance_id': instance_id,
            'file_f1': file_f1,
            'module_f1': module_f1,
            'entity_f1': entity_f1,
            'pred_files': pred_files,
            'pred_modules': pred_modules,
            'pred_entities': pred_entities,
            'gt_files': list(gt_files),
            'gt_modules': list(gt_modules),
            'gt_entities': list(gt_entities),
        }
        
        for k in top_k_values:
            # File-level metrics
            file_acc_at_k[k].append(compute_acc_at_k(pred_files, gt_files, k))
            file_recall_at_k[k].append(compute_recall_at_k(pred_files, gt_files, k))
            file_f1_at_k[k].append(compute_file_f1_score(pred_files[:k], gt_files))
            
            # Module-level metrics (only if GT exists)
            if gt_modules:
                module_acc_at_k[k].append(compute_acc_at_k(pred_modules, gt_modules, k))
                module_recall_at_k[k].append(compute_recall_at_k(pred_modules, gt_modules, k))
                module_f1_at_k[k].append(compute_file_f1_score(pred_modules[:k], gt_modules))
            
            # Entity-level metrics (only if GT exists)
            if gt_entities:
                entity_acc_at_k[k].append(compute_acc_at_k(pred_entities, gt_entities, k))
                entity_recall_at_k[k].append(compute_recall_at_k(pred_entities, gt_entities, k))
                entity_f1_at_k[k].append(compute_file_f1_score(pred_entities[:k], gt_entities))
            
            instance_metrics[f'file_acc@{k}'] = file_acc_at_k[k][-1]
            instance_metrics[f'file_f1@{k}'] = file_f1_at_k[k][-1]
        
        per_instance.append(instance_metrics)
    
    # Compute averages
    avg_file_f1 = sum(file_scores) / len(file_scores) if file_scores else 0.0
    avg_module_f1 = sum(module_scores) / len(module_scores) if module_scores else 0.0
    avg_entity_f1 = sum(entity_scores) / len(entity_scores) if entity_scores else 0.0
    
    # Compute average Acc@k, Recall@k, and F1@k
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    return {
        # F1 scores (all predictions)
        'file_f1': avg_file_f1,
        'module_f1': avg_module_f1,
        'entity_f1': avg_entity_f1,
        
        # Acc@k metrics
        'file_acc_at_k': {k: avg(file_acc_at_k[k]) for k in top_k_values},
        'module_acc_at_k': {k: avg(module_acc_at_k[k]) for k in top_k_values},
        'entity_acc_at_k': {k: avg(entity_acc_at_k[k]) for k in top_k_values},
        
        # Recall@k metrics
        'file_recall_at_k': {k: avg(file_recall_at_k[k]) for k in top_k_values},
        'module_recall_at_k': {k: avg(module_recall_at_k[k]) for k in top_k_values},
        'entity_recall_at_k': {k: avg(entity_recall_at_k[k]) for k in top_k_values},
        
        # Top-k thresholded F1 scores
        'file_f1_at_k': {k: avg(file_f1_at_k[k]) for k in top_k_values},
        'module_f1_at_k': {k: avg(module_f1_at_k[k]) for k in top_k_values},
        'entity_f1_at_k': {k: avg(entity_f1_at_k[k]) for k in top_k_values},
        
        # Counts
        'valid_file_count': valid_file_count,
        'valid_module_count': valid_module_count,
        'valid_entity_count': valid_entity_count,
        'total_instances': len(results),
        'per_instance': per_instance,
    }


def main():
    # Path to LocAgent results
    locagent_path = Path(__file__).parent.parent.parent / 'LocAgent' / 'evaluation' / 'loc_output' / 'locagent' / 'claude_3-5' / 'loc_outputs.jsonl'
    
    if not locagent_path.exists():
        print(f"Error: LocAgent results not found at {locagent_path}")
        sys.exit(1)
    
    print(f"Loading LocAgent results from: {locagent_path}")
    results = load_locagent_results(locagent_path)
    print(f"Loaded {len(results)} instances")
    
    print("\nEvaluating LocAgent (Claude 3.5) on SWE-Bench Lite...")
    metrics = evaluate_locagent(results)
    
    print("\n" + "="*70)
    print("LocAgent (Claude 3.5) Evaluation Results on SWE-Bench Lite")
    print("="*70)
    
    # F1 Scores (all predictions)
    print(f"\n--- F1 Scores (using all predictions) ---")
    print(f"File-level F1:     {metrics['file_f1']:.4f} ({metrics['valid_file_count']} instances)")
    print(f"Module-level F1:   {metrics['module_f1']:.4f} ({metrics['valid_module_count']} instances with module GT)")
    print(f"Entity-level F1:   {metrics['entity_f1']:.4f} ({metrics['valid_entity_count']} instances with entity GT)")
    
    # Acc@k metrics
    print(f"\n--- Acc@k Metrics (any GT in top-k) ---")
    print(f"{'Level':<10} {'Acc@1':>8} {'Acc@3':>8} {'Acc@5':>8} {'Acc@10':>8}")
    print(f"{'File':<10} {metrics['file_acc_at_k'][1]:>8.4f} {metrics['file_acc_at_k'][3]:>8.4f} {metrics['file_acc_at_k'][5]:>8.4f} {metrics['file_acc_at_k'][10]:>8.4f}")
    print(f"{'Module':<10} {metrics['module_acc_at_k'][1]:>8.4f} {metrics['module_acc_at_k'][3]:>8.4f} {metrics['module_acc_at_k'][5]:>8.4f} {metrics['module_acc_at_k'][10]:>8.4f}")
    print(f"{'Entity':<10} {metrics['entity_acc_at_k'][1]:>8.4f} {metrics['entity_acc_at_k'][3]:>8.4f} {metrics['entity_acc_at_k'][5]:>8.4f} {metrics['entity_acc_at_k'][10]:>8.4f}")
    
    # Recall@k metrics
    print(f"\n--- Recall@k Metrics (fraction of GT in top-k) ---")
    print(f"{'Level':<10} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}")
    print(f"{'File':<10} {metrics['file_recall_at_k'][1]:>8.4f} {metrics['file_recall_at_k'][3]:>8.4f} {metrics['file_recall_at_k'][5]:>8.4f} {metrics['file_recall_at_k'][10]:>8.4f}")
    print(f"{'Module':<10} {metrics['module_recall_at_k'][1]:>8.4f} {metrics['module_recall_at_k'][3]:>8.4f} {metrics['module_recall_at_k'][5]:>8.4f} {metrics['module_recall_at_k'][10]:>8.4f}")
    print(f"{'Entity':<10} {metrics['entity_recall_at_k'][1]:>8.4f} {metrics['entity_recall_at_k'][3]:>8.4f} {metrics['entity_recall_at_k'][5]:>8.4f} {metrics['entity_recall_at_k'][10]:>8.4f}")
    
    # Top-k thresholded F1 scores
    print(f"\n--- F1@k Scores (using only top-k predictions) ---")
    print(f"{'Level':<10} {'F1@1':>8} {'F1@3':>8} {'F1@5':>8} {'F1@10':>8}")
    print(f"{'File':<10} {metrics['file_f1_at_k'][1]:>8.4f} {metrics['file_f1_at_k'][3]:>8.4f} {metrics['file_f1_at_k'][5]:>8.4f} {metrics['file_f1_at_k'][10]:>8.4f}")
    print(f"{'Module':<10} {metrics['module_f1_at_k'][1]:>8.4f} {metrics['module_f1_at_k'][3]:>8.4f} {metrics['module_f1_at_k'][5]:>8.4f} {metrics['module_f1_at_k'][10]:>8.4f}")
    print(f"{'Entity':<10} {metrics['entity_f1_at_k'][1]:>8.4f} {metrics['entity_f1_at_k'][3]:>8.4f} {metrics['entity_f1_at_k'][5]:>8.4f} {metrics['entity_f1_at_k'][10]:>8.4f}")
    
    print(f"\nTotal instances: {metrics['total_instances']}")
    
    # Show some examples
    print("\n" + "-"*70)
    print("Sample Results (first 5 instances):")
    print("-"*70)
    for inst in metrics['per_instance'][:5]:
        print(f"\n{inst['instance_id']}:")
        print(f"  File F1: {inst['file_f1']:.4f} | F1@1: {inst['file_f1@1']:.4f} | Acc@1: {inst['file_acc@1']:.4f}")
        module_str = f"{inst['module_f1']:.4f}" if inst['module_f1'] is not None else 'N/A'
        entity_str = f"{inst['entity_f1']:.4f}" if inst['entity_f1'] is not None else 'N/A'
        print(f"  Module F1: {module_str}")
        print(f"  Entity F1: {entity_str}")
        print(f"  Pred files: {inst['pred_files'][:3]}{'...' if len(inst['pred_files']) > 3 else ''}")
        print(f"  GT files: {inst['gt_files'][:3]}{'...' if len(inst['gt_files']) > 3 else ''}")
    
    # Save detailed results
    output_path = Path(__file__).parent.parent / 'evaluation_results' / 'locagent_evaluation.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'method': 'LocAgent',
            'model': 'Claude 3.5',
            'benchmark': 'SWE-Bench Lite',
            'metrics': {
                'file_f1': metrics['file_f1'],
                'module_f1': metrics['module_f1'],
                'entity_f1': metrics['entity_f1'],
                'file_acc_at_k': metrics['file_acc_at_k'],
                'module_acc_at_k': metrics['module_acc_at_k'],
                'entity_acc_at_k': metrics['entity_acc_at_k'],
                'file_recall_at_k': metrics['file_recall_at_k'],
                'module_recall_at_k': metrics['module_recall_at_k'],
                'entity_recall_at_k': metrics['entity_recall_at_k'],
                'file_f1_at_k': metrics['file_f1_at_k'],
                'module_f1_at_k': metrics['module_f1_at_k'],
                'entity_f1_at_k': metrics['entity_f1_at_k'],
            },
            'counts': {
                'total_instances': metrics['total_instances'],
                'valid_file_count': metrics['valid_file_count'],
                'valid_module_count': metrics['valid_module_count'],
                'valid_entity_count': metrics['valid_entity_count'],
            },
            'per_instance': metrics['per_instance'],
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
