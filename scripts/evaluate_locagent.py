"""
Evaluate LocAgent localization results using our evaluation metrics.

This script evaluates the LocAgent baseline on SWE-Bench Lite using
file-level, module-level, and entity/function-level F1 scores.
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


def evaluate_locagent(results: list) -> dict:
    """
    Evaluate LocAgent results at file, module, and entity levels.
    
    Returns dict with:
    - file_f1: Average F1 score at file level
    - module_f1: Average F1 score at module level  
    - entity_f1: Average F1 score at entity/function level
    - per_instance: Detailed per-instance results
    """
    file_scores = []
    module_scores = []
    entity_scores = []
    per_instance = []
    
    # Track instances with valid ground truth
    valid_file_count = 0
    valid_module_count = 0
    valid_entity_count = 0
    
    for result in results:
        instance_id = result['instance_id']
        
        # Get predictions
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
        
        # Compute F1 scores
        file_f1 = compute_file_f1_score(pred_files, gt_files)
        module_f1 = compute_file_f1_score(pred_modules, gt_modules) if gt_modules else None
        entity_f1 = compute_file_f1_score(pred_entities, gt_entities) if gt_entities else None
        
        # Track scores (only for instances with valid ground truth)
        file_scores.append(file_f1)
        valid_file_count += 1
        
        if module_f1 is not None:
            module_scores.append(module_f1)
            valid_module_count += 1
            
        if entity_f1 is not None:
            entity_scores.append(entity_f1)
            valid_entity_count += 1
        
        per_instance.append({
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
        })
    
    # Compute averages
    avg_file_f1 = sum(file_scores) / len(file_scores) if file_scores else 0.0
    avg_module_f1 = sum(module_scores) / len(module_scores) if module_scores else 0.0
    avg_entity_f1 = sum(entity_scores) / len(entity_scores) if entity_scores else 0.0
    
    return {
        'file_f1': avg_file_f1,
        'module_f1': avg_module_f1,
        'entity_f1': avg_entity_f1,
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
    
    print("\n" + "="*60)
    print("LocAgent (Claude 3.5) Evaluation Results on SWE-Bench Lite")
    print("="*60)
    print(f"\nFile-level F1:     {metrics['file_f1']:.4f} ({metrics['valid_file_count']} instances)")
    print(f"Module-level F1:   {metrics['module_f1']:.4f} ({metrics['valid_module_count']} instances with module GT)")
    print(f"Entity-level F1:   {metrics['entity_f1']:.4f} ({metrics['valid_entity_count']} instances with entity GT)")
    print(f"\nTotal instances:   {metrics['total_instances']}")
    
    # Show some examples
    print("\n" + "-"*60)
    print("Sample Results (first 5 instances):")
    print("-"*60)
    for inst in metrics['per_instance'][:5]:
        print(f"\n{inst['instance_id']}:")
        print(f"  File F1: {inst['file_f1']:.4f}")
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
