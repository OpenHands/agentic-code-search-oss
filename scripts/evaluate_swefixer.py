"""
Evaluate SWE-Fixer localization results using our evaluation metrics.

NOTE: SWE-Fixer only does FILE-LEVEL retrieval. It does NOT make module-level
or function-level predictions. The retriever model identifies which files need
modification, then the editor model generates patches for those files.

This script extracts file-level predictions from SWE-Fixer's patches and
evaluates them against ground truth.
"""

import json
import re
import sys
from pathlib import Path
from datasets import load_dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rewards.file_localization.file_localization import compute_file_f1_score


def extract_files_from_patch(patch: str) -> list:
    """Extract file paths from a git diff patch."""
    if not patch:
        return []
    files = re.findall(r'diff --git a/(.+?) b/', patch)
    return list(set(files))


def load_swefixer_results(filepath: str) -> list:
    """Load SWE-Fixer results from JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            data['predicted_files'] = extract_files_from_patch(data.get('model_patch', ''))
            results.append(data)
    return results


def load_ground_truth(benchmark: str = 'lite') -> dict:
    """Load ground truth from SWE-Bench dataset."""
    if benchmark == 'lite':
        dataset = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    else:
        dataset = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    
    gt_dict = {}
    for instance in dataset:
        instance_id = instance['instance_id']
        patch = instance.get('patch', '')
        gt_files = extract_files_from_patch(patch)
        gt_dict[instance_id] = gt_files
    
    return gt_dict


def evaluate_swefixer(results: list, gt_dict: dict) -> dict:
    """
    Evaluate SWE-Fixer results at file level only.
    
    NOTE: SWE-Fixer does NOT make module or function level predictions.
    """
    file_scores = []
    per_instance = []
    
    for result in results:
        instance_id = result['instance_id']
        pred_files = result.get('predicted_files', [])
        gt_files = gt_dict.get(instance_id, [])
        
        if not gt_files:
            continue
            
        file_f1 = compute_file_f1_score(pred_files, gt_files)
        file_scores.append(file_f1)
        
        per_instance.append({
            'instance_id': instance_id,
            'file_f1': file_f1,
            'pred_files': pred_files,
            'gt_files': gt_files,
        })
    
    avg_file_f1 = sum(file_scores) / len(file_scores) if file_scores else 0.0
    
    return {
        'file_f1': avg_file_f1,
        'total_instances': len(results),
        'evaluated_instances': len(file_scores),
        'per_instance': per_instance,
    }


def main():
    # Path to SWE-Fixer results
    swefixer_path = Path(__file__).parent.parent.parent / 'SWE-Fixer' / 'result' / 'lite_no_p2p_result.jsonl'
    
    if not swefixer_path.exists():
        print(f"Error: SWE-Fixer results not found at {swefixer_path}")
        sys.exit(1)
    
    print(f"Loading SWE-Fixer results from: {swefixer_path}")
    results = load_swefixer_results(swefixer_path)
    print(f"Loaded {len(results)} instances")
    
    print("\nLoading ground truth from SWE-Bench Lite...")
    gt_dict = load_ground_truth('lite')
    print(f"Loaded ground truth for {len(gt_dict)} instances")
    
    print("\nEvaluating SWE-Fixer on SWE-Bench Lite...")
    metrics = evaluate_swefixer(results, gt_dict)
    
    print("\n" + "="*60)
    print("SWE-Fixer Evaluation Results on SWE-Bench Lite")
    print("="*60)
    print(f"\nFile-level F1:     {metrics['file_f1']:.4f} ({metrics['evaluated_instances']} instances)")
    print(f"\nNOTE: SWE-Fixer does NOT make module-level or function-level predictions.")
    print("      It only performs file-level retrieval.")
    print(f"\nTotal instances:   {metrics['total_instances']}")
    
    # Show some examples
    print("\n" + "-"*60)
    print("Sample Results (first 5 instances):")
    print("-"*60)
    for inst in metrics['per_instance'][:5]:
        print(f"\n{inst['instance_id']}:")
        print(f"  File F1: {inst['file_f1']:.4f}")
        print(f"  Pred files: {inst['pred_files'][:3]}{'...' if len(inst['pred_files']) > 3 else ''}")
        print(f"  GT files: {inst['gt_files'][:3]}{'...' if len(inst['gt_files']) > 3 else ''}")
    
    # Save detailed results
    output_path = Path(__file__).parent.parent / 'evaluation_results' / 'swefixer_evaluation.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'method': 'SWE-Fixer',
            'model': 'Qwen2.5-7B (retriever) + Qwen2.5-72B (editor)',
            'benchmark': 'SWE-Bench Lite',
            'metrics': {
                'file_f1': metrics['file_f1'],
                'module_f1': None,  # Not available - SWE-Fixer doesn't predict modules
                'entity_f1': None,  # Not available - SWE-Fixer doesn't predict entities
            },
            'note': 'SWE-Fixer only performs file-level retrieval. Module and entity level predictions are not available.',
            'counts': {
                'total_instances': metrics['total_instances'],
                'evaluated_instances': metrics['evaluated_instances'],
            },
            'per_instance': metrics['per_instance'],
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
