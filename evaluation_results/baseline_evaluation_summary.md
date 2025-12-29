# Baseline Localization Evaluation Summary

This document summarizes the evaluation of baseline methods (LocAgent and SWE-Fixer) for code localization on SWE-Bench Lite.

## Overview

| Method | Model | File F1 | Module F1 | Entity/Function F1 | Notes |
|--------|-------|---------|-----------|-------------------|-------|
| LocAgent | Claude 3.5 | 0.4348 | 0.3402 | 0.1975 | Full multi-level localization |
| SWE-Fixer | Qwen2.5-7B/72B | 0.6917 | N/A | N/A | File-level only |

## LocAgent (arXiv:2503.09089)

**GitHub**: https://github.com/gersteinlab/LocAgent

### Approach
LocAgent uses a graph-guided LLM agent framework for code localization. It:
1. Parses codebases into directed heterogeneous graphs
2. Uses LLM agents to navigate and locate relevant code entities
3. Provides predictions at file, module (class), and function levels

### Results on SWE-Bench Lite (300 instances)

Using our F1 evaluation metrics:
- **File-level F1**: 0.4348
- **Module-level F1**: 0.3402 (297 instances with module GT)
- **Entity-level F1**: 0.1975 (297 instances with entity GT)

Using LocAgent's official Acc@k metrics:
| Level | Acc@1 | Acc@3 | Acc@5 | Acc@10 |
|-------|-------|-------|-------|--------|
| File | 0.7774 | 0.9197 | 0.9416 | - |
| Module | - | - | 0.8650 | 0.8759 |
| Function | - | - | 0.7336 | 0.7737 |

### Localization Output Format
LocAgent provides structured predictions:
```json
{
  "found_files": ["path/to/file.py"],
  "found_modules": ["path/to/file.py:ClassName"],
  "found_entities": ["path/to/file.py:ClassName.method_name"]
}
```

## SWE-Fixer (arXiv:2501.05040)

**GitHub**: https://github.com/InternLM/SWE-Fixer

### Approach
SWE-Fixer uses a pipeline-based approach with two components:
1. **Code File Retriever** (Qwen2.5-7B): BM25 + fine-tuned model for file-level retrieval
2. **Code Editor** (Qwen2.5-72B): Generates patches for identified files

**Important**: SWE-Fixer does NOT make module-level or function-level predictions. It only performs file-level retrieval.

### Results on SWE-Bench Lite (300 instances)

Using our F1 evaluation metrics:
- **File-level F1**: 0.6917
- **Module-level F1**: N/A (not predicted)
- **Entity-level F1**: N/A (not predicted)

### Why No Module/Function Predictions?
From the paper:
> "For the retrieval task, we use a coarse-to-fine strategy combining BM25 for initial file retrieval and a model to identify the defective files from the BM25 results."

The retriever only identifies which files need modification. The editor then generates patches directly without explicit module/function localization.

## Key Findings

1. **LocAgent provides full multi-level localization** (file, module, function) while **SWE-Fixer only provides file-level**.

2. **SWE-Fixer has higher file-level F1** (0.6917 vs 0.4348) but this is because it's evaluated on the final patch output, which may benefit from the editor's ability to correct retrieval errors.

3. **LocAgent's Acc@k metrics are much higher than F1** because Acc@k measures if any correct prediction is in top-k, while F1 penalizes both false positives and false negatives.

4. **For module/function level evaluation**, only LocAgent results are available. SWE-Fixer's architecture doesn't produce these predictions.

## Recommendations

1. For comparing module/function localization, use LocAgent as the baseline.
2. For file-level comparison, both methods can be used.
3. Consider that SWE-Fixer's file predictions come from the final patch, not intermediate retrieval.

## Files

- `locagent_evaluation.json`: Detailed LocAgent evaluation results
- `swefixer_evaluation.json`: Detailed SWE-Fixer evaluation results
- `../scripts/evaluate_locagent.py`: LocAgent evaluation script
- `../scripts/evaluate_swefixer.py`: SWE-Fixer evaluation script
