# Baseline Localization Evaluation Summary

This document summarizes the evaluation of baseline methods (LocAgent and SWE-Fixer) for code localization on SWE-Bench Lite.

## Overview

| Method | Model | File F1 | File F1@1 | File Acc@1 | Module F1 | Entity F1 | Notes |
|--------|-------|---------|-----------|------------|-----------|-----------|-------|
| LocAgent | Claude 3.5 | 0.4348 | **0.7733** | 0.7733 | 0.3402 | 0.1975 | Full multi-level localization |
| SWE-Fixer | Qwen2.5-7B/72B | 0.6917 | 0.6900 | 0.6900 | N/A | N/A | File-level only |

**Key Insight**: LocAgent's F1 (0.4348) is lower than SWE-Fixer's (0.6917) because LocAgent over-predicts files (avg 4.09 files vs 1.0 GT). However, when using only the top-1 prediction (F1@1), LocAgent achieves **0.7733**, which is higher than SWE-Fixer.

## LocAgent (arXiv:2503.09089)

**GitHub**: https://github.com/gersteinlab/LocAgent

### Approach
LocAgent uses a graph-guided LLM agent framework for code localization. It:
1. Parses codebases into directed heterogeneous graphs
2. Uses LLM agents to navigate and locate relevant code entities
3. Provides predictions at file, module (class), and function levels

### Results on SWE-Bench Lite (300 instances)

#### F1 Scores (using all predictions)
| Level | F1 | Instances |
|-------|-----|-----------|
| File | 0.4348 | 300 |
| Module | 0.3402 | 297 |
| Entity | 0.1975 | 297 |

#### Acc@k Metrics (any GT in top-k)
| Level | Acc@1 | Acc@3 | Acc@5 | Acc@10 |
|-------|-------|-------|-------|--------|
| File | 0.7733 | 0.9133 | 0.9367 | 0.9433 |
| Module | 0.6599 | 0.8316 | 0.8721 | 0.8822 |
| Entity | 0.5084 | 0.7542 | 0.7980 | 0.8316 |

#### F1@k Scores (using only top-k predictions)
| Level | F1@1 | F1@3 | F1@5 | F1@10 |
|-------|------|------|------|-------|
| File | **0.7733** | 0.5144 | 0.4453 | 0.4348 |
| Module | **0.6453** | 0.4505 | 0.3716 | 0.3411 |
| Entity | **0.4778** | 0.3892 | 0.3008 | 0.2216 |

### Localization Output Format
LocAgent provides structured predictions (ordered by relevance):
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

#### File-level Metrics
| Metric | Value |
|--------|-------|
| F1 | 0.6917 |
| Acc@1 | 0.6900 |
| Recall@1 | 0.6900 |

Note: SWE-Fixer typically predicts only 1 file per instance (from the final patch), so F1@k metrics are nearly identical across all k values.

### Why No Module/Function Predictions?
From the paper:
> "For the retrieval task, we use a coarse-to-fine strategy combining BM25 for initial file retrieval and a model to identify the defective files from the BM25 results."

The retriever only identifies which files need modification. The editor then generates patches directly without explicit module/function localization.

## Key Findings

1. **LocAgent provides full multi-level localization** (file, module, function) while **SWE-Fixer only provides file-level**.

2. **LocAgent over-predicts files** (avg 4.09 files per instance vs 1.0 GT), which hurts F1 but not Acc@k.

3. **Using top-1 predictions improves LocAgent's F1 significantly**:
   - File F1@1: 0.7733 (vs 0.4348 with all predictions)
   - Module F1@1: 0.6453 (vs 0.3402)
   - Entity F1@1: 0.4778 (vs 0.1975)

4. **SWE-Fixer's file predictions come from the final patch**, not intermediate retrieval, which may benefit from the editor's ability to correct retrieval errors.

5. **For module/function level evaluation**, only LocAgent results are available. SWE-Fixer's architecture doesn't produce these predictions.

## Recommendations

1. For comparing module/function localization, use LocAgent as the baseline.
2. For file-level comparison, both methods can be used.
3. Consider using F1@1 or F1@k metrics when comparing methods that produce ranked predictions.
4. LocAgent's predictions could potentially be improved by thresholding to top-k predictions.

## Files

- `locagent_evaluation.json`: Detailed LocAgent evaluation results
- `swefixer_evaluation.json`: Detailed SWE-Fixer evaluation results
- `../scripts/evaluate_locagent.py`: LocAgent evaluation script
- `../scripts/evaluate_swefixer.py`: SWE-Fixer evaluation script
