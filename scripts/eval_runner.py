#!/usr/bin/env python3
"""
Evaluation runner for agentic code search benchmark.

This script adds the benchmarks submodule to sys.path and runs the
agentic_code_search evaluation from the benchmarks package.

Usage:
    python scripts/eval_runner.py --dataset_file <path> --llm-config-path <path> [options]

Example:
    python scripts/eval_runner.py \
        --dataset_file ./data/test.jsonl \
        --llm-config-path ./configs/llm.json \
        --output-dir ./outputs \
        --max-iterations 25 \
        --num-workers 4

For all available options, run:
    python scripts/eval_runner.py --help
"""

import sys
from pathlib import Path

# Add the benchmarks submodule to sys.path so we can import from it
_benchmarks_path = Path(__file__).parent.parent / "benchmarks"
sys.path.insert(0, str(_benchmarks_path.resolve()))

from benchmarks.agentic_code_search.run_infer import main

if __name__ == "__main__":
    main()
