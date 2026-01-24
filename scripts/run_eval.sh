#!/bin/bash
#
# run_eval.sh - Wrapper script to run the evaluation runner with uv
#
# Usage:
#   ./scripts/run_eval.sh [OPTIONS]
#
# Example usage:
#   ./scripts/run_eval.sh \
#     --dataset_file benchmarks/gt_location.jsonl \
#     --llm-config-path configs/llm_config.json \
#     --max-iterations 10 \
#     --num-workers 1 \
#     --tools terminal
#
# Options are passed through to scripts/eval_runner.py
# Run with --help to see all available options:
#   ./scripts/run_eval.sh --help
#

uv run python scripts/eval_runner.py "$@"
