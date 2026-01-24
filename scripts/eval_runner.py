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

import subprocess
import sys
from pathlib import Path
from types import ModuleType

# Add the benchmarks submodule to sys.path so we can import from it
_benchmarks_path = Path(__file__).parent.parent / "benchmarks"
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_benchmarks_path.resolve()))


def _get_sdk_sha_from_parent_repo() -> str:
    """Get SDK SHA from the parent repo's software-agent-sdk submodule."""
    sdk_path = _project_root / "software-agent-sdk"
    try:
        result = subprocess.run(
            ["git", "submodule", "status", str(sdk_path)],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(_project_root),
        )
        sha = result.stdout.strip().split()[0].lstrip("+-")
        return sha
    except Exception:
        # Fallback if git command fails
        return "unknown"


# Pre-create the version module with our SDK SHA before benchmarks imports it
_sdk_sha = _get_sdk_sha_from_parent_repo()
_version_module = ModuleType("benchmarks.utils.version")
_version_module.SDK_SHA = _sdk_sha
_version_module.SDK_SHORT_SHA = _sdk_sha[:7] if _sdk_sha != "unknown" else "unknown"
_version_module.PROJECT_ROOT = _benchmarks_path
sys.modules["benchmarks.utils.version"] = _version_module

from benchmarks.agentic_code_search.run_infer import main

if __name__ == "__main__":
    main()
