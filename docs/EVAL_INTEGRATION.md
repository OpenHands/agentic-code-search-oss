# Evaluation Integration Documentation

This document explains how to run evaluations for code localization agents using the integrated benchmarks system.

## Quick Start

### 1. Start a Local Model with vLLM

Start vLLM with tool calling enabled:

```bash
# For a small model (quick testing)
uv run vllm serve Qwen/Qwen3-4B \
  --port 8000 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

### 2. Create LLM Config

```bash
mkdir -p configs
cat > configs/llm_config.json << 'EOF'
{
  "model": "openai/Qwen/Qwen3-4B",
  "api_key": "dummy",
  "base_url": "http://localhost:8000/v1",
  "temperature": 0.0
}
EOF
```

**Important:** The model name must be prefixed with `openai/` to tell litellm it's an OpenAI-compatible endpoint.

### 3. Run Evaluation

```bash
./scripts/run_eval.sh \
    --dataset_file benchmarks/gt_location.jsonl \
    --llm-config-path configs/llm_config.json \
    --system_prompt_file benchmarks/benchmarks/agentic_code_search/prompts/system_prompt.j2 \
    --user_prompt_file benchmarks/benchmarks/agentic_code_search/prompts/file_module_short.j2 \
    --tools terminal \
    --max-iterations 10 \
    --num-workers 1 \
    --output-dir ./agentic_code_search_outputs \
    --n-limit 1 \
    --workspace_base_dir /tmp/testbed/
```

**Key options:**
- `--n-limit 1` - Run on 1 instance (remove for full dataset)
- `--num-workers 1` - Parallel workers (increase for faster eval)
- `--max-iterations 10` - Max agent steps per instance

### 4. Check Results

```bash
# View full output
cat ./agentic_code_search_outputs/agentic_code_search_gt_location/openai/Qwen/Qwen3-4B_sdk_*/output.jsonl | jq .

# View just the reward scores
cat ./agentic_code_search_outputs/agentic_code_search_gt_location/openai/Qwen/Qwen3-4B_sdk_*/output.jsonl | jq '.test_result.reward'
```

### Example Output

```json
{
  "file_reward": 0.5,
  "module_reward": 0.5,
  "entity_reward": 0.4,
  "prediction": {
    "files": ["sklearn/calibration.py", "sklearn/_config.py", "sklearn/isotonic.py"],
    "modules": ["sklearn/calibration.py:_CalibratedClassifier", "sklearn/_config.py:set_config", "sklearn/isotonic.py:IsotonicRegression"],
    "entities": ["sklearn/isotonic.py:IsotonicRegression.predict", "sklearn/_config.py:set_config", "sklearn/calibration.py:_CalibratedClassifier.predict_proba"]
  },
  "ground_truth": {
    "files": ["sklearn/isotonic.py"],
    "modules": ["sklearn/isotonic.py:IsotonicRegression"],
    "entities": ["sklearn/isotonic.py:IsotonicRegression.predict", "sklearn/isotonic.py:IsotonicRegression.transform"]
  }
}
```

**Metrics explained:**
- **file_reward** - F1 score for file-level localization
- **module_reward** - F1 score for class-level localization  
- **entity_reward** - F1 score for function/method-level localization

---

## Implementation Details

### Goal

Integrate evaluation code from the [benchmarks repo](https://github.com/adityasoni9998/benchmarks/tree/agentic_code_search) into this repository to enable end-to-end training AND evaluation of code localization agents.

**Key requirements:**

- Run trained models on SWE-Bench Pro/Verified benchmarks
- Use the same `software-agent-sdk` for both training and evaluation
- No dependency conflicts with existing SkyRL training setup

### The Problem

The benchmarks repo is designed as a standalone project with its own workspace pointing to `vendor/software-agent-sdk/`. Directly integrating it as a workspace member caused:

1. **Nested workspace error** - uv doesn't support workspaces inside workspaces
2. **Dependency conflicts** - `commit0` requires `datasets==3.0.1`, we need `>=4.0.0`

### The Solution: Runtime sys.path Manipulation

Instead of making benchmarks a proper package in our workspace, we use Python's `sys.path` to import it at runtime:

```python
import sys
sys.path.insert(0, "/path/to/benchmarks")

# Now imports work - and they use OUR installed SDK
from benchmarks.agentic_code_search.run_infer import main
```

**Why this works:**

- When benchmarks code imports `openhands.sdk`, Python searches `sys.path`
- Our SDK packages are already installed via uv workspace
- Python finds our SDK first, not benchmarks' vendor/ (which doesn't exist anyway)

### Version Module Patching

The benchmarks code has a `version.py` that tries to get the SDK SHA from `vendor/software-agent-sdk` (which doesn't exist in our setup). The `eval_runner.py` script pre-creates this module with the SHA from our repo's SDK:

```python
# Pre-create the version module with our SDK SHA before benchmarks imports it
_sdk_sha = _get_sdk_sha_from_parent_repo()
_version_module = ModuleType("benchmarks.utils.version")
_version_module.SDK_SHA = _sdk_sha
_version_module.SDK_SHORT_SHA = _sdk_sha[:7]
sys.modules["benchmarks.utils.version"] = _version_module
```

### Files Added/Modified

| File                     | Description                                                             |
| ------------------------ | ----------------------------------------------------------------------- |
| `benchmarks/`            | Git submodule pointing to adityasoni9998/benchmarks@agentic_code_search |
| `.gitmodules`            | Submodule configuration                                                 |
| `pyproject.toml`         | Added jinja2, pandas, tqdm, lmnr dependencies                           |
| `scripts/eval_runner.py` | Python wrapper that sets up sys.path and runs eval                      |
| `scripts/run_eval.sh`    | Shell wrapper for `uv run`                                              |

### Architecture

```
agentic-code-search-oss/
├── software-agent-sdk/          # Our SDK (used for training AND eval)
│   ├── openhands-sdk/
│   ├── openhands-tools/
│   └── ...
├── benchmarks/                   # Submodule (NOT in workspace)
│   └── benchmarks/
│       └── agentic_code_search/
│           ├── run_infer.py      # Main eval script
│           ├── eval_infer.py     # Results aggregator
│           └── prompts/          # Jinja2 templates
├── scripts/
│   ├── eval_runner.py            # sys.path wrapper
│   └── run_eval.sh               # Shell wrapper
└── src/                          # Training code (unchanged)
```

### How Evaluation Works

```
┌─────────────────┐
│  run_eval.sh    │
└────────┬────────┘
         │ uv run
         ▼
┌─────────────────┐
│ eval_runner.py  │
│                 │
│ sys.path.insert │
│ (benchmarks/)   │
└────────┬────────┘
         │ import
         ▼
┌─────────────────────────────────┐
│ benchmarks.agentic_code_search  │
│                                 │
│ from openhands.sdk import ...   │──► Uses OUR SDK
└─────────────────────────────────┘
```

### Learnings

1. **uv workspaces don't nest** - Can't add a package with its own workspace as a member
2. **sys.path manipulation is clean** - Keeps submodule pristine, easy to update
3. **Python import resolution** - First match in sys.path wins, so our installed SDK is used
4. **Dependency isolation** - We only add deps we actually need, avoiding conflicts
5. **Version module patching** - Pre-create the version module to use our repo's SDK SHA
6. **litellm provider prefix** - Local vLLM endpoints need `openai/` prefix in model name
7. **vLLM tool calling** - Requires `--enable-auto-tool-choice --tool-call-parser hermes` flags

---

## Troubleshooting

### "LLM Provider NOT provided"

Add `openai/` prefix to your model name in `llm_config.json`:
```json
{"model": "openai/Qwen/Qwen3-4B", ...}
```

### "auto tool choice requires --enable-auto-tool-choice"

Restart vLLM with tool calling flags:
```bash
uv run vllm serve Qwen/Qwen3-4B \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### "Processing 0 instances"

Previous failed runs left stale output. Delete the output directory:
```bash
rm -rf ./agentic_code_search_outputs/
```

### Import errors from benchmarks

Ensure the submodule is initialized:
```bash
git submodule update --init --recursive
```
