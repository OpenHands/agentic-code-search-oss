# Evaluation Integration Documentation

## Goal

Integrate evaluation code from the [benchmarks repo](https://github.com/adityasoni9998/benchmarks/tree/agentic_code_search) into this repository to enable end-to-end training AND evaluation of code localization agents for the ICML submission.

**Key requirements:**
- Run trained models on SWE-Bench Pro/Verified benchmarks
- Use the same `software-agent-sdk` for both training and evaluation
- No dependency conflicts with existing SkyRL training setup

## Solution Approach

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

## Files Added/Modified

| File | Description |
|------|-------------|
| `benchmarks/` | Git submodule pointing to adityasoni9998/benchmarks@agentic_code_search |
| `.gitmodules` | Submodule configuration |
| `pyproject.toml` | Added jinja2, pandas, tqdm, lmnr dependencies |
| `scripts/eval_runner.py` | Python wrapper that sets up sys.path and runs eval |
| `scripts/run_eval.sh` | Shell wrapper for `uv run` |

## Architecture

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

## How Evaluation Works

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

## Learnings

1. **uv workspaces don't nest** - Can't add a package with its own workspace as a member
2. **sys.path manipulation is clean** - Keeps submodule pristine, easy to update
3. **Python import resolution** - First match in sys.path wins, so our installed SDK is used
4. **Dependency isolation** - We only add deps we actually need, avoiding conflicts

## What to Test

### On Linux with CUDA (training machine)

1. **Sync dependencies:**
   ```bash
   uv sync
   ```

2. **Test import works:**
   ```bash
   uv run python -c "
   import sys
   sys.path.insert(0, 'benchmarks')
   from benchmarks.agentic_code_search.run_infer import main
   print('Import successful!')
   "
   ```

3. **Run a minimal evaluation:**
   ```bash
   # Create LLM config file first
   cat > configs/llm_config.json << 'EOF'
   {
     "model": "openai/gpt-4o-mini",
     "api_key": "your-api-key",
     "base_url": "https://api.openai.com/v1",
     "temperature": 0.0
   }
   EOF

   # Run eval on 1 instance
   ./scripts/run_eval.sh \
     --dataset_file benchmarks/gt_location.jsonl \
     --llm-config-path configs/llm_config.json \
     --max-iterations 10 \
     --num-workers 1 \
     --tools terminal \
     --n-limit 1
   ```

4. **Verify training still works:**
   ```bash
   # Your existing training command should work unchanged
   bash scripts/run_async_training.sh -m Qwen/Qwen3-4B -d $DATA_PATH
   ```

### Expected Output Format

The evaluation produces JSONL output with F1 scores for:
- **File-level**: Did the agent find the correct files?
- **Module-level**: Did it find the correct classes?
- **Entity-level**: Did it find the correct functions/methods?

Example output:
```json
{
  "instance_id": "astropy__astropy-12907",
  "test_result": {
    "reward": {
      "file_reward": 1.0,
      "module_reward": 0.8,
      "entity_reward": 0.6
    },
    "raw_prediction": "astropy/modeling/separable.py\nfunction: _cstack",
    "wall_time_seconds": 45.2,
    "num_steps": 5,
    "num_tool_calls": 12
  }
}
```

## Next Steps

1. **Test on training machine** - Verify uv sync works with CUDA deps
2. **Prepare SWE-Bench Pro/Verified datasets** - May need to download separately
3. **Run base model evals** - Establish baseline before training
4. **Integrate with training loop** - Optional: run evals at checkpoints

## References

- [Benchmarks repo](https://github.com/adityasoni9998/benchmarks/tree/agentic_code_search)
- [Original Slack conversation](#) - Aditya's integration instructions
- [SWE-Bench](https://www.swebench.com/) - Benchmark website
