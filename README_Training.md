# Training

## Build Dataset

```bash
uv run src/build_dataset.py --output ../data/
```

## Train Model

### Option 1: Local GPU Training

If you have a machine with GPUs available:

```bash
bash scripts/run_training.sh -m Qwen/Qwen3-0.6B -d <Absolute Path to Data>
```

```bash
DATA_PATH=<Absolute Path to Data>
bash scripts/run_async_training.sh -m Qwen/Qwen3-4B -d $DATA_PATH 2>&1 | tee training.log
```

```bash
DATA_PATH=<Absolute Path to Data>
bash scripts/run_async_training.sh \
    -m Qwen/Qwen3-4B \
    -o "+generator.exp_config=configs/skyrl-experiments/read-only.yaml" \
    -d $DATA_PATH \
    2>&1 | tee training.log
```

### Option 2: Modal Cloud GPUs

Run training on [Modal](https://modal.com) cloud GPUs. Works from any machine (including Mac without CUDA).

#### One-time Setup

1. **Install Modal CLI** (no CUDA/Docker needed locally):

```bash
# Create a minimal environment for Modal
python -m venv .modal-venv
source .modal-venv/bin/activate
pip install modal

# Authenticate with Modal (opens browser)
modal setup
```

2. **Create Modal Secrets**:

```bash
# WandB for experiment tracking
modal secret create wandb-secret WANDB_API_KEY=<your-wandb-key>

# HuggingFace for model downloads
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>
```

3. **Upload Training Data** to Modal Volume:

```bash
modal run scripts/modal_train.py::upload_data_from_local
```

4. **Validate Setup** (runs on cheap CPU, no GPU costs):

```bash
modal run scripts/modal_train.py::validate_setup
```

#### Run Training

```bash
# Start a new experiment (will prompt for confirmation if run exists)
modal run scripts/modal_train.py --run-name exp-v1 --fresh

# Resume an existing experiment
modal run scripts/modal_train.py --run-name exp-v1

# Quick test: small batch, limited steps
modal run scripts/modal_train.py \
    --run-name test-run \
    --model Qwen/Qwen3-0.6B \
    --batch-size 2 \
    --max-steps 20 \
    --fresh

# Run in background (detached) with --force to skip confirmation
modal run --detach scripts/modal_train.py \
    --run-name exp-v1 \
    --force

# With config overrides
modal run scripts/modal_train.py \
    --run-name exp-readonly \
    --extra-args "+generator.exp_config=configs/skyrl-experiments/read-only.yaml"

# View logs
modal app logs agentic-code-search-training

# List existing runs
modal run scripts/modal_train.py::list_runs
```

#### Command Flags

| Flag           | Default         | Description                                        |
| -------------- | --------------- | -------------------------------------------------- |
| `--run-name`   | (from model)    | Experiment name, organizes checkpoints by run      |
| `--model`      | `Qwen/Qwen3-4B` | HuggingFace model path                             |
| `--n-rollouts` | `8`             | Number of rollouts per prompt                      |
| `--batch-size` | `8`             | Training batch size                                |
| `--max-length` | `8192`          | Maximum generation length                          |
| `--max-steps`  | `0` (unlimited) | Limit training to N steps (useful for quick tests) |
| `--fresh`      | `False`         | Start fresh, ignore previous checkpoints           |
| `--force`      | `False`         | Skip confirmation prompts (needed for `--detach`)  |
| `--extra-args` | `""`            | Additional Hydra config overrides                  |

#### Checkpoint Lifecycle

Each run's state is stored in its own subdirectory: `code-search-checkpoints/{run-name}/`

```
code-search-checkpoints/
├── exp-v1/
│   ├── global_step_100/
│   ├── global_step_200/
│   ├── exported_model/
│   └── trajectories/
├── exp-v2/
│   └── ...
```

- **Checkpoints** (`global_step_X/`): Model weights and optimizer state, saved every N steps
- **Trajectories** (`trajectories/`): Agent interaction traces from rollout generation
- **Exported Model** (`exported_model/`): HuggingFace-compatible model for inference

**Resume behavior**: Without `--fresh`, training resumes from the latest checkpoint in that run.

**Fresh start**: With `--fresh`, checkpoints are ignored but files remain. To fully delete a run:

```bash
# Delete a specific run
modal volume rm code-search-checkpoints exp-v1 --recursive

# Clear all runs
modal volume rm code-search-checkpoints --recursive
```

#### Download Results

```bash
# Download trained model from a specific run
modal volume get code-search-checkpoints exp-v1/exported_model/ ./local_model/

# Download trajectories
modal volume get code-search-checkpoints exp-v1/trajectories/ ./trajectories/
```

#### GPU Options

| GPU Config | Description       | Use Case         |
| ---------- | ----------------- | ---------------- |
| `H100:4`   | 4x H100 (default) | Fastest training |
| `A100:4`   | 4x A100           | Good balance     |
| `A10G:2`   | 2x A10G           | Budget/testing   |
