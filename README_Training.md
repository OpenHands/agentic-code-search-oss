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
# Basic training with 4x H100 GPUs (runs in background)
modal run --detach scripts/modal_train.py --model Qwen/Qwen3-4B

# With smaller batch for testing
modal run --detach scripts/modal_train.py \
    --model Qwen/Qwen3-0.6B \
    --batch-size 2 \
    --n-rollouts 1

# With config overrides
modal run --detach scripts/modal_train.py \
    --model Qwen/Qwen3-4B \
    --extra-args "+generator.exp_config=configs/skyrl-experiments/read-only.yaml"

# View logs
modal app logs agentic-code-search-training
```

#### Download Results

```bash
# Download trained model
modal volume get code-search-checkpoints exported_model/ ./local_model/

# Download trajectories
modal volume get code-search-checkpoints trajectories/ ./trajectories/
```

#### GPU Options

| GPU Config | Description       | Use Case         |
| ---------- | ----------------- | ---------------- |
| `H100:4`   | 4x H100 (default) | Fastest training |
| `A100:4`   | 4x A100           | Good balance     |
| `A10G:2`   | 2x A10G           | Budget/testing   |
