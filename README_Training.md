# Training

## Build Dataset

```bash
uv run src/build_dataset.py --output ../data/
```

## Train Model

### Basic Training

```bash
bash scripts/run_training.sh -m Qwen/Qwen3-0.6B -d <Absolute Path to Data>
```

### Async Training

```bash
DATA_PATH=<Absolute Path to Data>
bash scripts/run_async_training.sh -m Qwen/Qwen3-4B -d $DATA_PATH 2>&1 | tee training.log
```

### With Custom Config

```bash
DATA_PATH=<Absolute Path to Data>
bash scripts/run_async_training.sh \
    -m Qwen/Qwen3-4B \
    -o "+generator.exp_config=configs/skyrl-experiments/read-only.yaml" \
    -d $DATA_PATH \
    2>&1 | tee training.log
```

## Model Path Formats

The `-m` parameter supports multiple formats:
- **HuggingFace model ID**: `Qwen/Qwen2.5-7B-Instruct`
- **Absolute path**: `/mnt/models/qwen3-4b`
- **Relative path**: `./models/qwen3-4b` or `~/models/qwen3-4b`

> **Note**: For relative paths without `./` or `~/` prefix (e.g., `models/qwen`), the system will check if the path exists locally; otherwise, it will attempt to download from HuggingFace Hub.