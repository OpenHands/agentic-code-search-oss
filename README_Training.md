# Training

## Build Dataset

```
uv run src/build_dataset.py --output ../data/
```

## Train Model

Training is driven by Hydra configs under `config/` (see `config/base.yaml` defaults). Run `src.train` and select an experiment via `experiment=...`.
Most configuration options follow the SkyRL config schema; see [SkyRL configuration docs](https://skyrl.readthedocs.io/en/latest/configuration/config.html) for details.
See [Hydra docs](https://hydra.cc/docs/intro/) for more details.

### Minimal for local testing

```
uv run -m src.train model=Qwen/Qwen3-0.6B
```

### Async training

```
uv run -m src.train model=Qwen/Qwen3-4B training=async
```

### Async training with LoRA experiment config

```
uv run -m src.train model=Qwen/Qwen3-4B training=async experiment=lora
```

### Multirun
Note: the `none` experiment is a placeholder to allow multiruns including the baseline setting.

```
uv run -m src.train -m model=Qwen/Qwen3-4B training=async experiment=none,lora
```

### Slurm example (Submitit launcher)

```
uv run -m src.train model=Qwen/Qwen3-4B training=async platform=babel hydra/launcher=slurm
```

### Add a new experiment

Create `config/experiment/my_exp.yaml`:

```yaml
# @package _global_
trainer:
  logger: wandb
```

Then run with `experiment=my_exp`.