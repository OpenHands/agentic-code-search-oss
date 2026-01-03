# Training

## Build Dataset

```
uv run src/build_dataset.py --output ../data/
```

## Train Model with RLVR

```
bash scripts run_training.sh -m Qwen/Qwen3-0.6B -d <Absolute Path to Data>
```

```
DATA_PATH=<Absolute Path to Data>
bash scripts/run_async_training.sh -m Qwen/Qwen3-4B -d $DATA_PATH 2>&1 | tee training.log
```

```
DATA_PATH=<Absolute Path to Data>
bash scripts/run_async_training.sh \
    -m Qwen/Qwen3-4B \
    -o "+generator.exp_config=configs/skyrl-experiments/read-only.yaml" \
    -d $DATA_PATH \
    2>&1 | tee training.log
```

## Train Model with On-Policy Distillation

```
DATA_PATH=<Absolute Path to Data>
bash scripts/run_distillation.sh \
    -m Qwen/Qwen3-VL-4B-Instruct \  # Student model (model to be trained)
    -r Qwen/Qwen3-VL-32B-Instruct \     # Teacher model (model to distill from)
    -d $DATA_PATH \
    2>&1 | tee distillation.log
```

```
DATA_PATH=<Absolute Path to Data>
bash scripts/run_distillation.sh \
    -m Qwen/Qwen3-0.6B \  # Student model (model to be trained)
    -r Qwen/Qwen3-4B \     # Teacher model (model to distill from)
    -o "+generator.exp_config=configs/skyrl-experiments/read-only.yaml" \
    -d $DATA_PATH \
    2>&1 | tee distillation.log
```

