# Training

## Build Dataset

```
uv run src/build_dataset.py --output ../data/
```

## Train Model

```
bash scripts run_training.sh -m Qwen/Qwen3-0.6B -d <Absolute Path to Data>
```

### Train with vector search

```
bash scripts train_with_vector_search.sh
```
- Models, locations for storing embeddings/data can be configured within the script.
- Embedding models can be modified in src/mcp_server/training_semantic_search_server.py

