#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=300Gb
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:A100_80GB:8
#SBATCH -t 2-00:00:00
#SBATCH --job-name=rl_qwen3_4b_batched
#SBATCH --error=/data/user_data/sanidhyv/agentic-code-search-oss/logs/%x__%j.err
#SBATCH --output=/data/user_data/sanidhyv/agentic-code-search-oss/logs/%x__%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# ============================================================================
# Batched Training Script with Memory-Optimized Indexing
# ============================================================================
# This script trains in batches:
# 1. Index N repos (e.g., 15)
# 2. Train on those repos
# 3. Clean up indices
# 4. Move to next batch
# ============================================================================

# Cache Configuration
export UV_CACHE_DIR="/data/user_data/sanidhyv/.cache/uv"
export HF_HOME="/data/user_data/sanidhyv/.cache/huggingface"
export TRANSFORMERS_CACHE="/data/user_data/sanidhyv/.cache/transformers"
export TORCH_HOME="/data/user_data/sanidhyv/.cache/torch"
export XDG_CACHE_HOME="/data/user_data/sanidhyv/.cache"
export TMPDIR="/data/user_data/sanidhyv/tmp"
export RAY_TMPDIR="/data/user_data/sanidhyv/ray_temp_grep"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$TMPDIR" "$RAY_TMPDIR"

# NCCL Configuration
NETWORK_INTERFACE=$(ip route | grep default | awk '{print $5}' | head -n1)
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export CODE_SEARCH_BASE_PATH="/data/user_data/sanidhyv/agentic-code-search-oss"

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=OFF
export VLLM_FLASH_ATTN_VERSION=2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # ← CRITICAL for fragmentation


# Load .env if exists
[ -f .env ] && . .env

# ============================================================================
# Configuration
# ============================================================================
MODEL="Qwen/Qwen3-4B"
MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
DATA_PATH="${DATA_PATH:-data/SWE-Gym__SWE-Gym_train}"
CKPT_PATH="/data/user_data/sanidhyv/agentic-code-search-oss/ckpts/code_search_${MODEL_ALIAS}_batched"
CACHE_DIR="/data/user_data/sanidhyv/tmp/embedding_cache"

# Batched indexing configuration
BATCH_SIZE=15  # Number of repos per batch
REPOS_DIR="/data/user_data/sanidhyv/grep"  # Where repos are cloned

# Training configuration
N_ROLLOUTS="${N_ROLLOUTS:-4}"
TRAIN_BATCH_SIZE=4
MAX_LENGTH=2048
export WANDB_API_KEY="bd054e89bc6dc33ce731d090da4a87bffa973032"
export WANDB_PROJECT="code_search"

# Resource allocation
NUM_GPUS=8
NNODES=1
HALF_NUM_GPUS=$((NUM_GPUS / 2))
NUM_INFERENCE_ENGINES=4  # Half for inference
NUM_TRAINING_ENGINES=4   # Half for training
TP_SIZE=1  
LOGGER=wandb
RUN_NAME="code_search_${MODEL_ALIAS}_batched_b${BATCH_SIZE}"

mkdir -p $CKPT_PATH $CKPT_PATH/trajectories logs $CACHE_DIR
export RAY_object_store_memory=$((50 * 1024 * 1024 * 1024))  # 50GB
export RAY_memory_monitor_refresh_ms=0  
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/data/user_data/sanidhyv/ray_spill"}}'
export HYDRA_FULL_ERROR=1

mkdir -p /data/user_data/sanidhyv/ray_spill
rm -r ckpts
# ============================================================================
# Pre-Index First Batch (GPU)
# ============================================================================
echo "=================================================="
echo "Pre-indexing first batch on GPU"
echo "=================================================="
echo "This uses GPU for fast embedding generation"
echo "Training will use CPU for retrieval (no GPU contention)"
echo "=================================================="

# # Clone repos if needed
# if [ ! -d "$REPOS_DIR" ] || [ -z "$(ls -A $REPOS_DIR)" ]; then
#     echo "Cloning repos..."
#     uv run python scripts/clone_and_index_repos.py \
#         --output-dir "$REPOS_DIR" \
#         --cache-dir "$CACHE_DIR" \
#         --dataset "SWE-Gym/SWE-Gym" \
#         --split train \
#         --max-repos "$BATCH_SIZE" \
#         --skip-indexing  # Only clone, indexing done separately
# fi

# # Index first batch on GPU
# echo "Indexing batch 0 on GPU..."
# uv run python scripts/index_batch.py \
#     --batch-idx 0 \
#     --batch-size "$BATCH_SIZE" \
#     --cache-dir "$CACHE_DIR" \
#     --repos-dir "$REPOS_DIR"

# echo "First batch indexed successfully!"

# ============================================================================
# Training Loop with Batched Indexing
# ============================================================================
echo "=================================================="
echo "Starting Batched Training"
echo "=================================================="
echo "Model: $MODEL"
echo "Batch Size: $BATCH_SIZE repos/batch"
echo "N Rollouts: $N_ROLLOUTS"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "=================================================="

# Cleanup function
cleanup() {
    python3 -c "import ray; ray.shutdown() if ray.is_initialized() else None" 2>/dev/null
    rm -rf "$TMPDIR"/* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

set -x

# Launch training with batched indexing
CUDA_LAUNCH_BLOCKING=1 uv run python src/train_batched.py \
  --config-name=ppo_base_config \
  +run_async_trainer=true \
  +batched_indexing.enabled=true \
  +batched_indexing.batch_size=$BATCH_SIZE \
  +batched_indexing.cache_dir=$CACHE_DIR \
  +batched_indexing.repos_dir=$REPOS_DIR \
  data.train_data=["$DATA_PATH/train.parquet"] \
  data.val_data=["$DATA_PATH/validation.parquet"] \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path=$MODEL \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  trainer.policy.fsdp_config.fsdp_size=-1 \
  trainer.fully_async.num_parallel_generation_workers=4 \
  trainer.placement.policy_num_gpus_per_node=$NUM_TRAINING_ENGINES \
  trainer.placement.ref_num_gpus_per_node=$NUM_TRAINING_ENGINES \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=1 \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  +generator.traj_dir=$CKPT_PATH/trajectories/ \
  +generator.engine_init_kwargs.enable_auto_tool_choice=true \
  +generator.engine_init_kwargs.tool_call_parser=hermes \
  +generator.engine_init_kwargs.reasoning_parser=qwen3 \
  +generator.engine_init_kwargs.max_model_len=16384 \
  trainer.epochs=20 \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.export_path="$CKPT_PATH/exported_model/" \
  trainer.hf_save_interval=-1 \
  trainer.ckpt_interval=100 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=$MAX_LENGTH \
  generator.sampling_params.temperature=1.0 \
  generator.max_input_length=24000 \
  generator.max_num_batched_tokens=48000 \
  generator.max_turns=20 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=False \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host=0.0.0.0 \
  generator.http_endpoint_port=8080 \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=$N_ROLLOUTS \
  generator.gpu_memory_utilization=0.7 \
  generator.enforce_eager=false \
  trainer.step_wise_training=true \
  trainer.logger=$LOGGER \
  trainer.project_name=$WANDB_PROJECT \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  trainer.ckpt_path=$CKPT_PATH \
  trainer.max_ckpts_to_keep=3 \
  +semantic_search.enabled=true \
  +semantic_search.device=cpu \
  +semantic_search.embedding_model="jinaai/jina-code-embeddings-0.5b" \
  +semantic_search.reranker_model=null \
  +semantic_search.max_indices=15

# ============================================================================
# Cleanup
# ============================================================================
echo "=================================================="
echo "Training Complete - Cleaning Up"
echo "=================================================="

# Clean up temporary files
echo "Cleaning up temporary files..."
find /data/user_data/sanidhyv/tmp -maxdepth 1 -type d -name "testbed_*" -mtime +1 -exec rm -rf {} + 2>/dev/null
echo "Cleanup complete!"

exit $?