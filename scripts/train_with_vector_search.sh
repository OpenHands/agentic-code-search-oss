#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=300Gb
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:8
#SBATCH -t 2-00:00:00
#SBATCH --job-name=rl_qwen3_8b
#SBATCH --error=/home/sanidhyv/agentic-code-search-oss/logs/%x__%j.err
#SBATCH --output=/home/sanidhyv/agentic-code-search-oss/logs/%x__%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

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
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export CODE_SEARCH_BASE_PATH="/home/sanidhyv/agentic-code-search-oss"

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=OFF

# Load .env if exists
[ -f .env ] && . .env

# Configuration
MODEL="Qwen/Qwen3-4B"
MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
DATA_PATH="${DATA_PATH:-data/SWE-Gym__SWE-Gym_train}"
CKPT_PATH="/data/user_data/sanidhyv/grep/train"
N_ROLLOUTS="${N_ROLLOUTS:-4}"
export WANDB_API_KEY=""
export WANDB_PROJECT="grep"

# Resource allocation 
NUM_GPUS=8
NNODES=1
NUM_INFERENCE_ENGINES=8  
TP_SIZE=1  
LOGGER=wandb
RUN_NAME="code_search_${MODEL_ALIAS}"

mkdir -p $CKPT_PATH $CKPT_PATH/trajectories logs
export RAY_object_store_memory=$((50 * 1024 * 1024 * 1024))  # 50GB
export RAY_memory_monitor_refresh_ms=0  
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/data/user_data/sanidhyv/ray_spill"}}'

mkdir -p /data/user_data/sanidhyv/ray_spill

echo "Starting RL Training (Working Config)"
echo "Model: $MODEL"
echo "N Rollouts: $N_ROLLOUTS"
echo "======================================"

# Cleanup
cleanup() {
    python3 -c "import ray; ray.shutdown() if ray.is_initialized() else None" 2>/dev/null
    rm -rf "$TMPDIR"/* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

set -x

# Launch training 
CUDA_LAUNCH_BLOCKING=1 uv run --isolated -m src.train \
  data.train_data=["data/SWE-Gym__SWE-Gym_train/train.parquet"] \
  data.val_data=["data/SWE-Gym__SWE-Gym_train/validation.parquet"] \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path=Qwen/Qwen3-4B \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.policy.sequence_parallel_size=8 \
  generator.num_inference_engines=8 \
  generator.inference_engine_tensor_parallel_size=1 \
  +generator.traj_dir=/data/user_data/sanidhyv/grep/train/trajectories/ \
  trainer.epochs=20 \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=100 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=2048 \
  generator.max_input_length=14000 \
  generator.max_num_batched_tokens=36000 \
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
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.4 \
  trainer.logger=wandb \
  trainer.project_name=code_search \
  trainer.run_name=code_search_Qwen-Qwen3-4B \
  trainer.resume_mode=null \
  trainer.ckpt_path=/data/user_data/sanidhyv/grep/train \
  +generator.semantic_search=true \
  +generator.engine_init_kwargs="{enable_auto_tool_choice:true,tool_call_parser:hermes,max_model_len:16384}"

CACHE_DIR="/data/user_data/sanidhyv/tmp/embedding_cache"
MAX_AGE_DAYS=7
# Clean up temporary files from training/eval
echo "Cleaning up temporary files..."
# Remove old workspaces (testbed_*)
echo "Removing testbed workspaces..."
find /data/user_data/sanidhyv/tmp -maxdepth 1 -type d -name "testbed_*" -mtime +1 -exec rm -rf {} + 2>/dev/null
echo "Testbed cleanup complete"

# Remove old embedding caches (keep recent ones for reuse)
echo "Removing old embedding caches (>7 days)..."
find /data/user_data/sanidhyv/tmp/embedding_cache -maxdepth 1 -type d -mtime +7 -exec rm -rf {} + 2>/dev/null
echo "Embedding cache cleanup complete"

# Remove orphaned lock files
echo "Removing orphaned lock files..."
find /data/user_data/sanidhyv/tmp/embedding_cache -name ".lock" -mtime +1 -delete 2>/dev/null
echo "Lock file cleanup complete"

# Clean Ray temp files
echo "Cleaning Ray temp files..."
find /data/user_data/sanidhyv/ray_temp_grep -type f -name "*.log" -mtime +3 -delete 2>/dev/null
echo "Ray cleanup complete"

echo "Cleanup complete!"
echo "Cleaning embedding cache older than ${MAX_AGE_DAYS} days..."
find "$CACHE_DIR" -type d -mtime +${MAX_AGE_DAYS} -exec rm -rf {} +
echo "Done!"
exit $?
