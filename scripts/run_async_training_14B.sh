#!/bin/bash
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu_qos
#SBATCH --account=gneubig
#SBATCH --job-name=cso
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.out
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH --mem=1500G
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=orchard-flame-16

# export REWARD=file_loc
# sbatch scripts/run_async_training.sh \
#   -m Qwen/Qwen3-8B -n 8 -b 1 -i 4 -t 4 \
#   -d data/swe_gym \
#   -s /project/flame/lsutawik/cso/ckpts/qwen3-8b-8x8-${REWARD}/ \
#   -o "+generator.reward=configs/rewards/${REWARD}.yaml"

. .env

while getopts ":m:n:d:s:l:o:i:t:b:c:r:w:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    n ) N_ROLLOUTS=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    s ) CKPT_PATH=$OPTARG;;
    l ) LCAL_PATH=$OPTARG;;
    o ) OTHER_OPTION=$OPTARG;;
    i ) NUM_INFERENCE_ENGINES=$OPTARG;;
    t ) NUM_TRAINING_ENGINES=$OPTARG;;
    b ) BATCH_SIZE=$OPTARG;;
    c ) MICRO_BATCH_SIZE=$OPTARG;;
    r ) RUN_NAME=$OPTARG;;
    w ) STEP_WISE=$OPTARG;;
    # \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
# Get number of GPUs available
NUM_GPUS=$(nvidia-smi -L | wc -l)
N_ROLLOUTS="${N_ROLLOUTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH=8192
RUN_NAME="${RUN_NAME:-${MODEL_ALIAS}-${BATCH_SIZE}x${N_ROLLOUTS}}"
set -x

DATA_PATH="${DATA_PATH:-data/swe_smith}"
CKPT_PATH="${CKPT_PATH:-$(pwd)/ckpts/${MODEL_ALIAS}}"
# If LCAL_PATH is not set, use CKPT_PATH
LCAL_PATH="${LCAL_PATH:-$CKPT_PATH}"
mkdir -p $CKPT_PATH

HALF_NUM_GPUS=$((NUM_GPUS / 2))
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-$HALF_NUM_GPUS}"
NUM_TRAINING_ENGINES="${NUM_TRAINING_ENGINES:-$HALF_NUM_GPUS}"
STEP_WISE="${STEP_WISE:-false}"

export VLLM_FLASH_ATTN_VERSION=2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export RAY_worker_register_timeout_seconds=600

uv run --isolated -m src.train \
  +run_async_trainer=true \
  data.train_data="['$DATA_PATH/train.parquet']" \
  data.val_data="['$DATA_PATH/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.policy.model.path=${MODEL} \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  trainer.policy.fsdp_config.fsdp_size=-1 \
  trainer.fully_async.num_parallel_generation_workers=${BATCH_SIZE} \
  trainer.placement.policy_num_gpus_per_node=${NUM_TRAINING_ENGINES} \
  trainer.placement.ref_num_gpus_per_node=${NUM_TRAINING_ENGINES} \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.policy.sequence_parallel_size=1 \
  generator.num_inference_engines=${NUM_INFERENCE_ENGINES} \
  generator.inference_engine_tensor_parallel_size=1 \
  +generator.traj_dir=${CKPT_PATH}trajectories/ \
  +generator.engine_init_kwargs.enable_auto_tool_choice=true \
  +generator.engine_init_kwargs.tool_call_parser=hermes \
  +generator.engine_init_kwargs.rope_scaling.rope_type=yarn \
  +generator.engine_init_kwargs.rope_scaling.factor=2.0 \
  +generator.engine_init_kwargs.rope_scaling.original_max_position_embeddings=32768 \
  +generator.engine_init_kwargs.max_model_len=50_000 \
  +generator.engine_init_kwargs.disable_cascade_attn=true \
  generator.eval_n_samples_per_prompt=1 \
  trainer.epochs=10 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${BATCH_SIZE} \
  trainer.policy_mini_batch_size=${BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  trainer.dump_data_batch=true \
  trainer.export_path="${CKPT_PATH}exported_model/" \
  trainer.hf_save_interval=10 \
  trainer.ckpt_interval=100 \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=32768 \
  generator.sampling_params.max_generate_length=${MAX_LENGTH} \
  generator.sampling_params.temperature=1.0 \
  generator.max_input_length=32768 \
  generator.max_num_batched_tokens=131072 \
  generator.max_turns=6 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=False \
  trainer.algorithm.use_kl_in_reward=False \
  trainer.algorithm.policy_loss_type="gspo" \
  trainer.algorithm.eps_clip_low=3e-4 \
  trainer.algorithm.eps_clip_high=4e-4 \
  trainer.algorithm.loss_reduction="sequence_mean" \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host='0.0.0.0' \
  generator.http_endpoint_port=8080 \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=${N_ROLLOUTS} \
  generator.gpu_memory_utilization=0.8 \
  generator.enforce_eager=false \
  trainer.step_wise_training=${STEP_WISE} \
  trainer.logger="wandb" \
  trainer.project_name="code_search" \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$LCAL_PATH" \
  trainer.max_ckpts_to_keep=-1 \
  $OTHER_OPTION