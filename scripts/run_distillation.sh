#!/bin/bash
#
# Usage: bash scripts/run_distillation.sh \
#   -m Qwen/Qwen3-4B \              # Student model (model to be trained)
#   -r Qwen/Qwen3-32B \             # Reference/Teacher model (model to distill from)
#   -d data/swe_gym \               # Data path
#   [-s ckpt_path] [-n n_rollouts] [-i num_inference_engines] [-t num_training_engines]
#

. .env 2>/dev/null || true

while getopts ":m:r:n:d:s:o:i:t:b:" opt; do
  case ${opt} in
    m ) STUDENT_MODEL=$OPTARG;;      # -m: Student model (model to be trained)
    r ) TEACHER_MODEL=$OPTARG;;      # -r: Reference/Teacher model (model to distill from)
    n ) N_ROLLOUTS=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    s ) CKPT_PATH=$OPTARG;;
    o ) OTHER_OPTION=$OPTARG;;
    i ) NUM_INFERENCE_ENGINES=$OPTARG;;
    t ) NUM_TRAINING_ENGINES=$OPTARG;;
    b ) MICRO_BATCH_SIZE=$OPTARG;;
    \? ) echo "Usage: $0 -m <student_model> -r <teacher_model> [-d data_path] [-s ckpt_path] [-n n_rollouts] [-i num_inference_engines] [-t num_training_engines] [-b micro_batch_size] [-o other_options]"; exit 1;;
  esac
done

# Validate required parameters
if [ -z "$STUDENT_MODEL" ]; then
    echo "Error: Student model (-m) is required"
    echo "Usage: $0 -m <student_model> -r <teacher_model> -d <data_path>"
    exit 1
fi
if [ -z "$TEACHER_MODEL" ]; then
    echo "Error: Teacher model (-r) is required"
    echo "Usage: $0 -m <student_model> -r <teacher_model> -d <data_path>"
    exit 1
fi

STUDENT_MODEL_ALIAS=$(echo $STUDENT_MODEL | sed 's/\//-/g')
TEACHER_MODEL_ALIAS=$(echo $TEACHER_MODEL | sed 's/\//-/g')
# Get number of GPUs available
NUM_GPUS=$(nvidia-smi -L | wc -l)
N_ROLLOUTS="${N_ROLLOUTS:-8}"
BATCH_SIZE=16  # Must be <= num_parallel_generation_workers (set to 16 below)
MAX_LENGTH=8192
RUN_NAME="code_search_distillation_${STUDENT_MODEL_ALIAS}_${TEACHER_MODEL_ALIAS}"
set -x

DATA_PATH="${DATA_PATH:-data/swe_smith}"
CKPT_PATH="${CKPT_PATH:-$(pwd)/ckpts/${STUDENT_MODEL_ALIAS}}"
mkdir -p $CKPT_PATH

HALF_NUM_GPUS=$((NUM_GPUS / 2))
NUM_INFERENCE_ENGINES="${NUM_INFERENCE_ENGINES:-$NUM_GPUS}"
NUM_TRAINING_ENGINES="${NUM_TRAINING_ENGINES:-$NUM_GPUS}"

export VLLM_FLASH_ATTN_VERSION=2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export RAY_worker_register_timeout_seconds=600


uv run python -m src.train \
  +run_async_trainer=false \
  +use_distillation=true \
  data.train_data="['$DATA_PATH/train.parquet']" \
  data.val_data="['$DATA_PATH/validation.parquet']" \
  trainer.algorithm.advantage_estimator="no_op" \
  trainer.algorithm.policy_loss_type="importance_sampling" \
  trainer.algorithm.use_kl_in_reward=true \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path=${STUDENT_MODEL} \
  trainer.ref.model.path=${TEACHER_MODEL} \
  trainer.placement.colocate_all=true \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  trainer.policy.fsdp_config.fsdp_size=-1 \
  trainer.fully_async.num_parallel_generation_workers=16 \
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
  +generator.engine_init_kwargs.reasoning_parser=qwen3 \
  trainer.epochs=20 \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${BATCH_SIZE} \
  trainer.policy_mini_batch_size=${BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=${MICRO_BATCH_SIZE:-1} \
  trainer.dump_data_batch=true \
  trainer.export_path="${CKPT_PATH}exported_model/" \
  trainer.hf_save_interval=5 \
  trainer.ckpt_interval=5 \
  trainer.max_prompt_length=32768 \
  generator.sampling_params.max_generate_length=${MAX_LENGTH} \
  generator.sampling_params.temperature=1.0 \
  generator.max_input_length=32768 \
  generator.max_num_batched_tokens=131072 \
  generator.max_turns=6 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=False \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host='0.0.0.0' \
  generator.http_endpoint_port=8080 \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=${N_ROLLOUTS} \
  generator.gpu_memory_utilization=0.75 \
  generator.enforce_eager=false \
  trainer.step_wise_training=false \
  trainer.logger="wandb" \
  trainer.project_name="code_search" \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.max_ckpts_to_keep=3 \
  $OTHER_OPTION
