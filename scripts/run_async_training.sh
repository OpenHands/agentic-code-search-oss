#!/bin/bash
#SBATCH --job-name=cso
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=750G
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=babel-q5-28,babel-o5-20,babel-n5-28,babel-q5-24,babel-p5-20,babel-q5-20,babel-q5-32,babel-o9-20,babel-q9-28,babel-s9-16,babel-t5-24,babel-p5-32

. .env

while getopts ":m:n:d:s:o:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    n ) N_ROLLOUTS=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    s ) CKPT_PATH=$OPTARG;;
    o ) OTHER_OPTION=$OPTARG;;
    # \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
# Get number of GPUs available
NUM_GPUS=$(nvidia-smi -L | wc -l)
N_ROLLOUTS="${N_ROLLOUTS:-8}"
BATCH_SIZE=4
MAX_LENGTH=8192
RUN_NAME="code_search_${MODEL_ALIAS}"
set -x

DATA_PATH="${DATA_PATH:-data/swe_smith}"
CKPT_PATH="${CKPT_PATH:-ckpts/${MODEL_ALIAS}}"
mkdir -p $CKPT_PATH

NNODES=1
NUM_INFERENCE_ENGINES=4
TP_SIZE=1
LOGGER=wandb

uv run --isolated --frozen -m src.train \
  +run_async_trainer=true \
  data.train_data="['$DATA_PATH/train.parquet']" \
  data.val_data="['$DATA_PATH/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=${MODEL} \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.policy.fsdp_config.reshard_after_forward=true \
  trainer.policy.fsdp_config.fsdp_size=-1 \
  trainer.fully_async.num_parallel_generation_workers=20 \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.policy.sequence_parallel_size=1 \
  generator.num_inference_engines=1 \
  generator.inference_engine_tensor_parallel_size=1 \
  +generator.traj_dir=$CKPT_PATH/trajectories/ \
  +generator.engine_init_kwargs="{enable_auto_tool_choice:true,tool_call_parser:hermes,reasoning_parser:qwen3}" \
  trainer.epochs=20 \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=${BATCH_SIZE} \
  trainer.policy_mini_batch_size=${BATCH_SIZE} \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=${MAX_LENGTH} \
  generator.max_input_length=24000 \
  generator.max_num_batched_tokens=48000 \
  generator.max_turns=20 \
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
  trainer.logger="$LOGGER" \
  trainer.project_name="code_search" \
  trainer.run_name=${RUN_NAME} \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$CKPT_PATH" \
  $OTHER_OPTION