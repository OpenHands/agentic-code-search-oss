"""
Modal GPU Training Script for Agentic Code Search

This script runs RL training on Modal's cloud GPUs with:
- Persistent volumes for checkpoints and training data
- Secrets for WandB logging and HuggingFace model access
- Multi-GPU support (H100 or A100)

Usage:
    # First time setup
    modal secret create wandb-secret WANDB_API_KEY=<your-key>
    modal secret create huggingface-secret HF_TOKEN=<your-token>
    modal run scripts/modal_train.py::upload_data

    # Validate setup (no GPU needed)
    modal run scripts/modal_train.py::validate_setup

    # Run training
    modal run --detach scripts/modal_train.py --model Qwen/Qwen3-4B --timeout-hours 24

    # View logs
    modal app logs agentic-code-search-training
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# --- Constants ---
MINUTES = 60
HOURS = 60 * MINUTES

# --- Modal App ---
app = modal.App("agentic-code-search-training")

# --- Volumes for persistent storage ---
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

data_volume = modal.Volume.from_name("code-search-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("code-search-checkpoints", create_if_missing=True)

# --- Custom image with all dependencies ---
# Using CUDA 12.4 devel image for full CUDA toolkit (needed for flash-attn)
# Note: We use copy=True for local files because we need to run uv sync after adding them
training_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.13",
    )
    .entrypoint([])  # Remove verbose NVIDIA logging
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "tmux",
        "wget",
    )
    .pip_install("uv")
    # Add project files (copy=True required since we run commands after)
    .add_local_dir("src", remote_path="/app/src", copy=True)
    .add_local_dir("configs", remote_path="/app/configs", copy=True)
    .add_local_dir("software-agent-sdk", remote_path="/app/software-agent-sdk", copy=True)
    .add_local_dir("prime-rl", remote_path="/app/prime-rl", copy=True)
    .add_local_file("pyproject.toml", remote_path="/app/pyproject.toml", copy=True)
    .add_local_file("uv.lock", remote_path="/app/uv.lock", copy=True)
    .add_local_file("swe_grep_oss_env.py", remote_path="/app/swe_grep_oss_env.py", copy=True)
    # Install dependencies using uv
    .run_commands(
        "cd /app && uv sync --frozen --no-dev",
    )
    .env(
        {
            "VLLM_FLASH_ATTN_VERSION": "2",
            "CUDA_LAUNCH_BLOCKING": "1",
            "TORCH_USE_CUDA_DSA": "1",
        }
    )
)


def get_num_gpus() -> int:
    """Get the number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip().split("\n"))
    except Exception:
        return 4  # Default fallback


def build_training_command(
    model: str,
    n_rollouts: int,
    batch_size: int,
    max_length: int,
    num_gpus: int,
    extra_args: str,
) -> list[str]:
    """Build the training command with all parameters."""
    model_alias = model.replace("/", "-")
    run_name = f"code_search_{model_alias}"

    # Split GPUs between training and inference
    half_gpus = max(1, num_gpus // 2)
    num_training_engines = half_gpus
    num_inference_engines = half_gpus

    # Scale num_parallel_generation_workers based on batch size
    # Constraint: batch_size <= workers <= batch_size * (max_staleness_steps + 1)
    # With max_staleness_steps=4, max_workers = batch_size * 5
    max_staleness_steps = 4
    num_parallel_workers = min(16, batch_size * (max_staleness_steps + 1))
    num_parallel_workers = max(batch_size, num_parallel_workers)  # At least batch_size

    cmd = [
        "uv",
        "run",
        "--isolated",
        "-m",
        "src.train",
        "+run_async_trainer=true",
        f"data.train_data=['{DATA_PATH}/train.parquet']",
        f"data.val_data=['{DATA_PATH}/validation.parquet']",
        "trainer.algorithm.advantage_estimator=grpo",
        f"trainer.policy.model.path={model}",
        "trainer.placement.colocate_all=false",
        "trainer.placement.colocate_policy_ref=true",
        "trainer.strategy=fsdp2",
        "trainer.policy.fsdp_config.cpu_offload=true",
        "trainer.policy.fsdp_config.reshard_after_forward=true",
        "trainer.policy.fsdp_config.fsdp_size=-1",
        f"trainer.fully_async.num_parallel_generation_workers={num_parallel_workers}",
        f"trainer.placement.policy_num_gpus_per_node={num_training_engines}",
        f"trainer.placement.ref_num_gpus_per_node={num_training_engines}",
        "trainer.placement.policy_num_nodes=1",
        "trainer.placement.ref_num_nodes=1",
        "trainer.policy.sequence_parallel_size=1",
        f"generator.num_inference_engines={num_inference_engines}",
        "generator.inference_engine_tensor_parallel_size=1",
        f"+generator.traj_dir={CHECKPOINTS_PATH}/trajectories/",
        "+generator.engine_init_kwargs.enable_auto_tool_choice=true",
        "+generator.engine_init_kwargs.tool_call_parser=hermes",
        "+generator.engine_init_kwargs.reasoning_parser=qwen3",
        "trainer.epochs=20",
        "trainer.eval_batch_size=100",
        "trainer.eval_before_train=false",
        "trainer.eval_interval=100",
        "trainer.update_epochs_per_batch=1",
        f"trainer.train_batch_size={batch_size}",
        f"trainer.policy_mini_batch_size={batch_size}",
        "trainer.micro_forward_batch_size_per_gpu=1",
        "trainer.micro_train_batch_size_per_gpu=1",
        "trainer.dump_data_batch=true",
        f"trainer.export_path={CHECKPOINTS_PATH}/exported_model/",
        "trainer.hf_save_interval=5",
        "trainer.ckpt_interval=5",
        "trainer.max_prompt_length=4096",
        f"generator.sampling_params.max_generate_length={max_length}",
        "generator.sampling_params.temperature=1.0",
        "generator.max_input_length=24000",
        "generator.max_num_batched_tokens=48000",
        "generator.max_turns=20",
        "trainer.policy.optimizer_config.lr=1.0e-6",
        "trainer.algorithm.use_kl_loss=False",
        "generator.backend=vllm",
        "generator.run_engines_locally=True",
        "generator.enable_http_endpoint=True",
        "generator.http_endpoint_host=0.0.0.0",
        "generator.http_endpoint_port=8080",
        "generator.weight_sync_backend=nccl",
        "generator.async_engine=true",
        "generator.batched=false",
        f"generator.n_samples_per_prompt={n_rollouts}",
        "generator.gpu_memory_utilization=0.75",
        "generator.enforce_eager=false",
        "trainer.step_wise_training=true",
        "trainer.logger=wandb",
        "trainer.project_name=code_search",
        f"trainer.run_name={run_name}",
        "trainer.resume_mode=latest",
        f"trainer.ckpt_path={CHECKPOINTS_PATH}",
        "trainer.max_ckpts_to_keep=3",
    ]

    # Add extra args if provided
    if extra_args:
        cmd.append(extra_args)

    return cmd


@app.function(
    image=training_image,
    gpu="H100:4",  # 4x H100 GPUs (configurable via with_options)
    volumes={
        DATA_PATH: data_volume,
        CHECKPOINTS_PATH: checkpoints_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=24 * HOURS,  # Max timeout allowed by Modal (24 hours)
)
def train(
    model: str = "Qwen/Qwen3-4B",
    n_rollouts: int = 8,
    batch_size: int = 8,
    max_length: int = 8192,
    extra_args: str = "",
) -> None:
    """
    Run the async training loop.

    Args:
        model: HuggingFace model path (e.g., Qwen/Qwen3-4B)
        n_rollouts: Number of rollouts per prompt
        batch_size: Training batch size
        max_length: Maximum generation length
        extra_args: Additional Hydra config overrides
    """
    # Reload volume to ensure latest data is available
    data_volume.reload()

    # Get number of GPUs
    num_gpus = get_num_gpus()
    print(f"Training with {num_gpus} GPUs")

    # Build and run the training command
    cmd = build_training_command(
        model=model,
        n_rollouts=n_rollouts,
        batch_size=batch_size,
        max_length=max_length,
        num_gpus=num_gpus,
        extra_args=extra_args,
    )

    print(f"Running command:\n{' '.join(cmd)}")

    # Run training
    result = subprocess.run(
        cmd,
        cwd="/app",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Commit checkpoint changes to volume
    checkpoints_volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    print("Training completed successfully!")


# --- Helper Functions ---


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("pyarrow"),
    volumes={DATA_PATH: data_volume},
)
def upload_data(
    train_path: str = "data/swe_gym/train.parquet",
    val_path: str = "data/swe_gym/validation.parquet",
):
    """
    Upload local training data to Modal Volume.

    This function should be called from your local machine to upload data files.
    The files are read from the local filesystem and written to the Modal Volume.

    Usage:
        modal run scripts/modal_train.py::upload_data
    """
    import shutil

    # Copy files to volume
    train_dest = DATA_PATH / "train.parquet"
    val_dest = DATA_PATH / "validation.parquet"

    # Note: Modal mounts local files, so we can copy from the mounted paths
    local_train = Path("/app") / train_path
    local_val = Path("/app") / val_path

    if local_train.exists():
        shutil.copy(local_train, train_dest)
        print(f"Uploaded {train_path} -> {train_dest}")
    else:
        print(f"Warning: {local_train} not found")

    if local_val.exists():
        shutil.copy(local_val, val_dest)
        print(f"Uploaded {val_path} -> {val_dest}")
    else:
        print(f"Warning: {local_val} not found")

    # Commit changes to volume
    data_volume.commit()

    print(f"\nData files in volume: {list(DATA_PATH.iterdir())}")


# Lightweight image for upload that includes local data
upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow")
    .add_local_dir("data", remote_path="/app/data")
)


@app.function(
    image=upload_image,
    volumes={DATA_PATH: data_volume},
)
def upload_data_from_local():
    """
    Upload training data from the local data/ directory to Modal Volume.

    Usage:
        modal run scripts/modal_train.py::upload_data_from_local
    """
    import shutil

    local_data = Path("/app/data/swe_gym")
    train_file = local_data / "train.parquet"
    val_file = local_data / "validation.parquet"

    if train_file.exists():
        shutil.copy(train_file, DATA_PATH / "train.parquet")
        print(f"Uploaded train.parquet ({train_file.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"Error: {train_file} not found!")
        return

    if val_file.exists():
        shutil.copy(val_file, DATA_PATH / "validation.parquet")
        print(f"Uploaded validation.parquet ({val_file.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"Error: {val_file} not found!")
        return

    data_volume.commit()
    print("\n✅ Data uploaded successfully!")
    print(f"Files in volume: {list(DATA_PATH.iterdir())}")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={DATA_PATH: data_volume, CHECKPOINTS_PATH: checkpoints_volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def validate_setup():
    """
    Validate that volumes, secrets, and data are configured correctly.

    This function runs on a cheap CPU container to verify your setup
    before spending money on GPU time.

    Usage:
        modal run scripts/modal_train.py::validate_setup
    """
    print("=" * 50)
    print("Validating Modal Setup for Training")
    print("=" * 50)

    errors = []

    # Check secrets
    print("\n1. Checking secrets...")
    wandb_key = os.environ.get("WANDB_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")

    if wandb_key:
        print(f"   ✓ WANDB_API_KEY is set (length: {len(wandb_key)})")
    else:
        errors.append(
            "WANDB_API_KEY not set! Run: modal secret create wandb-secret WANDB_API_KEY=<key>"
        )
        print("   ✗ WANDB_API_KEY not set")

    if hf_token:
        print(f"   ✓ HF_TOKEN is set (length: {len(hf_token)})")
    else:
        errors.append(
            "HF_TOKEN not set! Run: modal secret create huggingface-secret HF_TOKEN=<token>"
        )
        print("   ✗ HF_TOKEN not set")

    # Check data volume
    print("\n2. Checking data volume...")
    train_file = DATA_PATH / "train.parquet"
    val_file = DATA_PATH / "validation.parquet"

    if train_file.exists():
        size_mb = train_file.stat().st_size / 1e6
        print(f"   ✓ train.parquet exists ({size_mb:.2f} MB)")
    else:
        errors.append(
            f"Missing {train_file}! Run: modal run scripts/modal_train.py::upload_data_from_local"
        )
        print("   ✗ train.parquet not found")

    if val_file.exists():
        size_mb = val_file.stat().st_size / 1e6
        print(f"   ✓ validation.parquet exists ({size_mb:.2f} MB)")
    else:
        errors.append(
            f"Missing {val_file}! Run: modal run scripts/modal_train.py::upload_data_from_local"
        )
        print("   ✗ validation.parquet not found")

    # Check checkpoints volume
    print("\n3. Checking checkpoints volume...")
    print(f"   ✓ Checkpoints path: {CHECKPOINTS_PATH}")
    if list(CHECKPOINTS_PATH.iterdir()):
        print(f"   ℹ Existing checkpoints: {list(CHECKPOINTS_PATH.iterdir())}")
    else:
        print("   ℹ No existing checkpoints (will be created during training)")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print("❌ Setup validation FAILED!")
        print("\nErrors to fix:")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")
        raise RuntimeError("Setup validation failed. See errors above.")
    else:
        print("✅ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("   modal run --detach scripts/modal_train.py --model Qwen/Qwen3-4B")


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-4B",
    n_rollouts: int = 8,
    batch_size: int = 8,
    max_length: int = 8192,
    extra_args: str = "",
):
    """
    Run training on Modal GPUs.

    Args:
        model: HuggingFace model path (e.g., Qwen/Qwen3-4B)
        n_rollouts: Number of rollouts per prompt
        batch_size: Training batch size
        max_length: Maximum generation length
        extra_args: Additional Hydra config overrides

    Note: GPU config (H100:4) and timeout (24h) are set in the @app.function decorator.
    To change these, edit the decorator in modal_train.py.
    """
    print("Starting training with:")
    print(f"  Model: {model}")
    print("  GPU: H100:4 (edit decorator to change)")
    print("  Timeout: 24 hours (Modal max)")
    print(f"  N Rollouts: {n_rollouts}")
    print(f"  Batch Size: {batch_size}")

    # Run training
    train.remote(
        model=model,
        n_rollouts=n_rollouts,
        batch_size=batch_size,
        max_length=max_length,
        extra_args=extra_args,
    )
