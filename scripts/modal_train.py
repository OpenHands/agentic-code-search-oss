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

MINUTES = 60
HOURS = 60 * MINUTES
GPU_CONFIG = "H100:8"  # Default GPU configuration
TIMEOUT = 1 * HOURS  # Default timeout for training runs
# --- Modal App ---
app = modal.App("agentic-code-search-training")

# --- Volumes for persistent storage ---
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

data_volume = modal.Volume.from_name("code-search-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("code-search-checkpoints", create_if_missing=True)

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
        "ripgrep"
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
        "cd /app && uv sync --all-extras",
    )
    .env(
        {
            "VLLM_FLASH_ATTN_VERSION": "2",
            # "CUDA_LAUNCH_BLOCKING": "1", #TODO: why are these needed?
            # "TORCH_USE_CUDA_DSA": "1",
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

@app.function(
    image=training_image,
    gpu=GPU_CONFIG,  # 8x H100 GPUs (configurable via with_options)
    volumes={
        DATA_PATH: data_volume,
        CHECKPOINTS_PATH: checkpoints_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    timeout=TIMEOUT,
)
def train(
    train_script: str = "/data/scripts/run_async_training.sh",
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    run_name: str = "",  # Experiment name (defaults to model alias)
    n_rollouts: int = 8,
    batch_size: int = 8,
    micro_batch_size: int = 1,
    step_wise: bool = False,
    num_inference_gpus: int = 4,
    num_train_gpus: int = 4,
    data_path: str = "data/adityasoni17__SWE-smith-py-code-search_train/",
    max_length: int = 8192,
    max_steps: int = 0,  # 0 = no limit, otherwise limit training steps
    fresh: bool = False,  # Start fresh training, ignoring previous checkpoints
    extra_args: str = "+generator.reward=custom_config.yaml"
) -> None:
    """
    Run the async training loop.

    Args:
    train_script (str): Path to the training script.
    """
    # Reload volume to ensure latest data is available
    data_volume.reload()
    print(f"Files in volume: {list(DATA_PATH.iterdir())} {list((DATA_PATH/'scripts').iterdir())}")
    if not run_name:
        run_name = model.replace("/", "-")
    ckpt_path = f"{CHECKPOINTS_PATH}/{run_name}"

    # Get number of GPUs
    num_gpus = get_num_gpus()
    print(f"Training with {num_gpus} GPUs")
    assert num_gpus == (num_inference_gpus + num_train_gpus), \
        f"num_gpus ({num_gpus}) must equal num_inference_gpus ({num_inference_gpus}) + num_train_gpus ({num_train_gpus}). Don't under-use or over-use GPUs!"

    cmd = f"bash {train_script} \
-m {model} \
-n {n_rollouts} \
-b {batch_size} \
-c {micro_batch_size} \
-r {run_name} \
-w {step_wise} \
-s {ckpt_path} \
-i {num_inference_gpus} \
-t {num_train_gpus} \
-d {data_path} \
-o \"{extra_args}\""

    print(f"Running command:\n{cmd}")

    # Run training
    result = subprocess.run(
        cmd,
        cwd="/app",
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=True
    )

    # Commit checkpoint changes to volume
    checkpoints_volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    print("Training completed successfully!")

@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={CHECKPOINTS_PATH: checkpoints_volume},
)
def check_run_exists(run_name: str) -> dict:
    """
    Check if a run already exists in the checkpoints volume.

    Returns dict with:
        - exists: bool
        - has_checkpoints: bool
        - checkpoint_count: int
        - has_trajectories: bool
    """
    run_path = Path(CHECKPOINTS_PATH) / run_name

    result = {
        "exists": run_path.exists(),
        "has_checkpoints": False,
        "checkpoint_count": 0,
        "has_trajectories": False,
    }

    if run_path.exists():
        # Check for checkpoint directories (global_step_*)
        checkpoints = list(run_path.glob("global_step_*"))
        result["has_checkpoints"] = len(checkpoints) > 0
        result["checkpoint_count"] = len(checkpoints)

        # Check for trajectories
        traj_path = run_path / "trajectories"
        result["has_trajectories"] = traj_path.exists() and any(traj_path.iterdir())

    return result

@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={CHECKPOINTS_PATH: checkpoints_volume},
)
def list_runs() -> None:
    """List all existing run names in the checkpoints volume."""
    ckpt_path = Path(CHECKPOINTS_PATH)
    if not ckpt_path.exists():
        print("No runs found (checkpoints volume is empty)")
        return

    runs = [d.name for d in ckpt_path.iterdir() if d.is_dir()]
    if not runs:
        print("No runs found")
        return

    print(f"Found {len(runs)} run(s):")
    for run in sorted(runs):
        run_path = ckpt_path / run
        checkpoints = list(run_path.glob("global_step_*"))
        traj_path = run_path / "trajectories"
        has_traj = traj_path.exists() and any(traj_path.iterdir()) if traj_path.exists() else False
        print(f"  - {run} ({len(checkpoints)} checkpoints, trajectories: {has_traj})")

@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("pyarrow"),
    volumes={DATA_PATH: data_volume},
)
def upload_data(
    train_path: str = "data/adityasoni17__SWE-smith-py-code-search_train/train.parquet",
    val_path: str = "data/adityasoni17__SWE-smith-py-code-search_train/validation.parquet",
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
    local_train = Path(train_path)
    local_val = Path(val_path)

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

    
    scripts_dest = DATA_PATH / "scripts"
    local_scripts = Path("scripts/")
    if local_scripts.exists() and local_scripts.is_dir():
        shutil.copytree(local_scripts, scripts_dest, dirs_exist_ok=True)
        print(f"Uploaded scripts/ -> {scripts_dest}")
    else:
        print(f"Warning: {local_scripts} not found")
    
    custom_config_dest = DATA_PATH / "custom_config.yaml"
    local_custom_config = Path("custom_config.yaml")
    if local_custom_config.exists():
        shutil.copy(local_custom_config, custom_config_dest)
        print(f"Uploaded custom_config.yaml -> {custom_config_dest}")
    else:
        print(f"Warning: {local_custom_config} not found")

    # Commit changes to volume
    data_volume.commit()

    print(f"\nData files in volume: {list(DATA_PATH.iterdir())}")

upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow")
    .add_local_dir("data", remote_path="/app/data")
    .add_local_dir("scripts/", remote_path="/app/scripts")
    .add_local_file("custom_config.yaml", remote_path="/app/custom_config.yaml")
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

    local_data = Path("/app/data/adityasoni17__SWE-smith-py-code-search_train")
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
    
    # copy scripts/
    local_scripts = Path("/app/scripts/")
    scripts_dest = DATA_PATH / "scripts"
    if local_scripts.exists() and local_scripts.is_dir():
        shutil.copytree(local_scripts, scripts_dest, dirs_exist_ok=True)
        print(f"Uploaded scripts/ -> {scripts_dest}")
    else:
        print(f"Warning: {local_scripts} not found")

    # copy custom_config.yaml
    local_custom_config = Path("/app/custom_config.yaml")
    custom_config_dest = DATA_PATH / "custom_config.yaml"
    if local_custom_config.exists():
        shutil.copy(local_custom_config, custom_config_dest)
        print(f"Uploaded custom_config.yaml -> {custom_config_dest}")

    data_volume.commit()
    print("\n✅ Data uploaded successfully!")
    print(f"Files in volume: {list(DATA_PATH.iterdir())} {list((DATA_PATH/'scripts').iterdir())}")

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


    # Check scripts/ and custom_config.yaml in data volume
    print("\n4. Checking scripts/ and custom_config.yaml in data volume...")
    scripts_path = DATA_PATH / "scripts"
    custom_config_path = DATA_PATH / "custom_config.yaml"
    if scripts_path.exists() and any(scripts_path.iterdir()):
        print(f"   ✓ scripts/ directory exists with files")
    else:
        errors.append(
            f"Missing scripts/ directory in data volume! Run: modal run scripts/modal_train.py::upload_data"
        )
        print("   ✗ scripts/ directory not found or empty")

    if custom_config_path.exists():
        print(f"   ✓ custom_config.yaml exists")
    else:
        errors.append(
            f"Missing custom_config.yaml in data volume! Run: modal run scripts/modal_train.py::upload_data"
        )
        print("   ✗ custom_config.yaml not found")

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
    train_script: str = f"{DATA_PATH}/scripts/run_async_training.sh",    
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    run_name: str = "Qwen3-4b-custom-finish-tool-seq-tis",
    n_rollouts: int = 8,
    batch_size: int = 8,
    micro_batch_size: int = 1,
    step_wise: bool = False,
    num_inference_gpus: int = 4,
    num_train_gpus: int = 4,
    data_path: str = f"{DATA_PATH}/",
    max_length: int = 8192,
    max_steps: int = 0,  # 0 = no limit, otherwise limit training steps
    fresh: bool = False,  # Start fresh training, ignoring previous checkpoints
    extra_args: str = f"+generator.reward={DATA_PATH}/custom_config.yaml"
):
    """
    Run training on Modal GPUs.

    Args:
        model: HuggingFace model path (e.g., Qwen/Qwen3-4B)
        run_name: Experiment name for organizing checkpoints (defaults to model alias)
        n_rollouts: Number of rollouts per prompt
        batch_size: Training batch size
        max_length: Maximum generation length
        max_steps: Limit training to N steps (0 = no limit, use for quick tests)
        fresh: Start fresh training, ignoring previous checkpoints
        force: Skip confirmation prompts
        extra_args: Additional Hydra config overrides

    Note: GPU config (H100:4) and timeout (24h) are set in the @app.function decorator.
    To change these, edit the decorator in modal_train.py.
    """
    # Generate run_name from model if not provided
    if not run_name:
        run_name = model.replace("/", "-")

    print(f"\n{'=' * 60}")
    print("Training Configuration")
    print(f"{'=' * 60}")
    print(f"  Run Name: {run_name}")
    print(f"  Model: {model}")
    print(f"  GPU: {GPU_CONFIG} (edit GPU_CONFIG to change)")
    print(f"  Timeout: {TIMEOUT} (Modal max)")
    print(f"  N Rollouts: {n_rollouts}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Max Steps: {max_steps if max_steps > 0 else 'unlimited'}")
    print(f"  Fresh: {fresh}")
    print(f"{'=' * 60}\n")

    # Check if run already exists
    print("Checking if run exists...")
    run_info = check_run_exists.remote(run_name)

    if run_info["exists"]:
        print(f"\n⚠️  Run '{run_name}' already exists!")
        print(f"   Checkpoints: {run_info['checkpoint_count']}")
        print(f"   Has trajectories: {run_info['has_trajectories']}")

        if fresh:
            print("\n   --fresh is set: will start fresh training (ignoring existing checkpoints)")
        else:
            print("\n   Will attempt to resume from latest checkpoint")
    else:
        print(f"✅ New run: '{run_name}'")

    print("\nStarting training...")

    # Run training
    train.remote(
        train_script=train_script,
        model=model,
        run_name=run_name,
        n_rollouts=n_rollouts,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        step_wise=step_wise,
        num_inference_gpus=num_inference_gpus,
        num_train_gpus=num_train_gpus,
        data_path=data_path,
        max_length=max_length,
        max_steps=max_steps,
        fresh=fresh,
        extra_args=extra_args
    )